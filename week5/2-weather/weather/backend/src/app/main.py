from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from strands import Agent
from strands import tool
from strands.session.s3_session_manager import S3SessionManager
import boto3
import json
import logging
import os
import uuid
import uvicorn
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta

model_id = os.environ.get("MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0")
state_bucket = os.environ.get("STATE_BUCKET", "")
state_prefix = os.environ.get("STATE_PREFIX", "sessions/")
logging.getLogger("strands").setLevel(logging.WARNING)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s", 
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Logger initialized")

if not state_bucket:
    raise RuntimeError("STATE_BUCKET environment variable must be set")

if state_prefix and not state_prefix.endswith("/"):
    state_prefix = f"{state_prefix}/"

boto_session = boto3.Session()
s3_client = boto_session.client("s3")

# Cache configuration
weather_cache_prefix = os.environ.get("WEATHER_CACHE_PREFIX", "weather_cache/")
try:
    weather_cache_ttl_seconds = int(os.environ.get("WEATHER_CACHE_TTL_SECONDS", "600"))
except ValueError:
    weather_cache_ttl_seconds = 600


def _make_weather_cache_key(city: str) -> str:
    # Normalize and URL-encode for safe S3 key usage
    normalized = (city or "").strip().lower()
    encoded = urllib.parse.quote(normalized, safe="")
    prefix = weather_cache_prefix if weather_cache_prefix.endswith("/") else f"{weather_cache_prefix}/"
    return f"{prefix}{encoded}.json"


def _get_cached_weather(city: str) -> str | None:
    if not city:
        return None
    key = _make_weather_cache_key(city)
    try:
        head = s3_client.head_object(Bucket=state_bucket, Key=key)
        last_modified: datetime = head["LastModified"]
        # Convert to aware UTC and check TTL
        age = datetime.now(timezone.utc) - last_modified.astimezone(timezone.utc)
        if age > timedelta(seconds=weather_cache_ttl_seconds):
            return None
        obj = s3_client.get_object(Bucket=state_bucket, Key=key)
        data = obj["Body"].read().decode("utf-8")
        # ensure it looks like JSON
        json.loads(data)
        logger.info("Cache hit for city '%s'", city)
        return data
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception:
        # Any failure -> treat as cache miss
        return None


def _put_cached_weather(city: str, data: str) -> None:
    try:
        key = _make_weather_cache_key(city)
        s3_client.put_object(
            Bucket=state_bucket,
            Key=key,
            Body=data.encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as e:
        logger.warning("Failed to write weather cache for '%s': %s", city, str(e))


class ChatRequest(BaseModel):
    prompt: str


@tool
def get_weather(city: str) -> str:
    """get weather for a given city

    Args:
        city: The city to get the weather for.

    Returns:
        The current weather for the given city in JSON format.
    """
    if not city or not city.strip():
        return json.dumps({
            "error": "city is required"
        })

    # Check cache first
    cached = _get_cached_weather(city)
    if cached is not None:
        return cached

    encoded_city = urllib.parse.quote(city.strip())
    url = f"https://wttr.in/{encoded_city}?format=j1"

    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            data = response.read().decode(charset)
            # Ensure it's valid JSON; if not, wrap as error
            try:
                json.loads(data)
                _put_cached_weather(city, data)
                return data
            except json.JSONDecodeError:
                return json.dumps({
                    "error": "Non-JSON response from wttr.in",
                    "raw": data[:5000]
                })
    except Exception as e:
        return json.dumps({
            "error": f"Failed to fetch weather: {str(e)}"
        })


@tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius and Fahrenheit.

    Args:
        value: The temperature value to convert.
        from_unit: The input unit ("C", "Celsius", "F", or "Fahrenheit").
        to_unit: The desired output unit ("C", "Celsius", "F", or "Fahrenheit").

    Returns:
        JSON string with the converted value and normalized units, or error.
    """
    try:
        normalized_from = (from_unit or "").strip().lower()
        normalized_to = (to_unit or "").strip().lower()

        def normalize_unit(u: str) -> str | None:
            if u in {"c", "celsius"}:
                return "C"
            if u in {"f", "fahrenheit"}:
                return "F"
            return None

        src = normalize_unit(normalized_from)
        dst = normalize_unit(normalized_to)

        if src is None or dst is None:
            return json.dumps({
                "error": "Unsupported unit. Use C/Celsius or F/Fahrenheit.",
                "from_unit": from_unit,
                "to_unit": to_unit,
            })

        if src == dst:
            return json.dumps({
                "input": {"value": value, "unit": src},
                "output": {"value": value, "unit": dst}
            })

        if src == "C" and dst == "F":
            converted = (value * 9.0 / 5.0) + 32.0
        else:  # src == "F" and dst == "C"
            converted = (value - 32.0) * 5.0 / 9.0

        return json.dumps({
            "input": {"value": value, "unit": src},
            "output": {"value": converted, "unit": dst}
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

def create_agent(session_id: str) -> Agent:
    session_manager_kwargs = {
        "session_id": session_id,
        "bucket": state_bucket,
        "boto_session": boto_session,
    }
    if state_prefix:
        session_manager_kwargs["prefix"] = state_prefix

    session_manager = S3SessionManager(**session_manager_kwargs)
    agent = Agent(
        model=model_id,
        session_manager=session_manager,
        tools=[get_weather, convert_temperature],
    )
    logger.info("Agent initialized for session %s", session_id)
    return agent

app = FastAPI()

# Called by the Lambda Adapter to check liveness
@app.get("/")
async def root():
    return {"message": "OK"}

@app.get('/chat')
def chat_history(request: Request):
    session_id = request.cookies.get("session_id", str(uuid.uuid4()))
    agent = create_agent(session_id)

    # Filter messages to only include first text content
    filtered_messages = []
    for message in agent.messages:
        if (message.get("content") and 
            len(message["content"]) > 0 and 
            "text" in message["content"][0]):
            filtered_messages.append({
                "role": message["role"],
                "content": [{
                    "text": message["content"][0]["text"]
                }]
            })
 
    response = Response(
        content = json.dumps({
            "messages": filtered_messages,
        }),
        media_type="application/json",
    )
    response.set_cookie(key="session_id", value=session_id)
    return response

@app.post('/chat')
async def chat(chat_request: ChatRequest, request: Request):
    session_id = request.cookies.get("session_id", str(uuid.uuid4()))
    agent = create_agent(session_id)
    response = StreamingResponse(
        generate(agent, session_id, chat_request.prompt, request),
        media_type="text/event-stream"
    )
    response.set_cookie(key="session_id", value=session_id)
    return response

async def generate(agent: Agent, session_id: str, prompt: str, request: Request):
    try:
        async for event in agent.stream_async(prompt):
            if await request.is_disconnected():
                logger.info("Client disconnected before completion for session %s", session_id)
                break
            if "complete" in event:
                logger.info("Response generation complete")
            if "data" in event:
                yield f"data: {json.dumps(event['data'])}\n\n"
 
    except Exception as e:
        error_message = json.dumps({"error": str(e)})
        yield f"event: error\ndata: {error_message}\n\n"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
