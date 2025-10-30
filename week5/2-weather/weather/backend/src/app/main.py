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

    encoded_city = urllib.parse.quote(city.strip())
    url = f"https://wttr.in/{encoded_city}?format=j1"

    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            data = response.read().decode(charset)
            # Ensure it's valid JSON; if not, wrap as error
            try:
                json.loads(data)
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
        tools=[get_weather],
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
