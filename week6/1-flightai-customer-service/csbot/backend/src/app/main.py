from botocore.exceptions import ClientError
from fastapi import Cookie, FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from strands import Agent, tool
from strands.agent.conversation_manager import SlidingWindowConversationManager
from trip import FullTrip, Trip
from flight import Flight, PaymentStatus, TicketType
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import boto3
import json
import logging
import os
import requests
import uuid
import uvicorn

model_id = os.environ.get("MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0")
state_bucket = os.environ.get("STATE_BUCKET", "")
logging.getLogger("strands").setLevel(logging.WARNING)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s", 
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Logger initialized")
s3_client = boto3.client("s3")
conversation_manager = SlidingWindowConversationManager(
    window_size=4,  # Maximum number of messages to keep
    should_truncate_results=True, # Enable truncating the tool result when a message is too large for the model's context window 
)
system_prompt = """
You are a customer service assistant for FlightAI. You should be polite, reserved and helpful.

You can discuss customer's trips and explain our policies.
You can also give more general flight related tips from your knowledge.

A trip is made up of one or more flights. You can look up trips using `list_trips`, and find details
for a specific trips using the `flights_for_trip` function.

Refunds are only available on tickets labelled 'Economy fully refundable'. Refunds are not allowed on 'Basic Economy' tickets.
If the customer asks for a refund on a basic economy ticket you must politely but firmly decline.
If a ticket is 'Economy fully refundable' you should use the `cancel_flight` fuction to cancel it if requested.
"""
current_user_id = ""

@tool
def t_list_trips() -> list[dict]:
    """List trips for the user
    """

    logger.info(f"Fetching trips for user {current_user_id}")
    trips = Trip.list_for_user(current_user_id)
    logger.info(f"Found {len(trips)} trips for user {current_user_id}")
    return trips

@tool
def t_flights_for_trip(trip_id: str) -> FullTrip:
    """Get all flights for a given trip

    Args:
      trip_id: The ID of the trip to get flights for (starts T#)
    """
    logger.info(f"Fetching flights for trip {trip_id} for user {current_user_id}")
    return Trip.get_full_trip(current_user_id, trip_id)

@tool
def t_refund_flight(trip_id: str, flight_id: str) -> str:
    """Cancel a flight and refund the customer

    Args:
      trip_id: The ID of the trip the flight is associated with (starts T#)
      flight_id: The ID of the flight to refund (starts F#)
    """
    logger.info(f"Attempting to cancel flight {flight_id} for user {current_user_id}")
    Flight.update_payment_status(trip_id, flight_id, payment_status=PaymentStatus.REFUNDED)
    return f"Flight {flight_id} refunded"

class ChatRequest(BaseModel):
    prompt: str


class RenameTripRequest(BaseModel):
    name: str


def SaveHistory(agent: Agent, session_id: str):
    state = {
        "messages": agent.messages,
        "system_prompt": agent.system_prompt,
    }
    # Serialize the state to JSON
    state_json = json.dumps(state, indent=2)
    s3_key = f"sessions/{session_id}.json"
    try:
        # Upload to S3
        s3_client.put_object(
            Bucket=state_bucket,
            Key=s3_key,
            Body=state_json,
            ContentType="application/json"
        )
        logger.info(f"Successfully saved session {session_id} to S3")
    except Exception as e:
        logger.error(f"Failed to save session {session_id} to S3: {str(e)}")
        raise


def create_dummy_trips(user_id: str):
    """Creates some sample trips for a new user."""
    logger.info(f"Creating dummy trips for new user {user_id}")
    try:
        # Trip 1: Nana's 80th
        nana_trip = Trip(user_id=user_id, name="Nana's 80th")
        nana_trip_id = nana_trip.save()

        # Flights for Nana's 80th
        now_utc = datetime.now(timezone.utc)
        nana_departure = (now_utc - timedelta(weeks=12)).replace(hour=12, minute=0, second=0, microsecond=0)
        nana_return_departure = nana_departure + timedelta(days=7)

        flight1_nana = Flight(
            trip_id=nana_trip_id,
            from_airport="MUC",
            to_airport="ICN",
            departure_time=nana_departure,
            arrival_time=nana_departure + timedelta(hours=11),
            price=Decimal("800.00"),
            ticket_type=TicketType.BASIC_ECONOMY,
        )
        flight1_nana.save()

        flight2_nana = Flight(
            trip_id=nana_trip_id,
            from_airport="ICN",
            to_airport="MUC",
            departure_time=nana_return_departure,
            arrival_time=nana_return_departure + timedelta(hours=11),
            price=Decimal("850.00"),
            ticket_type=TicketType.BASIC_ECONOMY,
        )
        flight2_nana.save()
        logger.info(f"Created trip \"Nana's 80th\" for user {user_id}")

        # Trip 2: Summer weekend
        summer_trip = Trip(user_id=user_id, name="Summer weekend")
        summer_trip_id = summer_trip.save()

        target_year = now_utc.year if now_utc.month < 7 else now_utc.year + 1
        july_first = datetime(target_year, 7, 1, tzinfo=timezone.utc)
        days_to_friday = (4 - july_first.weekday()) % 7
        friday_date = july_first + timedelta(days=days_to_friday)
        monday_date = friday_date + timedelta(days=3)

        # Flights for Summer weekend
        flight1_summer = Flight(
            trip_id=summer_trip_id,
            from_airport="MUC",
            to_airport="NCE",
            departure_time=friday_date.replace(
                hour=18, minute=0, second=0, microsecond=0
            ),
            arrival_time=friday_date.replace(
                hour=19, minute=30, second=0, microsecond=0
            ),
            price=Decimal("250.00"),
            ticket_type=TicketType.BASIC_ECONOMY,
        )
        flight1_summer.save()

        flight2_summer = Flight(
            trip_id=summer_trip_id,
            from_airport="NCE",
            to_airport="MUC",
            departure_time=monday_date.replace(
                hour=20, minute=0, second=0, microsecond=0
            ),
            arrival_time=monday_date.replace(
                hour=21, minute=30, second=0, microsecond=0
            ),
            price=Decimal("275.00"),
            ticket_type=TicketType.BASIC_ECONOMY,
        )
        flight2_summer.save()
        logger.info(f"Created trip 'Summer weekend' for user {user_id}")

    except Exception as e:
        # Log error but don't prevent user session from being created
        logger.error(f"Failed to create dummy trips for user {user_id}: {e}")


def LoadHistory(session_id: str) -> Agent:
    global current_user_id
    current_user_id = session_id
    s3_key = f"sessions/{session_id}.json"
    try:
        response = s3_client.get_object(Bucket=state_bucket, Key=s3_key)
        state_json = response['Body'].read().decode('utf-8')
        state = json.loads(state_json)
        logger.info(f"Successfully loaded session {session_id} from S3")
        return Agent(
            model=model_id,
            messages=state.get("messages"),
            system_prompt=state.get("system_prompt"),
            conversation_manager=conversation_manager,
            tools=[t_list_trips, t_flights_for_trip, t_refund_flight],
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.info(f"Session {session_id} does not exist, creating new agent")
            create_dummy_trips(session_id)
            agent = Agent(
                model=model_id,
                system_prompt=system_prompt,
                conversation_manager=conversation_manager,
                tools=[t_list_trips, t_flights_for_trip, t_refund_flight],
            )
            SaveHistory(agent, session_id)
            return agent
        else:
            logger.error(f"Error loading session {session_id}: {e}")
            raise
    except Exception as e:
        logger.error(f"Unexpected error loading session {session_id}: {e}")
        raise

app = FastAPI()

# Called by the Lambda Adapter to check liveness
@app.get("/")
async def root():
    return {"message": "OK"}

@app.get('/api/chat')
def chat_history(request: Request):
    session_id = request.cookies.get("session_id", str(uuid.uuid4()))
    agent = LoadHistory(session_id)

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


@app.get('/api/trips')
def list_trips(request: Request):
    """Lists all trips for a user."""
    session_id = request.cookies.get("session_id", str(uuid.uuid4()))
    user_id = session_id

    try:
        trips = Trip.list_for_user(user_id)
        response = Response(
            content=json.dumps(trips),
            media_type="application/json",
        )
        response.set_cookie(key="session_id", value=session_id)
        return response
    except Exception as e:
        logger.error(f"Error listing trips for user {user_id}: {e}")
        response = Response(
            content=json.dumps({"error": "Failed to list trips"}),
            status_code=500,
            media_type="application/json",
        )
        response.set_cookie(key="session_id", value=session_id)
        return response


@app.post("/api/trips/{trip_id}/rename")
def rename_trip(trip_id: str, rename_request: RenameTripRequest, request: Request):
    """Renames a trip for a user."""
    session_id = request.cookies.get("session_id", str(uuid.uuid4()))
    user_id = session_id

    try:
        Trip.update_name(user_id, trip_id, rename_request.name)
        response = Response(
            content=json.dumps({"message": "Trip renamed successfully"}),
            media_type="application/json",
        )
        response.set_cookie(key="session_id", value=session_id)
        return response
    except ValueError as e:
        logger.warning(f"Error renaming trip {trip_id} for user {user_id}: {e}")
        status_code = 404 if "not found" in str(e).lower() else 400
        response = Response(
            content=json.dumps({"error": str(e)}),
            status_code=status_code,
            media_type="application/json",
        )
        response.set_cookie(key="session_id", value=session_id)
        return response
    except Exception as e:
        logger.error(f"Error renaming trip {trip_id} for user {user_id}: {e}")
        response = Response(
            content=json.dumps({"error": "Failed to rename trip"}),
            status_code=500,
            media_type="application/json",
        )
        response.set_cookie(key="session_id", value=session_id)
        return response


@app.get("/api/trip/{trip_id}")
def get_trip_details(trip_id: str, request: Request):
    """Gets all details for a specific trip."""
    session_id = request.cookies.get("session_id", str(uuid.uuid4()))
    user_id = session_id

    try:
        trip_details = Trip.get_full_trip(user_id, trip_id)
        response = Response(
            content=trip_details.model_dump_json(),
            media_type="application/json",
        )
        response.set_cookie(key="session_id", value=session_id)
        return response
    except ValueError as e:
        logger.warning(f"Error getting trip {trip_id} for user {user_id}: {e}")
        status_code = 404 if "not found" in str(e).lower() else 400
        response = Response(
            content=json.dumps({"error": str(e)}),
            status_code=status_code,
            media_type="application/json",
        )
        response.set_cookie(key="session_id", value=session_id)
        return response
    except Exception as e:
        logger.error(f"Error getting trip {trip_id} for user {user_id}: {e}")
        response = Response(
            content=json.dumps({"error": "Failed to get trip details"}),
            status_code=500,
            media_type="application/json",
        )
        response.set_cookie(key="session_id", value=session_id)
        return response


@app.post('/api/chat')
async def chat(chat_request: ChatRequest, request: Request):
    session_id = request.cookies.get("session_id", str(uuid.uuid4()))
    agent = LoadHistory(session_id)
    response = StreamingResponse(
        generate(agent, session_id, chat_request.prompt, request),
        media_type="text/event-stream"
    )
    response.set_cookie(key="session_id", value=session_id)
    return response

async def generate(agent: Agent, session_id: str, prompt: str, request: Request):
    generation_cancelled = False
    try:
        async for event in agent.stream_async(prompt):
            if await request.is_disconnected():
                generation_cancelled = True
                break
            if "complete" in event:
                logger.info("Response generation complete")
            if "data" in event:
                yield f"data: {json.dumps(event['data'])}\n\n"
        # Save history after streaming is complete
        if not generation_cancelled: # Don't save if the client disconnected before completion
            try:
                SaveHistory(agent, session_id)
            except Exception as e:
                logger.error(f"Failed to save history for session {session_id}: {str(e)}")
                # Don't re-raise to avoid breaking the response
 
    except Exception as e:
        error_message = json.dumps({"error": str(e)})
        yield f"event: error\ndata: {error_message}\n\n"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
