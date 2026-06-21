# Session management endpoints

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.database.mongodb import mongodb
import uuid

router = APIRouter(prefix="/session", tags=["Session"])


# ── Request/Response Models ──

class StartSessionRequest(BaseModel):
    language: str = "en"
    traveler_type: str = None


class StartSessionResponse(BaseModel):
    session_id: str
    conversation_id: str
    message: str


class UpdateProfileRequest(BaseModel):
    traveler_type: str = None
    budget_level: str = None
    interests: list = []


# ── Endpoints ──

@router.post("/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest):
    """
    Starts a new user session.
    Creates user and conversation in MongoDB.
    Returns session_id and conversation_id.
    """
    # Generate unique session ID
    session_id = str(uuid.uuid4())

    # Create user in MongoDB
    mongodb.create_user(
        session_id=session_id,
        language=request.language
    )

    # Update traveler type if provided
    if request.traveler_type:
        mongodb.update_user_profile(
            session_id=session_id,
            profile={"traveler_type": request.traveler_type}
        )

    # Start conversation
    conversation_id = mongodb.start_conversation(session_id)

    return StartSessionResponse(
        session_id=session_id,
        conversation_id=conversation_id,
        message="Session started successfully"
    )


@router.get("/{session_id}")
async def get_session(session_id: str):
    """
    Gets user profile and conversation history.
    """
    user = mongodb.get_user(session_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    # Remove MongoDB internal ID
    user.pop("_id", None)

    return {
        "session_id": session_id,
        "profile": user
    }


@router.put("/{session_id}/profile")
async def update_profile(
    session_id: str,
    request: UpdateProfileRequest
):
    """
    Updates user travel preferences.
    """
    user = mongodb.get_user(session_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    profile = {}
    if request.traveler_type:
        profile["traveler_type"] = request.traveler_type
    if request.budget_level:
        profile["budget_level"] = request.budget_level
    if request.interests:
        profile["interests"] = request.interests

    mongodb.update_user_profile(session_id, profile)

    return {
        "message": "Profile updated successfully",
        "session_id": session_id,
        "updated": profile
    }


@router.delete("/{session_id}")
async def end_session(session_id: str):
    """
    Ends a session and clears memory.
    """
    from backend.core.smart_pipeline import smart_pipeline
    smart_pipeline.memories.pop(session_id, None)

    return {
        "message": "Session ended",
        "session_id": session_id
    }