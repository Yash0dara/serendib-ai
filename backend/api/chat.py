# Main chat endpoints

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.core.smart_pipeline import smart_pipeline
from backend.database.mongodb import mongodb
import uuid
from typing import Optional

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Request/Response Models ──

class ChatRequest(BaseModel):
    session_id: str
    conversation_id: str
    message: str




class ChatResponse(BaseModel):
    answer: str
    intent: str
    agent_used: str
    sentiment: str
    entities: dict
    sources: list
    user_state: Optional[str] = None
    validation_passed: bool = True


class QuickChatRequest(BaseModel):
    message: str
    language: str = "en"


# ── Endpoints ──

@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    Main chat endpoint.

    Requires:
    - session_id (from /session/start)
    - conversation_id (from /session/start)
    - message (user's message)

    Returns:
    - AI response with metadata
    """
    # Validate session exists
    user = mongodb.get_user(request.session_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call /session/start first."
        )

    # Process through smart pipeline
    result = await smart_pipeline.process(
        session_id=request.session_id,
        user_message=request.message,
        conversation_id=request.conversation_id
    )

    return ChatResponse(
        answer=result["answer"],
        intent=result["intent"],
        agent_used=result["agent_used"],
        sentiment=result["sentiment"],
        entities=result["entities"],
        sources=result["sources"],
        user_state=result.get("user_state"),
        validation_passed=result["validation"].get("passed", True)
    )


@router.post("/quick")
async def quick_chat(request: QuickChatRequest):
    """
    Quick chat endpoint.
    No session management needed.
    Creates temporary session automatically.
    Good for testing.
    """
    # Create temporary session
    session_id = f"temp_{str(uuid.uuid4())[:8]}"
    mongodb.create_user(session_id, language=request.language)
    conversation_id = mongodb.start_conversation(session_id)

    # Process message
    result = await smart_pipeline.process(
        session_id=session_id,
        user_message=request.message,
        conversation_id=conversation_id
    )

    return {
        "answer": result["answer"],
        "intent": result["intent"],
        "agent_used": result["agent_used"],
        "session_id": session_id
    }


@router.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 20):
    """
    Gets conversation history for a session.
    """
    user = mongodb.get_user(session_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    # Get memory from pipeline
    memory = smart_pipeline.get_memory(session_id)

    history = []
    for msg in memory:
        from langchain_core.messages import HumanMessage
        history.append({
            "role": "user" if isinstance(msg, HumanMessage) else "assistant",
            "content": msg.content
        })

    return {
        "session_id": session_id,
        "history": history[-limit:],
        "total_messages": len(history)
    }