import os
import json
import uuid
import asyncio
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db, SessionLocal
from models import Conversation, Message as DBMessage

load_dotenv()

app = FastAPI(title="Chatbot API", version="1.0.0")

client = AsyncOpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hardcoded user ID for Phase 4 (prepping for Phase 5 Auth)
CURRENT_USER_ID = "user_123"

# ------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------
class ChatRequest(BaseModel):
    conversation_id: Optional[uuid.UUID] = None
    content: str  # The new user prompt

class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str

    class Config:
        from_attributes = True

class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str

    class Config:
        from_attributes = True

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "FastAPI backend is running smoothly!"}

@app.get("/conversations", response_model=List[ConversationOut])
async def list_conversations(db: AsyncSession = Depends(get_db)):
    """Fetch all conversations for the sidebar."""
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == CURRENT_USER_ID)
        .order_by(Conversation.created_at.desc())
    )
    result = await db.execute(stmt)
    return result.scalars().all()

@app.get("/conversations/{conversation_id}/messages", response_model=List[MessageOut])
async def get_conversation_messages(conversation_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Fetch all historical messages for a selected conversation."""
    stmt = (
        select(DBMessage)
        .where(DBMessage.conversation_id == conversation_id)
        .order_by(DBMessage.created_at.asc())
    )
    result = await db.execute(stmt)
    return result.scalars().all()

from fastapi.responses import StreamingResponse

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Phase 4: Database-backed Chat Endpoint
    Handles lazy creation, saves messages, streams tokens, and saves assistant reply.
    """
    conv_id = payload.conversation_id

    # 1. Lazy creation of conversation if none provided
    if not conv_id:
        title_snippet = payload.content[:30] + ("..." if len(payload.content) > 30 else "")
        new_conv = Conversation(user_id=CURRENT_USER_ID, title=title_snippet)
        db.add(new_conv)
        await db.commit()
        await db.refresh(new_conv)
        conv_id = new_conv.id
    else:
        # Verify conversation exists
        conv = await db.get(Conversation, conv_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

    # 2. Save User Message to Postgres
    user_msg = DBMessage(conversation_id=conv_id, role="user", content=payload.content)
    db.add(user_msg)
    await db.commit()

    # 3. Fetch full historical context for Groq
    stmt = (
        select(DBMessage)
        .where(DBMessage.conversation_id == conv_id)
        .order_by(DBMessage.created_at.asc())
    )
    history_result = await db.execute(stmt)
    history_messages = history_result.scalars().all()

    messages_for_llm = [{"role": msg.role, "content": msg.content} for msg in history_messages]

    # 4. Generator function for streaming + final DB save
    async def generate():
        full_assistant_response = ""

        stream = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages_for_llm,
            stream=True
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            if content:
                full_assistant_response += content
                yield f"data: {json.dumps({'conversation_id': str(conv_id), 'content': content})}\n\n"
                await asyncio.sleep(0.02)

        # 5. Once streaming ends, save the complete Assistant response to DB
        async with SessionLocal() as fresh_db:
            assistant_msg = DBMessage(
                conversation_id=conv_id,
                role="assistant",
                content=full_assistant_response
            )
            fresh_db.add(assistant_msg)
            await fresh_db.commit()

    return StreamingResponse(generate(), media_type="text/event-stream")