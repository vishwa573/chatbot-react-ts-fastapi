import litellm
# litellm._turn_on_debug()

import os
import json
import uuid
import asyncio
from typing import List, Optional
import io
import uuid
from pypdf import PdfReader
from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select , delete

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from pydantic import BaseModel

from strands import Agent
from strands.models.litellm import LiteLLMModel 

from tools import WorkspaceTools
from fastapi.responses import StreamingResponse
from database import get_db, SessionLocal
from models import Conversation, Message as DBMessage

load_dotenv()

app = FastAPI(title="Chatbot API", version="1.0.0")

class RenameRequest(BaseModel):
    title: str

# --- Initialize RAG Globals ---
print("Loading Embedding Model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

qdrant = QdrantClient(path="./qdrant_db")
COLLECTION_NAME = "chat_documents"

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )


# client = AsyncOpenAI(
#     api_key=os.environ.get("GROQ_API_KEY"),
#     base_url="https://api.groq.com/openai/v1"
# )

# --- Initialize Strands Provider for Groq ---
# We use LiteLLMModel to connect Strands to Groq's OpenAI-compatible API
model = LiteLLMModel(
    client_args={
        "api_key": os.environ["GROQ_API_KEY"],
    },
    # model_id="groq/llama-3.1-8b-instant", # Keep this as is
    model_id="groq/llama-3.3-70b-versatile", # Keep this as is
    params={
        "temperature": 0.2, #
        "custom_llm_provider": "groq" # Force the provider
    }
)
# Instantiate the tools using the secure dependency injection pattern
workspace_tools = WorkspaceTools(
    qdrant=qdrant,
    embedding_model=embedding_model,
    session_factory=SessionLocal
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
    message_id: Optional[uuid.UUID] = None        # ID for new messages
    edit_message_id: Optional[uuid.UUID] = None   # ID if we are editing an old message

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


@app.put("/conversations/{conversation_id}")
async def rename_conversation(conversation_id: uuid.UUID, req: RenameRequest, db: AsyncSession = Depends(get_db)):
    conv = await db.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conv.title = req.title
    await db.commit()
    return {"status": "success"}

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Deletes a conversation and cascades to delete all its messages."""
    conv = await db.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    await db.delete(conv)
    await db.commit()
    return {"status": "success"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    conversation_id: uuid.UUID = Form(...),
    db: AsyncSession = Depends(get_db)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    conv = await db.get(Conversation, conversation_id)
    if not conv:
        conv = Conversation(
            id=conversation_id, 
            title=f"Doc: {file.filename[:20]}",
            user_id=CURRENT_USER_ID
        )
        db.add(conv)
        await db.commit()

    # 1. Extract text from PDF
    pdf_bytes = await file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    raw_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            raw_text += extracted + "\n"
            
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    # 2. Chunking (Simple word-count window for the prototype)
    words = raw_text.split()
    chunk_size = 200 # Approx paragraphs
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    
    # 3. Generate Embeddings
    embeddings = embedding_model.encode(chunks)
    
    # 4. Store in Qdrant with specific metadata
    document_id = str(uuid.uuid4())
    points = []
    
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "conversation_id": str(conversation_id),
                    "document_id": document_id,
                    "filename": file.filename,
                    "chunk_index": i,
                    "text": chunk
                }
            )
        )
    
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    
    return {
        "status": "success", 
        "filename": file.filename,
        "chunks_processed": len(chunks), 
        "document_id": document_id
    }

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Strands Agent Chat Endpoint
    """
    conv_id = payload.conversation_id

    # 1. Lazy creation of conversation
    if not conv_id:
        title_snippet = payload.content[:30] + ("..." if len(payload.content) > 30 else "")
        new_conv = Conversation(user_id=CURRENT_USER_ID, title=title_snippet)
        db.add(new_conv)
        await db.commit()
        await db.refresh(new_conv)
        conv_id = new_conv.id
    else:
        conv = await db.get(Conversation, conv_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

    # 2. Handle Message Updates (Edit vs New)
    if payload.edit_message_id:
        user_msg = await db.get(DBMessage, payload.edit_message_id)
        if not user_msg:
            raise HTTPException(status_code=404, detail="Message not found")
        user_msg.content = payload.content
        await db.execute(
            delete(DBMessage).where(
                DBMessage.conversation_id == conv_id,
                DBMessage.created_at > user_msg.created_at
            )
        )
        await db.commit()
    else:
        user_msg = DBMessage(
            id=payload.message_id or uuid.uuid4(),
            conversation_id=conv_id,
            role="user",
            content=payload.content
        )
        db.add(user_msg)
        await db.commit()

    stmt = (
        select(DBMessage)
        .where(DBMessage.conversation_id == conv_id)
        .order_by(DBMessage.created_at.asc())
    )
    history_result = await db.execute(stmt)
    history_messages = history_result.scalars().all()

    # Format the last 10 messages for the Agent
    messages_for_agent = [{"role": msg.role, "content": msg.content} for msg in history_messages[-10:]]

    # 4. Initialize a FRESH stateless Agent for this specific request
    # This prevents users from colliding with each other's state
    request_agent = Agent(
        model=model,
        tools=[
            workspace_tools.list_uploaded_documents, #[cite: 1]
            workspace_tools.search_documents, #[cite: 1]
            workspace_tools.summarize_document, #[cite: 1]
            workspace_tools.search_chat_history, #[cite: 1]
            workspace_tools.rename_conversation #[cite: 1]
        ],
        system_prompt = f"""You are a versatile AI Assistant and Workspace Companion. 

        CAPABILITIES:
        1. General Knowledge & Chat: You can answer general knowledge questions, discuss industry trends (like AI, tech, news), help with coding, and engage in standard conversational Q&A.
        2. Workspace Management: You have tools to search, list, and summarize uploaded PDF documents, search past chat history, and rename conversations.

        CRITICAL CONTEXT FOR THIS REQUEST:
        - The current active conversation ID is: `{conv_id}`
        - Whenever you need to rename this conversation using the `rename_conversation` tool, you MUST use this exact UUID string. Never use placeholder words like "current".

        RULES:
        1. General Queries: For general knowledge, casual conversation, or external topics (e.g., recent AI trends, coding help, general facts), respond directly using your knowledge WITHOUT using any tools.
        2. Workspace Queries: ONLY use workspace tools when the user explicitly asks to search, summarize, list, or manage uploaded documents or past chat history.
        3. Tool Output Rules: When calling a tool, strictly output valid JSON and ensure all Tool IDs are integers.
        4. Privacy: Never expose internal metadata (like raw UUIDs or database keys) to the user in your final response text.""")
    # 5. The Streaming Generator
    async def generate():
        full_assistant_response = ""
        # Get the last 5 messages, excluding the current one we just added
        past_messages = history_messages[-6:-1] 
        history_text = "\n".join([f"{m.role.upper()}: {m.content}" for m in past_messages])
        
        if history_text.strip():
            contextual_prompt = f"--- Previous Chat Context ---\n{history_text}\n\n--- Current User Request ---\nUSER: {payload.content}"
        else:
            contextual_prompt = payload.content
        try:
            # Bug 1 & 3 Fixed: Use stream_async and pass the list of messages directly to prompt
            async for event in request_agent.stream_async(prompt=contextual_prompt):
                
                if "current_tool_use" in event:
                    print(f"🔧 AGENT USING TOOL: {event['current_tool_use']['name']}")
                
                if "data" in event and event["data"]:
                    chunk = event["data"]
                    full_assistant_response += chunk
                    yield f"data: {json.dumps({'conversation_id': str(conv_id), 'content': chunk})}\n\n"
                    await asyncio.sleep(0.01)
                    
        except Exception as e:
            error_str = str(e)
            print(f"Agent Error: {error_str}")

            # --- NEW: Graceful Rate Limit Handling ---
            if "RateLimitError" in error_str or "rate_limit_exceeded" in error_str:
                safe_error_msg = "\n\n⚠️ **Rate limit exceeded.** We hit the 6,000 tokens-per-minute ceiling. Please wait a few seconds and try again."
            else:
                safe_error_msg = "\n\n⚠️ **An internal error occurred.** Please try again."

            yield f"data: {json.dumps({'conversation_id': str(conv_id), 'content': safe_error_msg})}\n\n"

        # 6. Save Final Response to DB
        if full_assistant_response:
            async with SessionLocal() as fresh_db:
                assistant_msg = DBMessage(
                    conversation_id=conv_id,
                    role="assistant",
                    content=full_assistant_response
                )
                fresh_db.add(assistant_msg)
                await fresh_db.commit()

    return StreamingResponse(generate(), media_type="text/event-stream")