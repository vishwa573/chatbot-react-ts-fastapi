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
from sqlalchemy import select

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

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
            title=f"Doc: {file.filename[:20]}"
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
            conv = Conversation(id=conv_id, title=payload.content[:30]) #for upload 
            db.add(conv)
            await db.commit()
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

    # Define the max number of historical messages to send (e.g., last 10 messages)
    MAX_HISTORY = 10
    # Base system prompt
    system_prompt = "You are a helpful, concise AI assistant. Format responses using Markdown."
    
    # --- RAG INJECTION START ---
    # 1. Check if this conversation has any uploaded documents
    count_result = qdrant.count(
        collection_name=COLLECTION_NAME,
        count_filter=Filter(
            must=[FieldCondition(key="conversation_id", match=MatchValue(value=str(conv_id)))]
        )
    )
    
    if count_result.count > 0:
        # 2. Embed the user's current question
        query_vector = embedding_model.encode(payload.content).tolist()
        
        # 3. Retrieve the top 3 most relevant chunks filtered by this exact chat
        search_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="conversation_id", match=MatchValue(value=str(conv_id)))]
            ),
            limit=3
        )
        
        # Safely extract the points (handles both list and QueryResponse object formats)
        search_results = getattr(search_response, 'points', search_response)
        
        if search_results:
            context_texts = [hit.payload["text"] for hit in search_results]
            
            # 4. Modify the system prompt to enforce RAG boundaries
            system_prompt += (
                "\n\nContext information is below.\n"
                "---------------------\n"
                f"{'\n'.join(context_texts) if context_texts else ''}\n"
                "---------------------\n"
                "Answer the user's question using the context provided above. "
            )
    messages_for_llm = [{"role": "system", "content": system_prompt}]
    
    # Slice the history to only include the most recent messages
    recent_history = history_messages[-MAX_HISTORY:] if len(history_messages) > MAX_HISTORY else history_messages
    
    # Append the recent history to the system prompt
    for msg in recent_history:
        messages_for_llm.append({"role": msg.role, "content": msg.content})

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