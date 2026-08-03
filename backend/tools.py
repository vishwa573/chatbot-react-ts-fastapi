from typing import Optional, List, Dict, Any
from strands import tool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from models import Conversation, Message as DBMessage


class WorkspaceTools:
    def __init__(self, qdrant: QdrantClient, embedding_model: SentenceTransformer, session_factory):
        self.qdrant = qdrant
        self.embedding_model = embedding_model
        self.session_factory = session_factory
        self.collection_name = "chat_documents"

    # ------------------------------------------------------------------
    # 1. List Uploaded Documents
    # ------------------------------------------------------------------
    @tool
    async def list_uploaded_documents(self, required_dummy: str, **kwargs) -> str:
        """
        Lists all PDF documents currently indexed in the workspace with their filenames and chat IDs.
        Use this when the user asks what documents they have uploaded or available.
        ALWAYS pass the word "ignored" to the required_dummy parameter.
        IMPORTANT: Never show the internal Chat IDs to the user in your final response. 
        Only refer to documents by their filename
        """
        print(f"\n>>> [TOOL START] list_uploaded_documents | Args: dummy='{required_dummy}', kwargs={kwargs}")
        
        scroll_result, _ = self.qdrant.scroll(
            collection_name=self.collection_name,
            limit=500,
            with_payload=["filename", "conversation_id", "document_id"],
            with_vectors=False
        )

        if not scroll_result:
            print(">>> [TOOL RETURNING] No documents found.")
            return "No documents found in the workspace."

        unique_docs: Dict[str, str] = {}
        for point in scroll_result:
            payload = point.payload or {}
            filename = payload.get("filename")
            conv_id = payload.get("conversation_id")
            if filename and filename not in unique_docs:
                unique_docs[filename] = conv_id

        formatted = [f"- {filename} (Chat ID: {conv_id})" for filename, conv_id in unique_docs.items()]
        final_result = "Uploaded Documents:\n" + "\n".join(formatted)
        
        print(f">>> [TOOL RETURNING] Success. Found {len(unique_docs)} unique docs.")
        return final_result

    # ------------------------------------------------------------------
    # 2. Search Documents
    # ------------------------------------------------------------------
    @tool
    async def search_documents(self, query: str, conversation_id: Optional[str] = None, **kwargs) -> str:
        """
        Searches the text content of uploaded PDF documents for answers to a specific query.
        If conversation_id is provided, it restricts the search to that document's chat.
        """
        print(f"\n>>> [TOOL START] search_documents | Args: query='{query}', conversation_id='{conversation_id}', kwargs={kwargs}")
        
        if not query:
            return "Error: No search query provided."
            
        query_vector = self.embedding_model.encode(query).tolist()

        query_filter = None
        if conversation_id:
            query_filter = Filter(
                must=[FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))]
            )

        search_response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=4
        )

        points = getattr(search_response, 'points', search_response)
        if not points:
            print(">>> [TOOL RETURNING] No relevant information found.")
            return "No relevant information found in the documents."

        snippets = [f"[Doc: {pt.payload.get('filename')}]\n{pt.payload.get('text')}" for pt in points if pt.score >= 0.15]
        
        if not snippets:
            print(">>> [TOOL RETURNING] Matches found, but relevance score too low.")
            return "Matches were found, but their relevance was too low."

        print(f">>> [TOOL RETURNING] Success. Returning {len(snippets)} snippets.")
        return "\n\n---\n\n".join(snippets)

    # ------------------------------------------------------------------
    # 3. Summarize Document
    # ------------------------------------------------------------------
    @tool
    async def summarize_document(self, filename: str, **kwargs) -> str:
        """
        Retrieves the main text content of a specific uploaded file by its filename to generate a summary.
        Use this when the user asks to summarize or give an overview of a specific PDF.
        """
        print(f"\n>>> [TOOL START] summarize_document | Args: filename='{filename}', kwargs={kwargs}")
        
        if not filename:
             return "Error: No filename provided for summarization."
             
        scroll_result, _ = self.qdrant.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
            ),
            limit=20, 
            with_payload=["text", "chunk_index"],
            with_vectors=False
        )

        if not scroll_result:
            print(f">>> [TOOL RETURNING] Could not find content for: {filename}")
            return f"Could not find any indexed content for file: {filename}"

        sorted_chunks = sorted(scroll_result, key=lambda x: x.payload.get("chunk_index", 0))
        full_text = "\n".join([pt.payload.get("text", "") for pt in sorted_chunks])

        print(f">>> [TOOL RETURNING] Success. Returning text snippet of length {len(full_text[:3000])}")
        return f"Document Content Snippet for '{filename}':\n\n{full_text[:3000]}"

    # ------------------------------------------------------------------
    # 4. Search Chat History
    # ------------------------------------------------------------------
    @tool
    async def search_chat_history(self, query: str, **kwargs) -> str:
        """
        Searches through past conversation messages for previous discussions, decisions, or user notes.
        Use this when the user asks what was discussed earlier or asks about previous chats.
        """
        print(f"\n>>> [TOOL START] search_chat_history | Args: query='{query}', kwargs={kwargs}")
        
        async with self.session_factory() as db:
            stmt = (
                select(DBMessage)
                .join(Conversation)
                .where(DBMessage.content.ilike(f"%{query}%"))
                .order_by(DBMessage.created_at.desc())
                .limit(5)
            )
            result = await db.execute(stmt)
            messages = result.scalars().all()

            if not messages:
                print(">>> [TOOL RETURNING] No chat history found.")
                return f"No chat history found matching query: '{query}'"

            history_str = []
            for msg in messages:
                history_str.append(f"[{msg.role.upper()}]: {msg.content[:200]}")

            print(f">>> [TOOL RETURNING] Success. Returning {len(messages)} historical messages.")
            return "Relevant Chat History:\n" + "\n---\n".join(history_str)

    # ------------------------------------------------------------------
    # 5. Rename Conversation
    # ------------------------------------------------------------------
    @tool
    async def rename_conversation(self, conversation_id: str, new_title: str, **kwargs) -> str:
        """
        Renames a conversation sidebar title based on conversation context or user request.
        """
        print(f"\n>>> [TOOL START] rename_conversation | Args: id='{conversation_id}', new_title='{new_title}', kwargs={kwargs}")
        import uuid as uuid_pkg
        async with self.session_factory() as db:
            try:
                conv_uuid = uuid_pkg.UUID(conversation_id)
                conv = await db.get(Conversation, conv_uuid)
                if not conv:
                    print(">>> [TOOL RETURNING] Conversation not found.")
                    return f"Conversation with ID {conversation_id} not found."

                conv.title = new_title
                await db.commit()
                print(f">>> [TOOL RETURNING] Success. Renamed to {new_title}")
                return f"Successfully renamed conversation {conversation_id} to '{new_title}'."
            except Exception as e:
                print(f">>> [TOOL RETURNING] Failed to rename: {e}")
                return f"Failed to rename conversation: {str(e)}"