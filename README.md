# Chatbot (React + TypeScript frontend, FastAPI backend)

A lightweight multi-turn chatbot project with a Vite + React TypeScript frontend and an async FastAPI backend that stores conversations in PostgreSQL and streams assistant replies using Server-Sent Events (SSE).

This README summarizes the implemented features, development setup, running the project locally, and troubleshooting tips.

---

## Key features

- Frontend
  - Vite + React (TypeScript)
  - Minimal, responsive chat UI (uses react-markdown for rendering assistant messages)
  - Development and build scripts (dev, build, preview)

- Backend
  - FastAPI async backend with these endpoints:
    - `GET /health` - simple health check
    - `GET /conversations` - list saved conversations
    - `GET /conversations/{id}/messages` - get messages for a conversation
    - `DELETE /conversations/{id}` - delete a conversation and its messages
    - `POST /chat` - main chat endpoint; accepts a prompt + optional conversation_id
  - Conversation and message persistence using SQLAlchemy (async) and PostgreSQL
  - Lazy conversation creation (new conversation created automatically on first user message when none provided)
  - Streaming assistant replies via Server-Sent Events (SSE) from the `/chat` endpoint
  - System prompt + limited message history (default: last 10 messages) sent to the LLM for context
  - Configured CORS to allow Vite dev server at `http://localhost:5173`
  - Uses an async OpenAI-compatible client (configured to use GROQ API base URL in code)

- Database utilities
  - `backend/init_db.py` to create database tables from SQLAlchemy models

---

## Prerequisites

- Git
- Node.js (recommended v16+ or latest LTS)
- npm or yarn (or pnpm) for frontend
- Python 3.10+ (3.11 recommended) for backend
- PostgreSQL accessible for the backend (or a hosted Postgres)

---

## Environment variables

Create a `.env` file in the `backend/` directory (the backend uses python-dotenv). Minimum variables required:

```
# backend/.env (example)
DATABASE_URL=postgresql+asyncpg://DB_USER:DB_PASS@DB_HOST:DB_PORT/DB_NAME
GROQ_API_KEY=your_groq_api_key_here
# (Optional) any other env used by your deployment or secrets manager
```

Notes:
- DATABASE_URL must be an async SQLAlchemy-compatible URL using asyncpg driver (example above).
- The backend code sets the OpenAI-compatible client with a base_url that points to Groq by default. Set GROQ_API_KEY accordingly.

On the frontend side (if you need to point the dev server at the backend), create a `.env` in `frontend/`:

```
# frontend/.env
VITE_API_URL=http://localhost:8000
```

(Adjust the name `VITE_API_URL` to match any environment variable your frontend code expects.)

---

## Setup and run (development)

1. Clone the repository

```bash
git clone <repo-url>
cd <repo-folder>
```

2. Backend (FastAPI)

- Create and activate a virtual environment, then install the dependencies. The project does not include a checked-in `requirements.txt`; install the common packages used by the code:

```bash
cd backend
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# or on cmd.exe
.\.venv\Scripts\activate

pip install --upgrade pip
pip install fastapi uvicorn[standard] python-dotenv openai sqlalchemy asyncpg
```

- Configure environment variables in `backend/.env` (see above).

- Create database tables:

```bash
python init_db.py
```

- Start the backend server (default port 8000):

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You can now view automatic API docs at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.

3. Frontend (Vite + React)

```bash
cd ../frontend
npm install
# or `yarn` / `pnpm install`

# Start dev server (defaults to http://localhost:5173)
npm run dev
```

If the frontend needs to call the backend, ensure `VITE_API_URL` (or the appropriate env variable used by the frontend) points to `http://localhost:8000`.

---

## Usage notes / behavior

- Multi-turn chat: The backend keeps conversation history in Postgres and sends a limited number of recent messages (default: last 10) to the LLM for context to avoid extremely long prompts.
- Streaming: `/chat` streams partial assistant output using SSE. The frontend should handle SSE events to render chunks progressively and then finalize the stored assistant message after streaming ends.
- Hardcoded user: The current implementation uses a hardcoded `CURRENT_USER_ID` in `backend/main.py` and `models.py` as `user_123`. This is intentional for the current development phase and prepares the codebase for a future auth phase.
- System prompt: A system message is prepended to every LLM call to control assistant behavior. It can be edited in `backend/main.py`.

---

## Configuration you may want to change

- `MAX_HISTORY` in `backend/main.py` (default 10) — number of recent messages sent to the LLM
- CORS origins in `backend/main.py` — add production frontend origins as needed
- Replace the hardcoded `CURRENT_USER_ID` with proper authentication (Phase 5 planned)
- OpenAI/Groq client configuration — set keys and base URLs as appropriate for your LLM provider

---

## Troubleshooting

- "Could not connect to the database": verify `DATABASE_URL` is correct and your Postgres server accepts connections. Use the asyncpg-style URL shown above.
- Missing env key errors (e.g., GROQ_API_KEY): ensure `backend/.env` exists and is loaded or set env vars in your environment.
- SSE streaming not working in browser: ensure the frontend connects to the backend `http://localhost:8000/chat` using `EventSource` or equivalent and that CORS allows the origin.

---

## Tests & Linting

- Frontend lint script is available in `frontend/package.json`:

```bash
cd frontend
npm run lint
```

- No centralized test suite is included by default. Add unit or integration tests as needed for your workflow.

---

## Contributing

- Open issues or pull requests for bugs or feature requests.
- Follow the project's coding conventions: TypeScript for frontend, idiomatic async FastAPI + SQLAlchemy for backend.

---

