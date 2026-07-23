import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")

# Create the async engine
engine = create_async_engine(DATABASE_URL, echo=False)

# Create the session factory
SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

# Dependency injection to get the DB session in FastAPI routes
async def get_db():
    async with SessionLocal() as session:
        yield session