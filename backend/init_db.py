import asyncio
from database import engine, Base
import models  # Imports models so metadata is registered

async def create_tables():
    async with engine.begin() as conn:
        # This creates all tables defined in models.py
        await conn.run_sync(Base.metadata.create_all)
    print("PostgreSQL Tables created successfully!")

if __name__ == "__main__":
    asyncio.run(create_tables())