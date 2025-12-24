"""
Database Migration Script
Adds meeting_id column to briefs table
"""

import os
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# Database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql+asyncpg://', 1)

async def migrate():
    """Run database migrations"""
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set")
        return
    
    print(f"🔄 Connecting to database...")
    engine = create_async_engine(DATABASE_URL, echo=True)
    
    async with engine.begin() as conn:
        # Check if column exists
        result = await conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='briefs' AND column_name='meeting_id'
        """))
        
        column_exists = result.fetchone() is not None
        
        if column_exists:
            print("✅ meeting_id column already exists - skipping")
        else:
            print("➕ Adding meeting_id column...")
            await conn.execute(text("""
                ALTER TABLE briefs 
                ADD COLUMN meeting_id VARCHAR
            """))
            
            print("📊 Creating index on meeting_id...")
            await conn.execute(text("""
                CREATE INDEX idx_briefs_meeting_id 
                ON briefs(meeting_id)
            """))
            
            print("✅ Migration completed successfully!")
    
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(migrate())
