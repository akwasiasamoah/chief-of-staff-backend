"""
Quick Migration Script - Run Locally to Update Render Database
"""

import sys

# Get database URL from command line
if len(sys.argv) < 2:
    print("Usage: python quick_migrate.py 'DATABASE_URL'")
    print("\nGet your DATABASE_URL from:")
    print("Render Dashboard → PostgreSQL → Connections → External Database URL")
    sys.exit(1)

DATABASE_URL = sys.argv[1]

# Use synchronous psycopg2 for simplicity
try:
    import psycopg2
except ImportError:
    print("❌ psycopg2 not installed")
    print("Install it with: pip install psycopg2-binary")
    sys.exit(1)

print(f"🔄 Connecting to database...")

try:
    # Connect to database
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Check if column exists
    cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name='briefs' AND column_name='meeting_id'
    """)
    
    exists = cursor.fetchone()
    
    if exists:
        print("✅ meeting_id column already exists - nothing to do!")
    else:
        print("➕ Adding meeting_id column...")
        cursor.execute("ALTER TABLE briefs ADD COLUMN meeting_id VARCHAR")
        
        print("📊 Creating index...")
        cursor.execute("CREATE INDEX idx_briefs_meeting_id ON briefs(meeting_id)")
        
        conn.commit()
        print("✅ Migration completed successfully!")
    
    # Verify
    print("\n📋 Current briefs table structure:")
    cursor.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name='briefs'
        ORDER BY ordinal_position
    """)
    
    for row in cursor.fetchall():
        print(f"  - {row[0]}: {row[1]}")
    
    cursor.close()
    conn.close()
    
    print("\n🎉 All done! You can now redeploy your app on Render.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
