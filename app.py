"""
Chief of Staff - Backend with PostgreSQL Support
Supports both SQLite (local) and PostgreSQL (production)
"""

import os
import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from dotenv import load_dotenv

# FastAPI
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Database
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, create_engine, select, desc
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# Google APIs
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Google Gemini AI (New SDK)
from google import genai

# SendGrid for email notifications
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Load environment
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Configure Gemini AI (New SDK)
GEMINI_CLIENT = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
GEMINI_MODEL = 'gemini-2.5-flash'  # Latest fast model

# Database URL - Auto-detect PostgreSQL or SQLite
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL:
    # Render provides DATABASE_URL for PostgreSQL
    # Fix for SQLAlchemy (needs postgresql:// not postgres://)
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql+asyncpg://', 1)
    elif not DATABASE_URL.startswith('postgresql+asyncpg://'):
        DATABASE_URL = DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://', 1)
    print(f"✅ Using PostgreSQL database")
else:
    # Local development uses SQLite
    DATABASE_URL = "sqlite+aiosqlite:///./chief_of_staff.db"
    print(f"✅ Using SQLite database (local)")

SCOPES = [
    'https://www.googleapis.com/auth/calendar.readonly',
    'https://www.googleapis.com/auth/gmail.readonly',
    'openid',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile'
]

# OAuth Flow Configuration
REDIRECT_URI = os.getenv('REDIRECT_URI', 'http://localhost:8000/auth/google/callback')

# =============================================================================
# DATABASE MODELS
# =============================================================================

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    google_id = Column(String, unique=True)
    name = Column(String)
    picture = Column(String, nullable=True)
    access_token = Column(Text, nullable=True)
    refresh_token = Column(Text, nullable=True)
    token_expiry = Column(DateTime, nullable=True)
    morning_brief_time = Column(String, default="08:00")
    notifications_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    briefs = relationship("Brief", back_populates="user", cascade="all, delete-orphan")

class Brief(Base):
    __tablename__ = "briefs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    brief_type = Column(String)
    title = Column(String)
    content = Column(Text)
    meeting_title = Column(String, nullable=True)
    meeting_time = Column(DateTime, nullable=True)
    meeting_id = Column(String, nullable=True, index=True)  # Google Calendar event ID
    attendee_email = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    read = Column(Boolean, default=False)
    
    user = relationship("User", back_populates="briefs")

# =============================================================================
# DATABASE CONNECTION
# =============================================================================

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database initialized")

async def get_db():
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        yield session

# =============================================================================
# FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    await init_db()
    asyncio.create_task(run_scheduler_loop())
    print("✅ Scheduler started")
    yield
    # Shutdown (cleanup if needed)
    print("👋 Shutting down...")

app = FastAPI(title="Chief of Staff Backend", lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000", 
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        os.getenv('FRONTEND_URL', 'http://localhost:3000'),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_credentials_from_user(user: User) -> Optional[Credentials]:
    """Convert user tokens to Google Credentials object"""
    if not user.access_token:
        return None
    
    creds = Credentials(
        token=user.access_token,
        refresh_token=user.refresh_token,
        token_uri='https://oauth2.googleapis.com/token',
        client_id=os.getenv('GOOGLE_CLIENT_ID'),
        client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
        scopes=SCOPES
    )
    
    return creds

async def refresh_user_token(user: User, db: AsyncSession):
    """Refresh expired access token"""
    creds = get_credentials_from_user(user)
    if creds and creds.expired and creds.refresh_token:
        try:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
            
            user.access_token = creds.token
            user.token_expiry = creds.expiry
            await db.commit()
            print(f"✅ Refreshed token for {user.email}")
            return creds
        except Exception as e:
            print(f"❌ Error refreshing token: {e}")
            return None
    return creds

# =============================================================================
# OAUTH ENDPOINTS
# =============================================================================

@app.get("/auth/google/url")
async def get_google_oauth_url(user_email: str = ""):
    """Generate Google OAuth authorization URL"""
    try:
        # Create OAuth flow
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": os.getenv('GOOGLE_CLIENT_ID'),
                    "client_secret": os.getenv('GOOGLE_CLIENT_SECRET'),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [REDIRECT_URI]
                }
            },
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        
        # Generate authorization URL
        auth_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent',  # Force consent screen to get refresh token
            login_hint=user_email if user_email else None
        )
        
        return {
            "auth_url": auth_url,
            "state": state
        }
        
    except Exception as e:
        print(f"Error generating OAuth URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auth/google/callback")
async def google_oauth_callback(code: str, state: str, db: AsyncSession = Depends(get_db)):
    """Handle OAuth callback from Google"""
    try:
        # Exchange authorization code for tokens
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": os.getenv('GOOGLE_CLIENT_ID'),
                    "client_secret": os.getenv('GOOGLE_CLIENT_SECRET'),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [REDIRECT_URI]
                }
            },
            scopes=SCOPES,
            state=state,
            redirect_uri=REDIRECT_URI
        )
        
        flow.fetch_token(code=code)
        credentials = flow.credentials
        
        # Get user info from Google
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests
        
        idinfo = id_token.verify_oauth2_token(
            credentials.id_token,
            google_requests.Request(),
            os.getenv('GOOGLE_CLIENT_ID')
        )
        
        email = idinfo['email']
        google_id = idinfo['sub']
        name = idinfo.get('name', '')
        picture = idinfo.get('picture', '')
        
        # Find or create user
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if not user:
            user = User(
                email=email,
                google_id=google_id,
                name=name,
                picture=picture,
                access_token=credentials.token,
                refresh_token=credentials.refresh_token,
                token_expiry=credentials.expiry
            )
            db.add(user)
            print(f"✅ New user created: {email}")
        else:
            user.access_token = credentials.token
            user.refresh_token = credentials.refresh_token
            user.token_expiry = credentials.expiry
            user.name = name
            user.picture = picture
            user.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            print(f"✅ User updated: {email}")
        
        await db.commit()
        
        # Redirect back to frontend with email parameter
        frontend_url = os.getenv('FRONTEND_URL', 'http://127.0.0.1:5500')
        redirect_url = f"{frontend_url}?auth=success&email={email}"
        return RedirectResponse(url=redirect_url)
        
    except Exception as e:
        print(f"❌ OAuth callback error: {e}")
        frontend_url = os.getenv('FRONTEND_URL', 'http://127.0.0.1:5500')
        return RedirectResponse(url=f"{frontend_url}?auth=error")

# =============================================================================
# USER ENDPOINTS
# =============================================================================

@app.get("/user/me")
async def get_user(user_email: str, db: AsyncSession = Depends(get_db)):
    """Get user information"""
    result = await db.execute(select(User).where(User.email == user_email))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "picture": user.picture,
        "morning_brief_time": user.morning_brief_time,
        "notifications_enabled": user.notifications_enabled,
        "has_calendar_access": user.access_token is not None
    }

@app.patch("/user/settings")
async def update_user_settings(
    user_email: str,
    settings: dict,
    db: AsyncSession = Depends(get_db)
):
    """Update user settings"""
    result = await db.execute(select(User).where(User.email == user_email))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if 'morning_brief_time' in settings:
        user.morning_brief_time = settings['morning_brief_time']
    if 'notifications_enabled' in settings:
        user.notifications_enabled = settings['notifications_enabled']
    
    user.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    await db.commit()
    
    return {"status": "success"}

# =============================================================================
# CALENDAR & GMAIL FUNCTIONS
# =============================================================================

async def get_calendar_events(user: User, db: AsyncSession, time_min=None, time_max=None):
    """Fetch calendar events for user"""
    creds = await refresh_user_token(user, db)
    if not creds:
        return []
    
    try:
        service = build('calendar', 'v3', credentials=creds)
        
        if not time_min:
            # Get current UTC time and format with Z suffix
            time_min = datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + 'Z'
        if not time_max:
            # Get tomorrow UTC time and format with Z suffix
            time_max = (datetime.now(timezone.utc) + timedelta(days=1)).replace(tzinfo=None).isoformat() + 'Z'
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        return events_result.get('items', [])
        
    except HttpError as e:
        print(f"Error fetching calendar: {e}")
        return []

async def get_email_history(user: User, db: AsyncSession, query: str = "", max_results: int = 10):
    """Fetch email history for user"""
    creds = await refresh_user_token(user, db)
    if not creds:
        return []
    
    try:
        service = build('gmail', 'v1', credentials=creds)
        
        results = service.users().messages().list(
            userId='me',
            q=query,
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])
        
        email_details = []
        for msg in messages[:5]:  # Limit to 5 for performance
            message = service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='metadata',
                metadataHeaders=['From', 'Subject', 'Date']
            ).execute()
            
            headers = {h['name']: h['value'] for h in message['payload']['headers']}
            email_details.append({
                'from': headers.get('From', ''),
                'subject': headers.get('Subject', ''),
                'date': headers.get('Date', '')
            })
        
        return email_details
        
    except HttpError as e:
        print(f"Error fetching emails: {e}")
        return []

# =============================================================================
# EMAIL NOTIFICATIONS
# =============================================================================

async def send_brief_email(user: User, brief):
    """Send email notification for new brief"""
    if not user.notifications_enabled:
        print(f"⏭️  Notifications disabled for {user.email}")
        return
    
    sendgrid_api_key = os.getenv('SENDGRID_API_KEY')
    sender_email = os.getenv('SENDGRID_FROM_EMAIL')
    
    if not sendgrid_api_key or not sender_email:
        print("⚠️  SendGrid not configured (SENDGRID_API_KEY or SENDGRID_FROM_EMAIL missing)")
        return
    
    try:
        # Disable SSL verification for SendGrid (workaround for corporate/proxy SSL issues)
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Format brief type
        brief_type_emoji = "☀️" if brief.brief_type == "morning" else "📅"
        
        # Create email content
        html_content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px 20px;
                    border-radius: 10px 10px 0 0;
                    text-align: center;
                }}
                .content {{
                    background: #f8fafc;
                    padding: 30px;
                    border-radius: 0 0 10px 10px;
                }}
                .brief-title {{
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 20px;
                    color: #1e293b;
                }}
                .brief-content {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                    white-space: pre-wrap;
                }}
                .button {{
                    display: inline-block;
                    background: #667eea;
                    color: white;
                    padding: 12px 30px;
                    text-decoration: none;
                    border-radius: 6px;
                    margin-top: 20px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    color: #64748b;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>👨‍💼 Chief of Staff</h1>
                <p>Your AI Executive Assistant</p>
            </div>
            <div class="content">
                <div class="brief-title">{brief_type_emoji} {brief.title}</div>
                <div class="brief-content">{brief.content}</div>
                <center>
                    <a href="{os.getenv('FRONTEND_URL', 'http://localhost:5500')}" class="button">
                        View in Dashboard
                    </a>
                </center>
            </div>
            <div class="footer">
                <p>You're receiving this because you have notifications enabled in Chief of Staff.</p>
                <p><a href="{os.getenv('FRONTEND_URL', 'http://localhost:5500')}">Manage Settings</a></p>
            </div>
        </body>
        </html>
        '''
        
        # Create message
        message = Mail(
            from_email=sender_email,
            to_emails=user.email,
            subject=f"{brief_type_emoji} {brief.title}",
            html_content=html_content
        )
        
        # Send email
        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)
        
        print(f"✅ Email sent to {user.email} (Status: {response.status_code})")
        
    except Exception as e:
        print(f"❌ Email error for {user.email}: {e}")

# =============================================================================
# BRIEF GENERATION
# =============================================================================

async def generate_morning_brief(user: User, db: AsyncSession):
    """Generate morning brief for user"""
    try:
        # Get today's events
        events = await get_calendar_events(user, db)
        
        if not events:
            content = "No meetings scheduled for today. You have a clear calendar!"
        else:
            # Format events
            events_text = "\n".join([
                f"- {event['summary']} at {event['start'].get('dateTime', event['start'].get('date'))}"
                for event in events
            ])
            
            # Generate brief with Gemini (New SDK)
            prompt = f"""Create a concise morning brief for the day. Here are today's meetings:

{events_text}

Provide:
1. A summary of the day ahead
2. Key meetings to prepare for
3. Any time gaps for focused work

Keep it brief and actionable."""

            response = GEMINI_CLIENT.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            content = response.text
        
        # Save brief
        brief = Brief(
            user_id=user.id,
            brief_type="morning",
            title=f"Morning Brief - {datetime.now().strftime('%B %d, %Y')}",
            content=content,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None)
        )
        db.add(brief)
        await db.commit()
        await db.refresh(brief)
        print(f"✅ Morning brief generated for {user.email}")
        
        # Send email notification
        await send_brief_email(user, brief)
        
    except Exception as e:
        print(f"❌ Error generating morning brief: {e}")

async def generate_meeting_brief(user: User, meeting: dict, db: AsyncSession):
    """Generate pre-meeting brief"""
    try:
        meeting_title = meeting.get('summary', 'Meeting')
        meeting_id = meeting.get('id')  # Google Calendar event ID
        attendees = meeting.get('attendees', [])
        
        if not attendees:
            return
        
        # Check if brief already exists for this meeting
        if meeting_id:
            result = await db.execute(
                select(Brief).where(
                    Brief.user_id == user.id,
                    Brief.meeting_id == meeting_id
                )
            )
            existing_brief = result.scalar_one_or_none()
            
            if existing_brief:
                print(f"⏭️  Brief already exists for meeting: {meeting_title} (skipping)")
                return
        
        # Get email history with attendees
        attendee_emails = [a['email'] for a in attendees if 'email' in a]
        email_query = ' OR '.join([f'from:{email}' for email in attendee_emails[:3]])
        
        emails = await get_email_history(user, db, query=email_query, max_results=5)
        
        # Format email context
        if emails:
            email_context = "\n".join([
                f"- From {e['from']}: {e['subject']}"
                for e in emails
            ])
        else:
            email_context = "No recent email history found."
        
        # Generate brief with Gemini (New SDK)
        prompt = f"""Create a pre-meeting brief for: {meeting_title}

Attendees: {', '.join(attendee_emails)}

Recent email context:
{email_context}

Provide:
1. Meeting context based on recent communications
2. Key topics to discuss
3. Suggested talking points

Keep it concise and actionable."""

        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        content = response.text
        
        # Save brief
        brief = Brief(
            user_id=user.id,
            brief_type="meeting",
            title=f"Pre-Meeting Brief: {meeting_title}",
            content=content,
            meeting_title=meeting_title,
            meeting_time=datetime.fromisoformat(meeting['start']['dateTime'].replace('Z', '+00:00')).replace(tzinfo=None),
            meeting_id=meeting_id,  # Store meeting ID to prevent duplicates
            attendee_email=attendee_emails[0] if attendee_emails else None,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None)
        )
        db.add(brief)
        await db.commit()
        await db.refresh(brief)
        print(f"✅ Meeting brief generated for {user.email}: {meeting_title}")
        
        # Send email notification
        await send_brief_email(user, brief)
        
    except Exception as e:
        print(f"❌ Error generating meeting brief: {e}")

# =============================================================================
# SCHEDULER
# =============================================================================

async def check_and_generate_briefs():
    """Check and generate briefs for all users"""
    async with AsyncSessionLocal() as db:
        try:
            # Get all users
            result = await db.execute(select(User))
            users = result.scalars().all()
            
            for user in users:
                if not user.access_token:
                    continue
                
                try:
                    # Check for morning brief
                    current_time = datetime.now().strftime('%H:%M')
                    if current_time == user.morning_brief_time:
                        await generate_morning_brief(user, db)
                    
                    # Check for upcoming meetings (10 minutes before)
                    upcoming_start = datetime.now(timezone.utc) + timedelta(minutes=10)
                    upcoming_end = upcoming_start + timedelta(minutes=1)
                    
                    events = await get_calendar_events(
                        user, db,
                        time_min=upcoming_start.replace(tzinfo=None).isoformat() + 'Z',
                        time_max=upcoming_end.replace(tzinfo=None).isoformat() + 'Z'
                    )
                    
                    for event in events:
                        await generate_meeting_brief(user, event, db)
                        
                except Exception as e:
                    print(f"Error processing user {user.email}: {e}")
        except Exception as e:
            print(f"Error in scheduler: {e}")

async def run_scheduler_loop():
    """Run scheduler in async loop"""
    print("📅 Scheduler loop started")
    while True:
        try:
            await check_and_generate_briefs()
        except Exception as e:
            print(f"❌ Scheduler error: {e}")
        
        # Wait 1 minute before next check
        await asyncio.sleep(60)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "app": "Chief of Staff Backend",
        "database": "PostgreSQL" if "postgresql" in DATABASE_URL else "SQLite"
    }

@app.get("/schedule/today")
async def get_today_schedule(user_email: str, db: AsyncSession = Depends(get_db)):
    """Get today's calendar events"""
    result = await db.execute(select(User).where(User.email == user_email))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    events = await get_calendar_events(user, db)
    
    return {
        "events": events,
        "has_access": user.access_token is not None
    }

@app.get("/schedule/debug")
async def debug_schedule(user_email: str, db: AsyncSession = Depends(get_db)):
    """Debug calendar events - see what Google returns"""
    result = await db.execute(select(User).where(User.email == user_email))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get current time info
    now = datetime.now(timezone.utc)
    tomorrow = now + timedelta(days=1)
    time_min = now.replace(tzinfo=None).isoformat() + 'Z'
    time_max = tomorrow.replace(tzinfo=None).isoformat() + 'Z'
    
    # Get events
    events = await get_calendar_events(user, db)
    
    return {
        "current_time_utc": now.isoformat(),
        "query_time_min": time_min,
        "query_time_max": time_max,
        "events_count": len(events),
        "events": events,
        "has_access": user.access_token is not None,
        "user_has_token": user.access_token is not None,
        "user_has_refresh_token": user.refresh_token is not None
    }

@app.get("/briefs")
async def get_briefs(
    user_email: str,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Get user's briefs"""
    result = await db.execute(select(User).where(User.email == user_email))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    result = await db.execute(
        select(Brief)
        .where(Brief.user_id == user.id)
        .order_by(desc(Brief.created_at))
        .limit(limit)
    )
    briefs = result.scalars().all()
    
    return [
        {
            "id": b.id,
            "brief_type": b.brief_type,
            "title": b.title,
            "content": b.content,
            "created_at": b.created_at.isoformat(),
            "read": b.read
        }
        for b in briefs
    ]

@app.get("/briefs/{brief_id}")
async def get_brief(brief_id: int, user_email: str, db: AsyncSession = Depends(get_db)):
    """Get specific brief"""
    result = await db.execute(select(User).where(User.email == user_email))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    result = await db.execute(
        select(Brief).where(Brief.id == brief_id, Brief.user_id == user.id)
    )
    brief = result.scalar_one_or_none()
    
    if not brief:
        raise HTTPException(status_code=404, detail="Brief not found")
    
    return {
        "id": brief.id,
        "brief_type": brief.brief_type,
        "title": brief.title,
        "content": brief.content,
        "created_at": brief.created_at.isoformat(),
        "read": brief.read
    }

@app.patch("/briefs/{brief_id}/read")
async def mark_brief_read(brief_id: int, user_email: str, db: AsyncSession = Depends(get_db)):
    """Mark brief as read"""
    result = await db.execute(select(User).where(User.email == user_email))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    result = await db.execute(
        select(Brief).where(Brief.id == brief_id, Brief.user_id == user.id)
    )
    brief = result.scalar_one_or_none()
    
    if not brief:
        raise HTTPException(status_code=404, detail="Brief not found")
    
    brief.read = True
    await db.commit()
    
    return {"status": "success"}

# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
