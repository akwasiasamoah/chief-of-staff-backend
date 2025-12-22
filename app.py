"""
Chief of Staff - Backend with Proper OAuth Flow
This version implements the full OAuth flow to get Calendar/Gmail access
"""

import os
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import List, Optional
from dotenv import load_dotenv

# FastAPI
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

# Database
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, create_engine, select, desc
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# Google APIs
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# OpenAI
from openai import OpenAI

# Scheduler
import schedule

# Load environment
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

OPENAI_CLIENT = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
DATABASE_URL = "sqlite+aiosqlite:///./chief_of_staff.db"
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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
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
    attendee_email = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    read = Column(Boolean, default=False)
    
    user = relationship("User", back_populates="briefs")

# =============================================================================
# DATABASE CONNECTION
# =============================================================================

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# =============================================================================
# GOOGLE & OPENAI SERVICES
# =============================================================================

def get_google_credentials(user):
    """Create Google credentials from stored tokens"""
    if not user.access_token:
        return None
    
    return Credentials(
        token=user.access_token,
        refresh_token=user.refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv('GOOGLE_CLIENT_ID'),
        client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
        scopes=SCOPES,
        expiry=user.token_expiry
    )

async def get_daily_schedule(user):
    try:
        creds = get_google_credentials(user)
        if not creds:
            print(f"No credentials for user {user.email}")
            return []
        
        service = build('calendar', 'v3', credentials=creds)
        
        now = datetime.now()
        start_of_day = now.replace(hour=0, minute=0, second=0).isoformat() + 'Z'
        end_of_day = now.replace(hour=23, minute=59, second=59).isoformat() + 'Z'
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=start_of_day,
            timeMax=end_of_day,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        return events_result.get('items', [])
    except Exception as e:
        print(f"Error fetching calendar: {e}")
        return []

async def get_upcoming_meeting(user):
    try:
        creds = get_google_credentials(user)
        if not creds:
            return None
        
        service = build('calendar', 'v3', credentials=creds)
        
        now = datetime.utcnow()
        start_window = (now + timedelta(minutes=10)).isoformat() + 'Z'
        end_window = (now + timedelta(minutes=11)).isoformat() + 'Z'
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=start_window,
            timeMax=end_window,
            maxResults=1,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        return events[0] if events else None
    except Exception as e:
        print(f"Error fetching upcoming meeting: {e}")
        return None

async def get_recent_emails(user, email_address, max_results=3):
    if not email_address:
        return "No email address found."
    
    try:
        creds = get_google_credentials(user)
        if not creds:
            return "No credentials available."
        
        service = build('gmail', 'v1', credentials=creds)
        query = f"from:{email_address} OR to:{email_address}"
        
        results = service.users().messages().list(userId='me', q=query, maxResults=max_results).execute()
        messages = results.get('messages', [])
        
        if not messages:
            return "No recent emails found."
        
        email_data = []
        for msg in messages:
            full_msg = service.users().messages().get(userId='me', id=msg['id'], format='metadata').execute()
            payload = full_msg.get('payload', {})
            headers = payload.get('headers', [])
            
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
            snippet = full_msg.get('snippet', '')
            
            email_data.append(f"Subject: {subject}\nSnippet: {snippet}\n---")
        
        return "\n".join(email_data)
    except Exception as e:
        print(f"Error fetching emails: {e}")
        return f"Error fetching emails: {e}"

def generate_morning_brief(events):
    if not events:
        return "No meetings today. Clear skies! 🌞"
    
    try:
        schedule_text = "\n".join([
            f"- {e['start'].get('dateTime', 'All Day')}: {e.get('summary', 'No Title')}"
            for e in events
        ])
        
        prompt = f"""
You are an elite Chief of Staff. Analyze my schedule for today:
{schedule_text}

Provide a 'Morning Download':
1. One-sentence theme for the day.
2. List of meetings with a 'difficulty rating' (1-5).
3. Identify conflicts or tight spots.
4. Suggest a time block for deep work.
"""
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating brief: {e}")
        return f"Error generating brief: {e}"

def generate_meeting_brief(title, attendee, email_context):
    try:
        prompt = f"""
I have a meeting in 10 minutes.
**Meeting:** {title}
**Attendee:** {attendee}
**Recent Emails:**
{email_context}

Brief me concisely:
1. Who is this person (if you can infer from emails)?
2. What is the status of our last conversation?
3. Critical points to mention.
"""
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating meeting brief: {e}")
        return f"Error generating brief: {e}"

# =============================================================================
# BACKGROUND SCHEDULER
# =============================================================================

async def morning_routine():
    print(f"☀️ Running morning routine at {datetime.now()}")
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User))
        users = result.scalars().all()
        
        for user in users:
            try:
                if not user.notifications_enabled or not user.access_token:
                    continue
                
                print(f"Generating morning brief for {user.email}")
                events = await get_daily_schedule(user)
                content = generate_morning_brief(events)
                
                brief = Brief(
                    user_id=user.id,
                    brief_type="morning",
                    title=f"Morning Brief - {datetime.now().strftime('%B %d, %Y')}",
                    content=content
                )
                session.add(brief)
                await session.commit()
                print(f"✅ Morning brief saved for {user.email}")
            except Exception as e:
                print(f"❌ Error for {user.email}: {e}")

async def meeting_check_routine():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User))
        users = result.scalars().all()
        
        for user in users:
            try:
                if not user.notifications_enabled or not user.access_token:
                    continue
                
                event = await get_upcoming_meeting(user)
                if event:
                    title = event.get('summary', 'No Title')
                    attendees = event.get('attendees', [])
                    meeting_time = event['start'].get('dateTime')
                    
                    target_email = next((p['email'] for p in attendees if p['email'] != user.email), None)
                    
                    if target_email:
                        print(f"🚀 Meeting found for {user.email}: {title}")
                        
                        existing = await session.execute(
                            select(Brief).where(
                                Brief.user_id == user.id,
                                Brief.meeting_title == title,
                                Brief.attendee_email == target_email
                            )
                        )
                        if existing.scalar_one_or_none():
                            continue
                        
                        email_context = await get_recent_emails(user, target_email)
                        content = generate_meeting_brief(title, target_email, email_context)
                        
                        brief = Brief(
                            user_id=user.id,
                            brief_type="meeting",
                            title=f"Pre-Meeting: {title}",
                            content=content,
                            meeting_title=title,
                            meeting_time=datetime.fromisoformat(meeting_time.replace('Z', '+00:00')),
                            attendee_email=target_email
                        )
                        session.add(brief)
                        await session.commit()
                        print(f"✅ Meeting brief saved for {user.email}")
            except Exception as e:
                print(f"❌ Error for {user.email}: {e}")

def run_scheduler():
    schedule.every().day.at("08:00").do(lambda: asyncio.run(morning_routine()))
    schedule.every(1).minutes.do(lambda: asyncio.run(meeting_check_routine()))
    
    print("📅 Scheduler started")
    while True:
        schedule.run_pending()
        time.sleep(1)

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(title="Chief of Staff API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv('FRONTEND_URL', 'http://127.0.0.1:5500')],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    picture: Optional[str]
    morning_brief_time: str
    notifications_enabled: bool
    has_calendar_access: bool

class BriefResponse(BaseModel):
    id: int
    brief_type: str
    title: str
    content: str
    meeting_title: Optional[str]
    meeting_time: Optional[datetime]
    attendee_email: Optional[str]
    created_at: datetime
    read: bool

class UpdateSettingsRequest(BaseModel):
    morning_brief_time: Optional[str] = None
    notifications_enabled: Optional[bool] = None

@app.on_event("startup")
async def startup_event():
    await init_db()
    print("✅ Database initialized")
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    print("✅ Scheduler started")

@app.get("/")
async def root():
    return {"message": "Chief of Staff API", "version": "2.0.0"}

@app.get("/auth/google/url")
async def get_auth_url(user_email: str):
    """Generate Google OAuth URL for user to authorize"""
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
    
    auth_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent',
        login_hint=user_email
    )
    
    return {"auth_url": auth_url, "state": state}

@app.get("/auth/google/callback")
async def google_callback(code: str, state: str, db: AsyncSession = Depends(get_db)):
    """Handle OAuth callback and store tokens"""
    try:
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
            redirect_uri=REDIRECT_URI,
            state=state
        )
        
        flow.fetch_token(code=code)
        credentials = flow.credentials
        
        # Get user info
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests
        
        idinfo = id_token.verify_oauth2_token(
            credentials.id_token,
            google_requests.Request(),
            os.getenv('GOOGLE_CLIENT_ID')
        )
        
        email = idinfo['email']
        
        # Update or create user with tokens
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if user:
            user.access_token = credentials.token
            user.refresh_token = credentials.refresh_token
            user.token_expiry = credentials.expiry
            user.updated_at = datetime.utcnow()
        else:
            user = User(
                email=email,
                google_id=idinfo['sub'],
                name=idinfo.get('name', ''),
                picture=idinfo.get('picture', ''),
                access_token=credentials.token,
                refresh_token=credentials.refresh_token,
                token_expiry=credentials.expiry
            )
            db.add(user)
        
        await db.commit()
        
        # Redirect back to frontend
        return RedirectResponse(url=os.getenv('FRONTEND_URL', 'http://127.0.0.1:5500') + '/frontend/index.html?auth=success')
        
    except Exception as e:
        print(f"OAuth callback error: {e}")
        return RedirectResponse(url=os.getenv('FRONTEND_URL', 'http://127.0.0.1:5500') + '/frontend/index.html?auth=error')

@app.get("/user/me", response_model=UserResponse)
async def get_current_user(user_email: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == user_email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        **user.__dict__,
        "has_calendar_access": user.access_token is not None
    }

@app.patch("/user/settings")
async def update_settings(user_email: str, settings: UpdateSettingsRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == user_email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if settings.morning_brief_time is not None:
        user.morning_brief_time = settings.morning_brief_time
    if settings.notifications_enabled is not None:
        user.notifications_enabled = settings.notifications_enabled
    
    user.updated_at = datetime.utcnow()
    await db.commit()
    return {"message": "Settings updated"}

@app.get("/briefs", response_model=List[BriefResponse])
async def get_briefs(user_email: str, brief_type: Optional[str] = None, limit: int = 20, db: AsyncSession = Depends(get_db)):
    user_result = await db.execute(select(User).where(User.email == user_email))
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    query = select(Brief).where(Brief.user_id == user.id)
    if brief_type:
        query = query.where(Brief.brief_type == brief_type)
    query = query.order_by(desc(Brief.created_at)).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

@app.get("/briefs/{brief_id}", response_model=BriefResponse)
async def get_brief(brief_id: int, user_email: str, db: AsyncSession = Depends(get_db)):
    user_result = await db.execute(select(User).where(User.email == user_email))
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    result = await db.execute(select(Brief).where(Brief.id == brief_id, Brief.user_id == user.id))
    brief = result.scalar_one_or_none()
    if not brief:
        raise HTTPException(status_code=404, detail="Brief not found")
    return brief

@app.patch("/briefs/{brief_id}/read")
async def mark_brief_read(brief_id: int, user_email: str, db: AsyncSession = Depends(get_db)):
    user_result = await db.execute(select(User).where(User.email == user_email))
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    result = await db.execute(select(Brief).where(Brief.id == brief_id, Brief.user_id == user.id))
    brief = result.scalar_one_or_none()
    if not brief:
        raise HTTPException(status_code=404, detail="Brief not found")
    
    brief.read = True
    await db.commit()
    return {"message": "Brief marked as read"}

@app.get("/schedule/today")
async def get_today_schedule(user_email: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == user_email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    events = await get_daily_schedule(user)
    return {"events": events, "has_access": user.access_token is not None}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
