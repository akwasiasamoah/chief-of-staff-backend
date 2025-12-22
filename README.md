# Chief of Staff - Backend API

FastAPI backend for Chief of Staff AI assistant.

## Features

- Google OAuth authentication
- Calendar & Gmail integration
- OpenAI GPT-4 brief generation
- Background scheduler for automated briefs
- SQLite database
- RESTful API

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-your-key
SECRET_KEY=random-secret-key
GOOGLE_CLIENT_ID=your-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-your-secret
FRONTEND_URL=http://localhost:3000
```

### 3. Run Server

```bash
python app.py
```

Server starts at: http://localhost:8000

API Documentation: http://localhost:8000/docs

## API Endpoints

### Authentication
- `POST /auth/google` - Login with Google ID token

### User
- `GET /user/me?user_email={email}` - Get user profile
- `PATCH /user/settings?user_email={email}` - Update settings

### Briefs
- `GET /briefs?user_email={email}` - List all briefs
- `GET /briefs/{id}?user_email={email}` - Get specific brief
- `PATCH /briefs/{id}/read?user_email={email}` - Mark as read

### Schedule
- `GET /schedule/today?user_email={email}` - Get today's events

### Health
- `GET /health` - Health check

## Background Jobs

- **Morning briefs**: Runs at 8:00 AM daily
- **Meeting checks**: Runs every minute

## Database

SQLite database: `chief_of_staff.db`

Tables:
- `users` - User accounts
- `briefs` - Generated briefs

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| OPENAI_API_KEY | OpenAI API key | Yes |
| SECRET_KEY | Random secret for JWT | Yes |
| GOOGLE_CLIENT_ID | OAuth client ID | Yes |
| GOOGLE_CLIENT_SECRET | OAuth client secret | Yes |
| FRONTEND_URL | Frontend URL for CORS | Yes |
| MORNING_BRIEF_TIME | Time for morning briefs | No (default: 08:00) |

## Development

### Run in development mode with auto-reload

```bash
uvicorn app:app --reload --port 8000
```

### View logs

Logs print to console.

### Database management

View database:
```bash
sqlite3 chief_of_staff.db
.tables
SELECT * FROM users;
SELECT * FROM briefs;
```

## Deployment

### Option 1: Heroku

```bash
# Install Heroku CLI
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your-key
heroku config:set GOOGLE_CLIENT_ID=your-id
# ... set other env vars
git push heroku main
```

### Option 2: Railway

1. Connect GitHub repo
2. Add environment variables
3. Deploy

### Option 3: VPS

```bash
# Install dependencies
pip install -r requirements.txt

# Run with systemd (see systemd.service example)
sudo systemctl start chief-of-staff
```

## Architecture

```
app.py
├── Configuration (OpenAI, Database URL, Scopes)
├── Database Models (User, Brief)
├── Database Connection (SQLAlchemy AsyncSession)
├── Google Services (Calendar, Gmail, OAuth)
├── OpenAI Services (Brief generation)
├── Background Scheduler (Morning & meeting routines)
└── FastAPI Application (API endpoints)
```

## Troubleshooting

**Database locked error:**
- Close any other connections to the database
- Restart the server

**Google API errors:**
- Check OAuth credentials are correct
- Verify user has granted permissions
- Check token hasn't expired

**OpenAI errors:**
- Verify API key is valid
- Check you have credits
- Review rate limits

## Testing

Test health endpoint:
```bash
curl http://localhost:8000/health
```

Test with frontend:
1. Start backend: `python app.py`
2. Start frontend on port 3000
3. Login and verify briefs generate

## License

MIT
