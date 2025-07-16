# Deployment Guide

## Quick Start

### 1. Prerequisites
- Python 3.9+
- Docker (for Qdrant and Redis)
- OpenAI API Key

### 2. Setup External Services

**Start Qdrant (Vector Database):**
```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

**Start Redis (Memory Layer):**
```bash
docker run -d -p 6379:6379 --name redis redis:alpine
```

### 3. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Run the Application
```bash
python main.py
```

### 6. Test the System
```bash
python test_system.py
```

## Production Deployment

### Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_URL=postgresql://user:pass@postgres:5432/ragdb
      - QDRANT_HOST=qdrant
      - REDIS_HOST=redis
    depends_on:
      - postgres
      - qdrant
      - redis

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ragdb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  qdrant_data:
  redis_data:
```

### Environment Variables

Required:
- `OPENAI_API_KEY`: Your OpenAI API key
- `SMTP_USER`: Email for sending confirmations
- `SMTP_PASS`: Email password/app password

Optional (with defaults):
- `DB_URL`: Database connection string
- `QDRANT_HOST`: Qdrant host (default: localhost)
- `REDIS_HOST`: Redis host (default: localhost)

## API Endpoints

### Document Processing
- `POST /api/upload` - Upload documents
- `GET /api/files` - List files
- `POST /api/compare-chunking` - Compare chunking methods

### RAG Agent
- `POST /api/query` - Query documents
- `GET /api/conversation/{session_id}` - Get conversation
- `POST /api/compare-similarity` - Compare similarity algorithms

### Interview Booking
- `POST /api/booking` - Book interview
- `GET /api/bookings` - List bookings
- `DELETE /api/bookings/{id}` - Cancel booking

### System
- `GET /health` - Health check
- `GET /metrics` - Performance metrics
- `GET /docs` - API documentation

## Monitoring

The system provides comprehensive monitoring:

1. **Health Checks**: `/health` endpoint
2. **Metrics**: `/metrics` endpoint  
3. **Logging**: Structured logs for all operations
4. **Performance**: Built-in timing and performance tracking

## Security

1. **API Keys**: Store securely in environment variables
2. **CORS**: Configure for your domain
3. **Rate Limiting**: Implement as needed
4. **File Validation**: Built-in file type and size validation
5. **Input Sanitization**: Pydantic models validate all inputs

## Scaling

1. **Horizontal**: Multiple FastAPI instances behind load balancer
2. **Database**: PostgreSQL with connection pooling
3. **Vector DB**: Qdrant cluster
4. **Caching**: Redis cluster
5. **File Storage**: Object storage (S3, etc.)
