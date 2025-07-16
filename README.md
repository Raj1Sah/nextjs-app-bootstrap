# RAG-based Document Processing and Interview Booking Backend

A comprehensive backend system built with FastAPI that provides two main functionalities:

1. **Document Processing API**: Upload, chunk, embed, and store documents in a vector database
2. **RAG-based Agent API**: Query documents using a conversational agent with memory and book interviews

## Features

### Document Processing
- ✅ PDF and TXT file upload support
- ✅ Multiple text chunking strategies (recursive, semantic, custom)
- ✅ Multiple embedding models support
- ✅ Vector storage in Qdrant with similarity search
- ✅ Performance metrics and comparison tools
- ✅ Metadata storage in relational database

### RAG-based Agent
- ✅ LangChain-based conversational agent (no RetrievalQA)
- ✅ Redis-based conversation memory
- ✅ Custom tools for document search and analysis
- ✅ Similarity algorithm comparison
- ✅ Session-based conversation management

### Interview Booking
- ✅ Form validation and booking creation
- ✅ Email confirmation via SMTP
- ✅ Booking management (list, view, cancel)
- ✅ Date-based booking queries

## Technology Stack

- **Backend Framework**: FastAPI
- **Vector Database**: Qdrant
- **Memory Layer**: Redis
- **LLM Framework**: LangChain with OpenAI GPT-3.5-turbo
- **Embedding Models**: Sentence Transformers
- **Database**: SQLAlchemy (SQLite/PostgreSQL)
- **Email**: SMTP with HTML templates

## Installation

### Prerequisites

1. **Python 3.9+**
2. **Qdrant** (vector database)
3. **Redis** (memory layer)
4. **OpenAI API Key** (for LLM)

### Setup Instructions

1. **Clone and navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional NLTK data**:
   ```python
   import nltk
   nltk.download('punkt')
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Start external services**:

   **Qdrant (using Docker)**:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

   **Redis (using Docker)**:
   ```bash
   docker run -p 6379:6379 redis:alpine
   ```

6. **Run the application**:
   ```bash
   python main.py
   # or
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# Database
DB_URL=sqlite:///./backend.db

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# SMTP
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password

# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### SMTP Configuration

For Gmail:
1. Enable 2-factor authentication
2. Generate an app password
3. Use the app password in `SMTP_PASS`

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Document Processing
- `POST /api/upload` - Upload and process documents
- `GET /api/files` - List all uploaded files
- `GET /api/files/{file_id}` - Get file details
- `POST /api/compare-chunking` - Compare chunking methods

#### RAG Agent
- `POST /api/query` - Query documents using the agent
- `GET /api/conversation/{session_id}` - Get conversation history
- `DELETE /api/conversation/{session_id}` - Clear conversation
- `POST /api/compare-similarity` - Compare similarity algorithms

#### Interview Booking
- `POST /api/booking` - Book an interview
- `GET /api/bookings` - List all bookings
- `GET /api/bookings/{booking_id}` - Get booking details
- `DELETE /api/bookings/{booking_id}` - Cancel booking

#### System
- `GET /health` - Health check
- `GET /metrics` - Performance metrics

## Usage Examples

### 1. Upload a Document

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "chunking_method=semantic" \
  -F "embedding_model=all-mpnet-base-v2"
```

### 2. Query Documents

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings about machine learning?",
    "session_id": "user123",
    "max_results": 5
  }'
```

### 3. Book an Interview

```bash
curl -X POST "http://localhost:8000/api/booking" \
  -H "Content-Type: application/json" \
  -d '{
    "full_name": "John Doe",
    "email": "john@example.com",
    "interview_date": "2024-02-15",
    "interview_time": "14:30"
  }'
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │     Qdrant      │    │      Redis      │
│                 │    │  (Vector DB)    │    │   (Memory)      │
│  ┌───────────┐  │    │                 │    │                 │
│  │ Upload API│  │◄──►│  Embeddings     │    │  Conversations  │
│  └───────────┘  │    │  Similarity     │    │  Sessions       │
│  ┌───────────┐  │    │  Search         │    │                 │
│  │Agent API  │  │◄──►│                 │◄──►│                 │
│  └───────────┘  │    └─────────────────┘    └─────────────────┘
│  ┌───────────┐  │    
│  │Booking API│  │    ┌─────────────────┐    ┌─────────────────┐
│  └───────────┘  │    │   SQLAlchemy    │    │      SMTP       │
│                 │◄──►│   (Metadata)    │    │   (Email)       │
└─────────────────┘    │                 │    │                 │
                       │  File Metadata  │    │  Confirmations  │
                       │  Bookings       │    │  Notifications  │
                       │  Performance    │    │                 │
                       └─────────────────┘    └─────────────────┘
```

## Performance Optimization

### Chunking Strategy Recommendations

- **High Quality**: Semantic chunking with all-mpnet-base-v2
- **Balanced**: Custom chunking with all-MiniLM-L6-v2
- **High Speed**: Recursive chunking with all-MiniLM-L6-v2

### Similarity Search

- **Text Documents**: Cosine similarity (recommended)
- **Numerical Data**: Euclidean distance

See `REPORT.md` for detailed performance analysis.

## Testing

### Manual Testing

1. **Health Check**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Upload Test File**:
   ```bash
   echo "This is a test document with some content." > test.txt
   curl -X POST "http://localhost:8000/api/upload" \
     -F "file=@test.txt"
   ```

3. **Query Test**:
   ```bash
   curl -X POST "http://localhost:8000/api/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "test content"}'
   ```

### Performance Testing

The system includes built-in performance metrics:
- Processing times for each component
- Chunking strategy comparisons
- Embedding model performance
- Similarity algorithm effectiveness

## Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**:
   - Ensure Qdrant is running on the specified host/port
   - Check firewall settings

2. **Redis Connection Failed**:
   - Ensure Redis is running
   - Check Redis password configuration

3. **SMTP Authentication Failed**:
   - Verify email credentials
   - Use app passwords for Gmail
   - Check SMTP server settings

4. **OpenAI API Errors**:
   - Verify API key is valid
   - Check API quota and billing

5. **File Upload Errors**:
   - Check file size limits
   - Ensure file format is supported (PDF/TXT)
   - Verify file is not corrupted

### Logs

The application provides detailed logging:
- Startup configuration
- Processing steps
- Error details
- Performance metrics

## Production Deployment

### Security Considerations

1. **Environment Variables**: Use secure secret management
2. **API Keys**: Rotate regularly and store securely
3. **Database**: Use PostgreSQL with proper authentication
4. **CORS**: Configure appropriately for your domain
5. **Rate Limiting**: Implement API rate limiting
6. **File Validation**: Enhanced file type and content validation

### Scaling Considerations

1. **Horizontal Scaling**: Multiple FastAPI instances behind load balancer
2. **Database**: Use PostgreSQL with connection pooling
3. **Vector Database**: Qdrant cluster for high availability
4. **Caching**: Redis cluster for distributed caching
5. **File Storage**: Use object storage (S3, etc.) for uploaded files

### Monitoring

1. **Health Checks**: Built-in health endpoints
2. **Metrics**: Performance metrics endpoint
3. **Logging**: Structured logging for analysis
4. **Alerting**: Set up alerts for system failures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Check system logs
4. Create an issue in the repository

---

**Note**: This is a production-ready backend system designed for scalability and maintainability. The modular architecture allows for easy extension and customization based on specific requirements.
