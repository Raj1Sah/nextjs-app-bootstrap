import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api import upload, agent, booking
from db import init_db
from config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="RAG-based Document Processing and Interview Booking API",
    description="""
    A comprehensive backend system with two main functionalities:
    
    1. **Document Processing API**: Upload PDF/TXT files, extract and chunk text, generate embeddings, and store in vector database
    2. **RAG-based Agent API**: Query documents using a conversational agent with memory, and book interviews
    
    Features:
    - Multiple chunking strategies (recursive, semantic, custom)
    - Vector storage in Qdrant with similarity search
    - LangChain-based conversational agent with Redis memory
    - Interview booking with email confirmation
    - Performance comparison and analytics
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database and perform startup checks"""
    try:
        # Initialize database tables
        init_db()
        logger.info("Database initialized successfully")
        
        # Log configuration
        logger.info("=== Backend System Configuration ===")
        logger.info(f"Database URL: {DB_URL}")
        logger.info(f"Qdrant Host: {QDRANT_HOST}:{QDRANT_PORT}")
        logger.info(f"Redis Host: {REDIS_HOST}:{REDIS_PORT}")
        logger.info(f"SMTP Host: {SMTP_HOST}:{SMTP_PORT}")
        logger.info(f"Embedding Model: {EMBEDDING_MODEL}")
        logger.info(f"Default Chunking Method: {DEFAULT_CHUNKING_METHOD}")
        logger.info("=====================================")
        
        # Test external connections
        try:
            from qdrant_client import QdrantClient
            if QDRANT_API_KEY:
                qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
            else:
                qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            qdrant_client.get_collections()
            logger.info("✓ Qdrant connection successful")
        except Exception as e:
            logger.warning(f"✗ Qdrant connection failed: {str(e)}")
        
        try:
            import redis
            redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)
            redis_client.ping()
            logger.info("✓ Redis connection successful")
        except Exception as e:
            logger.warning(f"✗ Redis connection failed: {str(e)}")
        
        try:
            from smtp_utils import test_smtp_connection
            if test_smtp_connection():
                logger.info("✓ SMTP connection successful")
            else:
                logger.warning("✗ SMTP connection failed")
        except Exception as e:
            logger.warning(f"✗ SMTP connection test failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise e

# Include routers
app.include_router(upload.router, prefix="/api", tags=["Document Upload & Processing"])
app.include_router(agent.router, prefix="/api", tags=["RAG Agent & Query"])
app.include_router(booking.router, prefix="/api", tags=["Interview Booking"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG-based Document Processing and Interview Booking API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "upload": "/api/upload",
            "query": "/api/query",
            "booking": "/api/booking",
            "health": "/health"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check for all system components"""
    health_status = {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "components": {
            "api": "healthy",
            "database": "unknown",
            "qdrant": "unknown",
            "redis": "unknown",
            "smtp": "unknown"
        }
    }
    
    # Check database
    try:
        from db import get_db_session
        db = get_db_session()
        db.execute("SELECT 1")
        db.close()
        health_status["components"]["database"] = "healthy"
    except Exception as e:
        health_status["components"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        if QDRANT_API_KEY:
            qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
        else:
            qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        qdrant_client.get_collections()
        health_status["components"]["qdrant"] = "healthy"
    except Exception as e:
        health_status["components"]["qdrant"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        import redis
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)
        redis_client.ping()
        health_status["components"]["redis"] = "healthy"
    except Exception as e:
        health_status["components"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check SMTP
    try:
        from smtp_utils import test_smtp_connection
        if test_smtp_connection():
            health_status["components"]["smtp"] = "healthy"
        else:
            health_status["components"]["smtp"] = "unhealthy: connection failed"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["smtp"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

# Performance metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get system performance metrics"""
    try:
        from db import get_db_session, FileMetadata, ChunkingPerformance, EmbeddingPerformance
        from sqlalchemy import func
        
        db = get_db_session()
        
        # File processing metrics
        total_files = db.query(func.count(FileMetadata.id)).scalar()
        avg_processing_time = db.query(func.avg(FileMetadata.processing_time)).scalar()
        total_chunks = db.query(func.sum(FileMetadata.num_chunks)).scalar()
        
        # Chunking performance
        chunking_stats = db.query(
            ChunkingPerformance.chunking_method,
            func.avg(ChunkingPerformance.processing_time).label('avg_time'),
            func.avg(ChunkingPerformance.avg_chunk_size).label('avg_size'),
            func.count(ChunkingPerformance.id).label('count')
        ).group_by(ChunkingPerformance.chunking_method).all()
        
        # Embedding performance
        embedding_stats = db.query(
            EmbeddingPerformance.embedding_model,
            func.avg(EmbeddingPerformance.embedding_time).label('avg_time'),
            func.count(EmbeddingPerformance.id).label('count')
        ).group_by(EmbeddingPerformance.embedding_model).all()
        
        db.close()
        
        return {
            "file_processing": {
                "total_files": total_files or 0,
                "avg_processing_time": float(avg_processing_time) if avg_processing_time else 0,
                "total_chunks": total_chunks or 0
            },
            "chunking_performance": [
                {
                    "method": stat.chunking_method,
                    "avg_processing_time": float(stat.avg_time),
                    "avg_chunk_size": float(stat.avg_size),
                    "usage_count": stat.count
                } for stat in chunking_stats
            ],
            "embedding_performance": [
                {
                    "model": stat.embedding_model,
                    "avg_embedding_time": float(stat.avg_time),
                    "usage_count": stat.count
                } for stat in embedding_stats
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
