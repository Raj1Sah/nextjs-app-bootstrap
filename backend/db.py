from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from config import DB_URL

# Create database engine
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if "sqlite" in DB_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FileMetadata(Base):
    """Model for storing file metadata and processing information"""
    __tablename__ = "file_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    chunking_method = Column(String, nullable=False)
    embedding_model = Column(String, nullable=False)
    num_chunks = Column(Integer, nullable=False)
    processing_time = Column(Float, nullable=True)  # Time taken to process in seconds
    metadata_json = Column(Text, nullable=True)  # Additional metadata as JSON string
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)

class Booking(Base):
    """Model for storing interview booking information"""
    __tablename__ = "bookings"
    
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    interview_date = Column(String, nullable=False)
    interview_time = Column(String, nullable=False)
    booking_time = Column(DateTime, default=datetime.datetime.utcnow)
    confirmation_sent = Column(String, default="pending")  # pending, sent, failed

class ChunkingPerformance(Base):
    """Model for storing chunking performance metrics"""
    __tablename__ = "chunking_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, nullable=False)
    chunking_method = Column(String, nullable=False)
    num_chunks = Column(Integer, nullable=False)
    avg_chunk_size = Column(Float, nullable=False)
    processing_time = Column(Float, nullable=False)
    retrieval_accuracy = Column(Float, nullable=True)  # To be populated during testing
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class EmbeddingPerformance(Base):
    """Model for storing embedding performance metrics"""
    __tablename__ = "embedding_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, nullable=False)
    embedding_model = Column(String, nullable=False)
    embedding_time = Column(Float, nullable=False)
    vector_dimension = Column(Integer, nullable=False)
    similarity_algorithm = Column(String, nullable=False)  # cosine, euclidean, etc.
    retrieval_latency = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully")
    except Exception as e:
        print(f"Error creating database tables: {str(e)}")
        raise e

def get_db_session():
    """Get a database session for direct use"""
    return SessionLocal()
