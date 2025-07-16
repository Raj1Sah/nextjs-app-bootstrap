import io
import time
import json
import logging
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query
from sqlalchemy.orm import Session
import PyPDF2
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from db import get_db, FileMetadata, ChunkingPerformance, EmbeddingPerformance
from utils.chunking import get_chunking_function, compare_chunking_methods
from utils.embeddings import generate_embeddings_with_metadata, EmbeddingGenerator
from config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, QDRANT_COLLECTION_NAME,
    MAX_FILE_SIZE, ALLOWED_EXTENSIONS, DEFAULT_CHUNKING_METHOD,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, EMBEDDING_MODEL
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize Qdrant client
try:
    if QDRANT_API_KEY:
        qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY
        )
    else:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    logger.info("Qdrant client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant client: {str(e)}")
    qdrant_client = None

def ensure_collection_exists(collection_name: str, vector_size: int):
    """Ensure that the Qdrant collection exists"""
    try:
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name not in collection_names:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Created collection: {collection_name}")
        else:
            logger.info(f"Collection {collection_name} already exists")
    except Exception as e:
        logger.error(f"Error ensuring collection exists: {str(e)}")
        raise e

def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from uploaded file"""
    try:
        file_content = file.file.read()
        
        if file.filename.lower().endswith('.txt'):
            return file_content.decode('utf-8')
        
        elif file.filename.lower().endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text_content = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
                    continue
            
            if not text_content.strip():
                raise ValueError("No text could be extracted from the PDF file")
            
            return text_content
        
        else:
            raise ValueError(f"Unsupported file type: {file.filename}")
            
    except UnicodeDecodeError:
        raise ValueError("Unable to decode text file. Please ensure it's in UTF-8 format.")
    except Exception as e:
        raise ValueError(f"Error extracting text from file: {str(e)}")

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    chunking_method: str = Query(DEFAULT_CHUNKING_METHOD, description="Chunking method: recursive, semantic, or custom"),
    chunk_size: int = Query(DEFAULT_CHUNK_SIZE, description="Chunk size for recursive chunking"),
    chunk_overlap: int = Query(DEFAULT_CHUNK_OVERLAP, description="Chunk overlap for recursive chunking"),
    embedding_model: str = Query(EMBEDDING_MODEL, description="Embedding model to use"),
    db: Session = Depends(get_db)
):
    """
    Upload and process a file (PDF or TXT)
    - Extract text from the file
    - Chunk the text using specified method
    - Generate embeddings for each chunk
    - Store embeddings in Qdrant vector database
    - Save metadata in relational database
    """
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_extension = "." + file.filename.split(".")[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed extensions: {ALLOWED_EXTENSIONS}"
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file provided")
    
    # Validate chunking method
    if chunking_method not in ["recursive", "semantic", "custom"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid chunking method. Use: recursive, semantic, or custom"
        )
    
    start_time = time.time()
    
    try:
        # Extract text from file
        logger.info(f"Extracting text from file: {file.filename}")
        text_content = extract_text_from_file(file)
        
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No text content found in file")
        
        # Chunk the text
        logger.info(f"Chunking text using method: {chunking_method}")
        chunking_func = get_chunking_function(chunking_method)
        
        if chunking_method == "recursive":
            chunk_result = chunking_func(text_content, chunk_size, chunk_overlap)
        else:
            chunk_result = chunking_func(text_content)
        
        chunks = chunk_result["chunks"]
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from text")
        
        # Generate embeddings
        logger.info(f"Generating embeddings using model: {embedding_model}")
        embedding_result = generate_embeddings_with_metadata(chunks, embedding_model)
        embeddings = embedding_result["embeddings"]
        embedding_dimension = embedding_result["embedding_dimension"]
        
        # Ensure Qdrant collection exists
        if qdrant_client:
            ensure_collection_exists(QDRANT_COLLECTION_NAME, embedding_dimension)
            
            # Store embeddings in Qdrant
            logger.info("Storing embeddings in Qdrant")
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = f"{file.filename}_{int(time.time())}_{i}"
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "file_name": file.filename,
                        "chunk_index": i,
                        "chunking_method": chunking_method,
                        "embedding_model": embedding_model,
                        "upload_time": time.time()
                    }
                )
                points.append(point)
            
            # Batch upsert to Qdrant
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=points
            )
            logger.info(f"Successfully stored {len(points)} embeddings in Qdrant")
        else:
            logger.warning("Qdrant client not available, skipping vector storage")
        
        # Calculate total processing time
        total_processing_time = time.time() - start_time
        
        # Save metadata to database
        metadata = FileMetadata(
            file_name=file.filename,
            file_size=file_size,
            chunking_method=chunking_method,
            embedding_model=embedding_model,
            num_chunks=len(chunks),
            processing_time=total_processing_time,
            metadata_json=json.dumps({
                "chunk_params": chunk_result.get("parameters", {}),
                "embedding_params": {
                    "model": embedding_model,
                    "dimension": embedding_dimension
                },
                "text_length": len(text_content),
                "avg_chunk_size": chunk_result.get("avg_chunk_size", 0)
            })
        )
        
        db.add(metadata)
        db.commit()
        db.refresh(metadata)
        
        # Save chunking performance metrics
        chunking_perf = ChunkingPerformance(
            file_id=metadata.id,
            chunking_method=chunking_method,
            num_chunks=len(chunks),
            avg_chunk_size=chunk_result.get("avg_chunk_size", 0),
            processing_time=chunk_result.get("processing_time", 0)
        )
        db.add(chunking_perf)
        
        # Save embedding performance metrics
        embedding_perf = EmbeddingPerformance(
            file_id=metadata.id,
            embedding_model=embedding_model,
            embedding_time=embedding_result.get("processing_time", 0),
            vector_dimension=embedding_dimension,
            similarity_algorithm="cosine"  # Default for Qdrant
        )
        db.add(embedding_perf)
        
        db.commit()
        
        logger.info(f"File processing completed successfully: {file.filename}")
        
        return {
            "message": "File processed successfully",
            "file_id": metadata.id,
            "file_name": file.filename,
            "file_size": file_size,
            "chunking_method": chunking_method,
            "embedding_model": embedding_model,
            "num_chunks": len(chunks),
            "embedding_dimension": embedding_dimension,
            "processing_time": total_processing_time,
            "text_length": len(text_content),
            "avg_chunk_size": chunk_result.get("avg_chunk_size", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/files")
async def list_files(db: Session = Depends(get_db)):
    """List all uploaded files with their metadata"""
    try:
        files = db.query(FileMetadata).order_by(FileMetadata.upload_time.desc()).all()
        
        file_list = []
        for file_meta in files:
            file_info = {
                "id": file_meta.id,
                "file_name": file_meta.file_name,
                "file_size": file_meta.file_size,
                "chunking_method": file_meta.chunking_method,
                "embedding_model": file_meta.embedding_model,
                "num_chunks": file_meta.num_chunks,
                "processing_time": file_meta.processing_time,
                "upload_time": file_meta.upload_time.isoformat(),
                "metadata": json.loads(file_meta.metadata_json) if file_meta.metadata_json else {}
            }
            file_list.append(file_info)
        
        return {
            "files": file_list,
            "total_files": len(file_list)
        }
        
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

@router.get("/files/{file_id}")
async def get_file_details(file_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific file"""
    try:
        file_meta = db.query(FileMetadata).filter(FileMetadata.id == file_id).first()
        
        if not file_meta:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get performance metrics
        chunking_perf = db.query(ChunkingPerformance).filter(
            ChunkingPerformance.file_id == file_id
        ).first()
        
        embedding_perf = db.query(EmbeddingPerformance).filter(
            EmbeddingPerformance.file_id == file_id
        ).first()
        
        return {
            "file_info": {
                "id": file_meta.id,
                "file_name": file_meta.file_name,
                "file_size": file_meta.file_size,
                "chunking_method": file_meta.chunking_method,
                "embedding_model": file_meta.embedding_model,
                "num_chunks": file_meta.num_chunks,
                "processing_time": file_meta.processing_time,
                "upload_time": file_meta.upload_time.isoformat(),
                "metadata": json.loads(file_meta.metadata_json) if file_meta.metadata_json else {}
            },
            "chunking_performance": {
                "method": chunking_perf.chunking_method if chunking_perf else None,
                "num_chunks": chunking_perf.num_chunks if chunking_perf else None,
                "avg_chunk_size": chunking_perf.avg_chunk_size if chunking_perf else None,
                "processing_time": chunking_perf.processing_time if chunking_perf else None
            } if chunking_perf else None,
            "embedding_performance": {
                "model": embedding_perf.embedding_model if embedding_perf else None,
                "embedding_time": embedding_perf.embedding_time if embedding_perf else None,
                "vector_dimension": embedding_perf.vector_dimension if embedding_perf else None,
                "similarity_algorithm": embedding_perf.similarity_algorithm if embedding_perf else None
            } if embedding_perf else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting file details: {str(e)}")

@router.post("/compare-chunking")
async def compare_chunking_methods_endpoint(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Compare different chunking methods on the same file
    """
    try:
        # Extract text from file
        text_content = extract_text_from_file(file)
        
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No text content found in file")
        
        # Compare chunking methods
        comparison_results = compare_chunking_methods(text_content)
        
        return {
            "file_name": file.filename,
            "text_length": len(text_content),
            "chunking_comparison": comparison_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing chunking methods: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing chunking methods: {str(e)}")
