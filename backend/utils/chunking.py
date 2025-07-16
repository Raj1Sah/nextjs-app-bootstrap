import re
import time
from typing import List, Dict, Any
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def recursive_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> Dict[str, Any]:
    """
    Recursive text chunking with overlap
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        Dictionary containing chunks and metadata
    """
    start_time = time.time()
    
    if not text or not text.strip():
        raise ValueError("Empty text provided for chunking.")
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If we're not at the end, try to break at word boundary
        if end < len(text):
            # Look for the last space within the chunk
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap if end - overlap > start else end
    
    processing_time = time.time() - start_time
    avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
    
    return {
        "chunks": chunks,
        "method": "recursive",
        "num_chunks": len(chunks),
        "avg_chunk_size": avg_chunk_size,
        "processing_time": processing_time,
        "parameters": {"chunk_size": chunk_size, "overlap": overlap}
    }

def semantic_chunk(text: str, max_chunk_size: int = 500) -> Dict[str, Any]:
    """
    Semantic chunking based on sentence boundaries
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum size of each chunk
    
    Returns:
        Dictionary containing chunks and metadata
    """
    start_time = time.time()
    
    if not text or not text.strip():
        raise ValueError("Empty text provided for chunking.")
    
    # Split text into sentences
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check if adding this sentence would exceed max_chunk_size
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            # Save current chunk if it's not empty
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Start new chunk with current sentence
            current_chunk = sentence + " "
    
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    processing_time = time.time() - start_time
    avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
    
    return {
        "chunks": chunks,
        "method": "semantic",
        "num_chunks": len(chunks),
        "avg_chunk_size": avg_chunk_size,
        "processing_time": processing_time,
        "parameters": {"max_chunk_size": max_chunk_size}
    }

def custom_chunk(text: str, delimiter: str = "\n\n", min_chunk_size: int = 100, max_chunk_size: int = 800) -> Dict[str, Any]:
    """
    Custom chunking based on paragraph breaks with size constraints
    
    Args:
        text: Input text to chunk
        delimiter: Delimiter to split on (default: double newline for paragraphs)
        min_chunk_size: Minimum size of each chunk
        max_chunk_size: Maximum size of each chunk
    
    Returns:
        Dictionary containing chunks and metadata
    """
    start_time = time.time()
    
    if not text or not text.strip():
        raise ValueError("Empty text provided for chunking.")
    
    # Split by delimiter (paragraphs)
    paragraphs = [p.strip() for p in text.split(delimiter) if p.strip()]
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If paragraph alone is too big, split it further
        if len(paragraph) > max_chunk_size:
            # Save current chunk if it exists
            if current_chunk and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split large paragraph by sentences
            sentences = sent_tokenize(paragraph)
            temp_chunk = ""
            
            for sentence in sentences:
                if len(temp_chunk) + len(sentence) + 1 <= max_chunk_size:
                    temp_chunk += sentence + " "
                else:
                    if temp_chunk.strip():
                        chunks.append(temp_chunk.strip())
                    temp_chunk = sentence + " "
            
            if temp_chunk.strip():
                current_chunk = temp_chunk
        else:
            # Check if adding this paragraph would exceed max_chunk_size
            if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                # Save current chunk if it meets minimum size
                if current_chunk and len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with current paragraph
                current_chunk = paragraph + "\n\n"
    
    # Add the last chunk if it meets minimum size
    if current_chunk and len(current_chunk.strip()) >= min_chunk_size:
        chunks.append(current_chunk.strip())
    elif current_chunk and chunks:
        # If last chunk is too small, merge with previous chunk
        chunks[-1] += "\n\n" + current_chunk.strip()
    
    processing_time = time.time() - start_time
    avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
    
    return {
        "chunks": chunks,
        "method": "custom",
        "num_chunks": len(chunks),
        "avg_chunk_size": avg_chunk_size,
        "processing_time": processing_time,
        "parameters": {
            "delimiter": delimiter,
            "min_chunk_size": min_chunk_size,
            "max_chunk_size": max_chunk_size
        }
    }

def get_chunking_function(method: str):
    """
    Get the appropriate chunking function based on method name
    
    Args:
        method: Chunking method name ('recursive', 'semantic', 'custom')
    
    Returns:
        Chunking function
    """
    chunking_methods = {
        "recursive": recursive_chunk,
        "semantic": semantic_chunk,
        "custom": custom_chunk
    }
    
    if method not in chunking_methods:
        raise ValueError(f"Invalid chunking method: {method}. Available methods: {list(chunking_methods.keys())}")
    
    return chunking_methods[method]

def compare_chunking_methods(text: str) -> Dict[str, Any]:
    """
    Compare all chunking methods on the same text
    
    Args:
        text: Input text to analyze
    
    Returns:
        Dictionary with comparison results
    """
    methods = ["recursive", "semantic", "custom"]
    results = {}
    
    for method in methods:
        try:
            chunking_func = get_chunking_function(method)
            result = chunking_func(text)
            results[method] = result
        except Exception as e:
            results[method] = {"error": str(e)}
    
    return results
