import time
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import openai
from config import EMBEDDING_MODEL, OPENAI_API_KEY

class EmbeddingGenerator:
    """Class to handle different embedding generation methods"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model based on model name"""
        try:
            if self.model_name.startswith("text-embedding"):
                # OpenAI embedding model
                if not OPENAI_API_KEY:
                    raise ValueError("OpenAI API key not provided for OpenAI embedding model")
                openai.api_key = OPENAI_API_KEY
                self.model_type = "openai"
                # Test embedding to get dimension
                test_response = openai.Embedding.create(
                    input="test",
                    model=self.model_name
                )
                self.embedding_dimension = len(test_response['data'][0]['embedding'])
            else:
                # Sentence Transformers model
                self.model = SentenceTransformer(self.model_name)
                self.model_type = "sentence_transformer"
                # Get embedding dimension
                test_embedding = self.model.encode("test")
                self.embedding_dimension = len(test_embedding)
                
        except Exception as e:
            print(f"Error initializing embedding model {self.model_name}: {str(e)}")
            # Fallback to a simple model
            self.model_name = "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name)
            self.model_type = "sentence_transformer"
            test_embedding = self.model.encode("test")
            self.embedding_dimension = len(test_embedding)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding
        """
        if not text or not text.strip():
            raise ValueError("Empty text provided for embedding generation.")
        
        try:
            if self.model_type == "openai":
                response = openai.Embedding.create(
                    input=text,
                    model=self.model_name
                )
                return response['data'][0]['embedding']
            else:
                embedding = self.model.encode(text)
                return embedding.tolist()
                
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            raise ValueError("Empty text list provided for embedding generation.")
        
        try:
            if self.model_type == "openai":
                response = openai.Embedding.create(
                    input=texts,
                    model=self.model_name
                )
                return [item['embedding'] for item in response['data']]
            else:
                embeddings = self.model.encode(texts)
                return embeddings.tolist()
                
        except Exception as e:
            raise Exception(f"Error generating batch embeddings: {str(e)}")

def generate_embedding(text: str, model_name: str = EMBEDDING_MODEL) -> List[float]:
    """
    Generate embedding for a single text (convenience function)
    
    Args:
        text: Input text to embed
        model_name: Name of the embedding model to use
        
    Returns:
        List of floats representing the embedding
    """
    generator = EmbeddingGenerator(model_name)
    return generator.generate_embedding(text)

def generate_embeddings_with_metadata(chunks: List[str], model_name: str = EMBEDDING_MODEL) -> Dict[str, Any]:
    """
    Generate embeddings for chunks with performance metadata
    
    Args:
        chunks: List of text chunks to embed
        model_name: Name of the embedding model to use
        
    Returns:
        Dictionary containing embeddings and metadata
    """
    start_time = time.time()
    
    generator = EmbeddingGenerator(model_name)
    embeddings = generator.generate_embeddings_batch(chunks)
    
    processing_time = time.time() - start_time
    
    return {
        "embeddings": embeddings,
        "model_name": model_name,
        "embedding_dimension": generator.embedding_dimension,
        "num_embeddings": len(embeddings),
        "processing_time": processing_time,
        "avg_time_per_embedding": processing_time / len(embeddings) if embeddings else 0
    }

def compare_embedding_models(text_chunks: List[str], models: List[str] = None) -> Dict[str, Any]:
    """
    Compare different embedding models on the same text chunks
    
    Args:
        text_chunks: List of text chunks to embed
        models: List of model names to compare (default: common models)
        
    Returns:
        Dictionary with comparison results
    """
    if models is None:
        models = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2"
        ]
    
    results = {}
    
    for model in models:
        try:
            result = generate_embeddings_with_metadata(text_chunks, model)
            results[model] = result
        except Exception as e:
            results[model] = {"error": str(e)}
    
    return results

def calculate_similarity(embedding1: List[float], embedding2: List[float], method: str = "cosine") -> float:
    """
    Calculate similarity between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        method: Similarity method ("cosine" or "euclidean")
        
    Returns:
        Similarity score
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    if method == "cosine":
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    elif method == "euclidean":
        # Euclidean distance (converted to similarity)
        distance = np.linalg.norm(vec1 - vec2)
        # Convert distance to similarity (higher is more similar)
        return 1 / (1 + distance)
    
    else:
        raise ValueError(f"Unsupported similarity method: {method}")

def find_most_similar_chunks(query_embedding: List[float], chunk_embeddings: List[List[float]], 
                           top_k: int = 5, method: str = "cosine") -> List[Dict[str, Any]]:
    """
    Find most similar chunks to a query embedding
    
    Args:
        query_embedding: Query embedding vector
        chunk_embeddings: List of chunk embedding vectors
        top_k: Number of top similar chunks to return
        method: Similarity method to use
        
    Returns:
        List of dictionaries with chunk indices and similarity scores
    """
    similarities = []
    
    for i, chunk_embedding in enumerate(chunk_embeddings):
        similarity = calculate_similarity(query_embedding, chunk_embedding, method)
        similarities.append({"index": i, "similarity": similarity})
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    
    return similarities[:top_k]
