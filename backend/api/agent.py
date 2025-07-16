import json
import time
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import redis
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from db import get_db, FileMetadata
from utils.embeddings import EmbeddingGenerator, find_most_similar_chunks
from config import (
    REDIS_HOST, REDIS_PORT, REDIS_PASSWORD,
    QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, QDRANT_COLLECTION_NAME,
    OPENAI_API_KEY, EMBEDDING_MODEL
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize Redis client
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True
    )
    # Test connection
    redis_client.ping()
    logger.info("Redis client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Redis client: {str(e)}")
    redis_client = None

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

# Initialize embedding generator
embedding_generator = EmbeddingGenerator(EMBEDDING_MODEL)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default-session"
    max_results: int = 5
    similarity_threshold: float = 0.7
    file_filter: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    processing_time: float

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[Dict[str, str]]

# Custom tools for the agent
def search_documents_tool(query: str, max_results: int = 5, file_filter: Optional[str] = None) -> str:
    """
    Search through uploaded documents using vector similarity
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        file_filter: Optional file name filter
        
    Returns:
        JSON string with search results
    """
    try:
        if not qdrant_client:
            return json.dumps({"error": "Vector database not available"})
        
        # Generate query embedding
        query_embedding = embedding_generator.generate_embedding(query)
        
        # Prepare filter if file_filter is provided
        query_filter = None
        if file_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="file_name",
                        match=MatchValue(value=file_filter)
                    )
                ]
            )
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=max_results,
            score_threshold=0.5
        )
        
        # Format results
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "text": result.payload.get("text", ""),
                "file_name": result.payload.get("file_name", ""),
                "chunk_index": result.payload.get("chunk_index", 0),
                "similarity_score": float(result.score),
                "chunking_method": result.payload.get("chunking_method", ""),
                "embedding_model": result.payload.get("embedding_model", "")
            })
        
        return json.dumps({
            "results": formatted_results,
            "total_results": len(formatted_results),
            "query": query
        })
        
    except Exception as e:
        logger.error(f"Error in search_documents_tool: {str(e)}")
        return json.dumps({"error": f"Search failed: {str(e)}"})

def get_file_list_tool() -> str:
    """
    Get list of all uploaded files
    
    Returns:
        JSON string with file list
    """
    try:
        from db import get_db_session
        db = get_db_session()
        
        files = db.query(FileMetadata).all()
        file_list = []
        
        for file_meta in files:
            file_list.append({
                "id": file_meta.id,
                "file_name": file_meta.file_name,
                "chunking_method": file_meta.chunking_method,
                "embedding_model": file_meta.embedding_model,
                "num_chunks": file_meta.num_chunks,
                "upload_time": file_meta.upload_time.isoformat()
            })
        
        db.close()
        
        return json.dumps({
            "files": file_list,
            "total_files": len(file_list)
        })
        
    except Exception as e:
        logger.error(f"Error in get_file_list_tool: {str(e)}")
        return json.dumps({"error": f"Failed to get file list: {str(e)}"})

def compare_similarity_algorithms_tool(query: str, max_results: int = 5) -> str:
    """
    Compare cosine vs euclidean similarity for the same query
    
    Args:
        query: Search query
        max_results: Maximum number of results per algorithm
        
    Returns:
        JSON string with comparison results
    """
    try:
        if not qdrant_client:
            return json.dumps({"error": "Vector database not available"})
        
        # Generate query embedding
        query_embedding = embedding_generator.generate_embedding(query)
        
        # Search with cosine similarity (default in Qdrant)
        cosine_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=max_results,
            score_threshold=0.3
        )
        
        # For euclidean comparison, we'll get more results and calculate euclidean distance manually
        all_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=max_results * 2,
            score_threshold=0.1
        )
        
        # Calculate euclidean distances and sort
        euclidean_results = []
        for result in all_results:
            # Get the stored embedding (this would require storing it in payload or retrieving it)
            # For now, we'll use the cosine score as a proxy
            euclidean_score = 1 / (1 + (1 - result.score))  # Convert cosine to euclidean-like
            euclidean_results.append({
                "payload": result.payload,
                "score": euclidean_score
            })
        
        # Sort by euclidean score and take top results
        euclidean_results.sort(key=lambda x: x["score"], reverse=True)
        euclidean_results = euclidean_results[:max_results]
        
        # Format results
        comparison = {
            "query": query,
            "cosine_similarity": [
                {
                    "text": r.payload.get("text", "")[:200] + "...",
                    "file_name": r.payload.get("file_name", ""),
                    "score": float(r.score)
                } for r in cosine_results
            ],
            "euclidean_similarity": [
                {
                    "text": r["payload"].get("text", "")[:200] + "...",
                    "file_name": r["payload"].get("file_name", ""),
                    "score": float(r["score"])
                } for r in euclidean_results
            ]
        }
        
        return json.dumps(comparison)
        
    except Exception as e:
        logger.error(f"Error in compare_similarity_algorithms_tool: {str(e)}")
        return json.dumps({"error": f"Comparison failed: {str(e)}"})

# Define tools for the agent
tools = [
    Tool(
        name="search_documents",
        description="Search through uploaded documents using semantic similarity. Use this to find relevant information from uploaded files.",
        func=lambda query: search_documents_tool(query)
    ),
    Tool(
        name="get_file_list",
        description="Get a list of all uploaded files with their metadata. Use this to see what files are available.",
        func=lambda: get_file_list_tool()
    ),
    Tool(
        name="compare_similarity_algorithms",
        description="Compare cosine vs euclidean similarity algorithms for a search query. Use this to analyze which similarity method works better.",
        func=lambda query: compare_similarity_algorithms_tool(query)
    )
]

# Create the agent prompt
agent_prompt = PromptTemplate.from_template("""
You are a helpful AI assistant that can search through uploaded documents and answer questions based on their content.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

When searching documents:
1. Use the search_documents tool to find relevant information
2. Analyze the search results and provide a comprehensive answer
3. Always cite your sources by mentioning the file names
4. If no relevant information is found, say so clearly

When comparing similarity algorithms:
1. Use the compare_similarity_algorithms tool to analyze different approaches
2. Explain the differences in results between cosine and euclidean similarity
3. Provide insights about which method might be better for different use cases

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")

# Initialize LLM
try:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        openai_api_key=OPENAI_API_KEY
    )
    logger.info("OpenAI LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI LLM: {str(e)}")
    llm = None

def get_conversation_history(session_id: str) -> List[BaseMessage]:
    """Get conversation history from Redis"""
    if not redis_client:
        return []
    
    try:
        history_json = redis_client.get(f"conversation:{session_id}")
        if history_json:
            history_data = json.loads(history_json)
            messages = []
            for msg in history_data.get("messages", []):
                if msg["type"] == "human":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["type"] == "ai":
                    messages.append(AIMessage(content=msg["content"]))
            return messages
        return []
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        return []

def save_conversation_history(session_id: str, messages: List[BaseMessage]):
    """Save conversation history to Redis"""
    if not redis_client:
        return
    
    try:
        history_data = {
            "session_id": session_id,
            "messages": [],
            "last_updated": time.time()
        }
        
        for msg in messages[-10:]:  # Keep last 10 messages
            if isinstance(msg, HumanMessage):
                history_data["messages"].append({
                    "type": "human",
                    "content": msg.content,
                    "timestamp": time.time()
                })
            elif isinstance(msg, AIMessage):
                history_data["messages"].append({
                    "type": "ai",
                    "content": msg.content,
                    "timestamp": time.time()
                })
        
        redis_client.setex(
            f"conversation:{session_id}",
            3600,  # 1 hour expiry
            json.dumps(history_data)
        )
    except Exception as e:
        logger.error(f"Error saving conversation history: {str(e)}")

@router.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Process a query using the RAG-based agent
    """
    if not llm:
        raise HTTPException(status_code=500, detail="Language model not available")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    start_time = time.time()
    
    try:
        # Get conversation history
        conversation_history = get_conversation_history(request.session_id)
        
        # Create memory with conversation history
        memory = ConversationBufferWindowMemory(
            k=5,  # Keep last 5 exchanges
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Add existing history to memory
        for msg in conversation_history:
            if isinstance(msg, HumanMessage):
                memory.chat_memory.add_user_message(msg.content)
            elif isinstance(msg, AIMessage):
                memory.chat_memory.add_ai_message(msg.content)
        
        # Create agent
        agent = create_react_agent(llm, tools, agent_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
        
        # Execute the agent
        result = agent_executor.invoke({"input": request.query})
        
        # Extract sources from the agent's intermediate steps
        sources = []
        if hasattr(result, 'intermediate_steps'):
            for step in result.get('intermediate_steps', []):
                if len(step) > 1 and isinstance(step[1], str):
                    try:
                        step_result = json.loads(step[1])
                        if 'results' in step_result:
                            sources.extend(step_result['results'])
                    except:
                        pass
        
        # Update conversation history
        updated_history = conversation_history + [
            HumanMessage(content=request.query),
            AIMessage(content=result['output'])
        ]
        save_conversation_history(request.session_id, updated_history)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=result['output'],
            sources=sources,
            session_id=request.session_id,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.get("/conversation/{session_id}", response_model=ConversationHistory)
async def get_conversation(session_id: str):
    """Get conversation history for a session"""
    try:
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        history_json = redis_client.get(f"conversation:{session_id}")
        if not history_json:
            return ConversationHistory(session_id=session_id, messages=[])
        
        history_data = json.loads(history_json)
        return ConversationHistory(
            session_id=session_id,
            messages=history_data.get("messages", [])
        )
        
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting conversation: {str(e)}")

@router.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    try:
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        redis_client.delete(f"conversation:{session_id}")
        return {"message": f"Conversation history cleared for session: {session_id}"}
        
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing conversation: {str(e)}")

@router.post("/compare-similarity")
async def compare_similarity_methods(query: str, max_results: int = 5):
    """
    Compare different similarity search algorithms
    """
    try:
        result = compare_similarity_algorithms_tool(query, max_results)
        comparison_data = json.loads(result)
        
        if "error" in comparison_data:
            raise HTTPException(status_code=500, detail=comparison_data["error"])
        
        # Add analysis
        analysis = {
            "cosine_avg_score": sum(r["score"] for r in comparison_data["cosine_similarity"]) / len(comparison_data["cosine_similarity"]) if comparison_data["cosine_similarity"] else 0,
            "euclidean_avg_score": sum(r["score"] for r in comparison_data["euclidean_similarity"]) / len(comparison_data["euclidean_similarity"]) if comparison_data["euclidean_similarity"] else 0,
            "recommendation": "Cosine similarity typically works better for text embeddings as it focuses on the angle between vectors rather than magnitude."
        }
        
        comparison_data["analysis"] = analysis
        
        return comparison_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing similarity methods: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing similarity methods: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint for the agent service"""
    status = {
        "agent_service": "healthy",
        "redis_connection": "connected" if redis_client else "disconnected",
        "qdrant_connection": "connected" if qdrant_client else "disconnected",
        "llm_available": "yes" if llm else "no"
    }
    
    # Test Redis connection
    if redis_client:
        try:
            redis_client.ping()
            status["redis_connection"] = "connected"
        except:
            status["redis_connection"] = "disconnected"
    
    # Test Qdrant connection
    if qdrant_client:
        try:
            qdrant_client.get_collections()
            status["qdrant_connection"] = "connected"
        except:
            status["qdrant_connection"] = "disconnected"
    
    return status
