#!/usr/bin/env python3
"""
Demo script to showcase the RAG-based backend system functionality
"""

import sys
import os
sys.path.append('.')

def demo_chunking_strategies():
    """Demonstrate different text chunking strategies"""
    print("🔍 DEMO: Text Chunking Strategies")
    print("=" * 50)
    
    # Sample text
    sample_text = """
    Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.

    Key concepts include supervised learning, which uses labeled data to train models, unsupervised learning that finds patterns in unlabeled data, and reinforcement learning that learns through interaction and feedback.

    Applications of machine learning are widespread, including natural language processing, computer vision, recommendation systems, and predictive analytics.

    Deep learning, a subset of machine learning, uses neural networks with multiple layers to model complex patterns in data. It has revolutionized fields like image recognition and natural language understanding.
    """
    
    try:
        from utils.chunking import recursive_chunk, semantic_chunk, custom_chunk
        
        print("📝 Original text length:", len(sample_text), "characters")
        print("\n" + "─" * 30)
        
        # Test recursive chunking
        print("\n🔄 RECURSIVE CHUNKING:")
        recursive_result = recursive_chunk(sample_text, chunk_size=200, overlap=20)
        print(f"   Chunks created: {recursive_result['num_chunks']}")
        print(f"   Avg chunk size: {recursive_result['avg_chunk_size']:.1f}")
        print(f"   Processing time: {recursive_result['processing_time']:.3f}s")
        print("   First chunk preview:", recursive_result['chunks'][0][:100] + "...")
        
        # Test semantic chunking
        print("\n🧠 SEMANTIC CHUNKING:")
        semantic_result = semantic_chunk(sample_text, max_chunk_size=300)
        print(f"   Chunks created: {semantic_result['num_chunks']}")
        print(f"   Avg chunk size: {semantic_result['avg_chunk_size']:.1f}")
        print(f"   Processing time: {semantic_result['processing_time']:.3f}s")
        print("   First chunk preview:", semantic_result['chunks'][0][:100] + "...")
        
        # Test custom chunking
        print("\n⚙️ CUSTOM CHUNKING:")
        custom_result = custom_chunk(sample_text, min_chunk_size=50, max_chunk_size=250)
        print(f"   Chunks created: {custom_result['num_chunks']}")
        print(f"   Avg chunk size: {custom_result['avg_chunk_size']:.1f}")
        print(f"   Processing time: {custom_result['processing_time']:.3f}s")
        print("   First chunk preview:", custom_result['chunks'][0][:100] + "...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in chunking demo: {str(e)}")
        return False

def demo_embedding_generation():
    """Demonstrate embedding generation (mock version)"""
    print("\n\n🔢 DEMO: Embedding Generation")
    print("=" * 50)
    
    try:
        # Mock embedding generation since we don't have the models installed
        sample_chunks = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Applications include computer vision and NLP."
        ]
        
        print("📊 Generating embeddings for sample chunks...")
        
        for i, chunk in enumerate(sample_chunks):
            # Mock embedding (in real system, this would use Sentence Transformers)
            mock_embedding = [0.1 * (ord(c) % 10) for c in chunk[:10]]  # Simple mock
            print(f"   Chunk {i+1}: {len(chunk)} chars → {len(mock_embedding)}D vector")
            print(f"   Preview: [{mock_embedding[0]:.2f}, {mock_embedding[1]:.2f}, ...]")
        
        print("✅ Embedding generation completed (mock)")
        return True
        
    except Exception as e:
        print(f"❌ Error in embedding demo: {str(e)}")
        return False

def demo_database_operations():
    """Demonstrate database operations"""
    print("\n\n🗄️ DEMO: Database Operations")
    print("=" * 50)
    
    try:
        from db import init_db, get_db_session, FileMetadata, Booking
        from datetime import datetime
        
        # Initialize database
        print("🔧 Initializing database...")
        init_db()
        print("✅ Database initialized")
        
        # Create a session
        db = get_db_session()
        
        # Demo file metadata storage
        print("\n📁 Storing file metadata...")
        file_meta = FileMetadata(
            file_name="demo_document.txt",
            file_size=1024,
            chunking_method="semantic",
            embedding_model="all-MiniLM-L6-v2",
            num_chunks=5,
            processing_time=2.5
        )
        db.add(file_meta)
        db.commit()
        print(f"✅ File metadata stored with ID: {file_meta.id}")
        
        # Demo booking storage
        print("\n📅 Storing booking information...")
        booking = Booking(
            full_name="John Doe",
            email="john@example.com",
            interview_date="2024-02-15",
            interview_time="14:30",
            confirmation_sent="pending"
        )
        db.add(booking)
        db.commit()
        print(f"✅ Booking stored with ID: {booking.id}")
        
        # Query data
        print("\n🔍 Querying stored data...")
        files = db.query(FileMetadata).all()
        bookings = db.query(Booking).all()
        
        print(f"   Files in database: {len(files)}")
        print(f"   Bookings in database: {len(bookings)}")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Error in database demo: {str(e)}")
        return False

def demo_api_structure():
    """Show the API structure and endpoints"""
    print("\n\n🌐 DEMO: API Structure")
    print("=" * 50)
    
    try:
        from main import app
        
        print("🚀 FastAPI Application Structure:")
        print("\n📋 Available Routes:")
        
        routes = []
        for route in app.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                methods = ', '.join(route.methods)
                routes.append(f"   {methods:<10} {route.path}")
        
        # Sort routes for better display
        routes.sort()
        for route in routes:
            print(route)
        
        print(f"\n📊 Total endpoints: {len(routes)}")
        print("✅ API structure loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error loading API structure: {str(e)}")
        return False

def demo_email_functionality():
    """Demonstrate email functionality (mock)"""
    print("\n\n📧 DEMO: Email Functionality")
    print("=" * 50)
    
    try:
        print("📮 Mock email confirmation...")
        
        # Mock email data
        recipient = "demo@example.com"
        full_name = "Jane Smith"
        interview_date = "2024-02-20"
        interview_time = "10:00"
        
        print(f"   To: {recipient}")
        print(f"   Subject: Interview Booking Confirmation")
        print(f"   Content: Hello {full_name}, your interview is scheduled for {interview_date} at {interview_time}")
        print("✅ Email would be sent successfully (mock)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in email demo: {str(e)}")
        return False

def main():
    """Run all demos"""
    print("🎯 RAG-BASED BACKEND SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the key components of the system:")
    print("• Text chunking strategies")
    print("• Embedding generation")
    print("• Database operations")
    print("• API structure")
    print("• Email functionality")
    print("=" * 60)
    
    demos = [
        ("Text Chunking", demo_chunking_strategies),
        ("Embedding Generation", demo_embedding_generation),
        ("Database Operations", demo_database_operations),
        ("API Structure", demo_api_structure),
        ("Email Functionality", demo_email_functionality)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        try:
            result = demo_func()
            results.append((demo_name, result))
        except Exception as e:
            print(f"❌ {demo_name} failed: {str(e)}")
            results.append((demo_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DEMONSTRATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    for demo_name, result in results:
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"{status} {demo_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} demos completed successfully")
    
    if passed == len(results):
        print("\n🎉 All system components demonstrated successfully!")
        print("\n🚀 To run the full system:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Set up external services (Qdrant, Redis)")
        print("   3. Configure environment variables in .env")
        print("   4. Run: python main.py")
        print("   5. Visit: http://localhost:8000/docs")
    else:
        print("\n⚠️ Some components need external dependencies to run fully.")
    
    return 0 if passed == len(results) else 1

if __name__ == "__main__":
    exit(main())
