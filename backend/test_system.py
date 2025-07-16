#!/usr/bin/env python3
"""
Simple test script to verify the backend system functionality
"""

import requests
import json
import time
import os
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            health_data = response.json()
            print(f"   Status: {health_data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Root endpoint working")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {str(e)}")
        return False

def create_test_file():
    """Create a test text file"""
    test_content = """
    This is a test document for the RAG-based backend system.
    
    Machine Learning Overview:
    Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.
    
    Key Concepts:
    1. Supervised Learning: Uses labeled data to train models
    2. Unsupervised Learning: Finds patterns in unlabeled data
    3. Reinforcement Learning: Learns through interaction and feedback
    
    Applications:
    - Natural Language Processing
    - Computer Vision
    - Recommendation Systems
    - Predictive Analytics
    
    Deep Learning:
    Deep learning uses neural networks with multiple layers to model complex patterns in data. It has revolutionized fields like image recognition and natural language understanding.
    
    Conclusion:
    Machine learning continues to evolve and find new applications across various industries, from healthcare to finance to entertainment.
    """
    
    test_file_path = "test_document.txt"
    with open(test_file_path, "w") as f:
        f.write(test_content)
    
    return test_file_path

def test_file_upload():
    """Test file upload functionality"""
    print("🔍 Testing file upload...")
    
    # Create test file
    test_file_path = create_test_file()
    
    try:
        with open(test_file_path, "rb") as f:
            files = {"file": (test_file_path, f, "text/plain")}
            data = {
                "chunking_method": "semantic",
                "embedding_model": "all-MiniLM-L6-v2"
            }
            
            response = requests.post(f"{BASE_URL}/api/upload", files=files, data=data)
            
        if response.status_code == 200:
            print("✅ File upload successful")
            upload_data = response.json()
            print(f"   File ID: {upload_data.get('file_id')}")
            print(f"   Chunks: {upload_data.get('num_chunks')}")
            return upload_data.get('file_id')
        else:
            print(f"❌ File upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ File upload error: {str(e)}")
        return None
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

def test_query_agent():
    """Test the RAG agent query functionality"""
    print("🔍 Testing RAG agent query...")
    
    try:
        query_data = {
            "query": "What is machine learning and what are its key concepts?",
            "session_id": "test_session",
            "max_results": 3
        }
        
        response = requests.post(
            f"{BASE_URL}/api/query",
            headers={"Content-Type": "application/json"},
            data=json.dumps(query_data)
        )
        
        if response.status_code == 200:
            print("✅ Query successful")
            query_result = response.json()
            print(f"   Answer length: {len(query_result.get('answer', ''))}")
            print(f"   Sources found: {len(query_result.get('sources', []))}")
            print(f"   Processing time: {query_result.get('processing_time', 0):.2f}s")
            return True
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        return False

def test_booking_system():
    """Test the interview booking functionality"""
    print("🔍 Testing interview booking...")
    
    try:
        booking_data = {
            "full_name": "John Doe",
            "email": "test@example.com",
            "interview_date": "2024-12-31",
            "interview_time": "14:30"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/booking",
            headers={"Content-Type": "application/json"},
            data=json.dumps(booking_data)
        )
        
        if response.status_code == 200:
            print("✅ Booking successful")
            booking_result = response.json()
            print(f"   Booking ID: {booking_result.get('booking_id')}")
            print(f"   Confirmation: {booking_result.get('confirmation_status')}")
            return booking_result.get('booking_id')
        else:
            print(f"❌ Booking failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Booking error: {str(e)}")
        return None

def test_list_files():
    """Test listing uploaded files"""
    print("🔍 Testing file listing...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/files")
        
        if response.status_code == 200:
            print("✅ File listing successful")
            files_data = response.json()
            print(f"   Total files: {files_data.get('total_files', 0)}")
            return True
        else:
            print(f"❌ File listing failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ File listing error: {str(e)}")
        return False

def test_metrics():
    """Test metrics endpoint"""
    print("🔍 Testing metrics endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        
        if response.status_code == 200:
            print("✅ Metrics endpoint working")
            metrics_data = response.json()
            print(f"   Total files processed: {metrics_data.get('file_processing', {}).get('total_files', 0)}")
            return True
        else:
            print(f"❌ Metrics failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Metrics error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Backend System Tests")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root_endpoint),
        ("File Upload", test_file_upload),
        ("List Files", test_list_files),
        ("RAG Query", test_query_agent),
        ("Interview Booking", test_booking_system),
        ("Metrics", test_metrics)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    exit(main())
