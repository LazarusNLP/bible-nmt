#!/usr/bin/env python3
"""
Test script to demonstrate async batch processing with Gemini API.

Usage:
    export GEMINI_API_KEY="your-api-key"
    python test_async_gemini.py
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from models.api_models import GeminiModel


def test_gemini_async_batch():
    """Test the async batch processing functionality."""
    print("Testing Gemini async batch processing...")
    
    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        return
    
    # Create model
    model = GeminiModel("gemini-2.5-flash")
    
    # Test data - simulate translation post-editing tasks
    test_messages = []
    
    # Create sample messages for post-editing
    sample_tasks = [
        {
            "src": "Hello, how are you today?",
            "pred": "Halo, bagaimana kabar Anda hari ini?",
            "tgt_lang": "Indonesian"
        },
        {
            "src": "The weather is very nice.",
            "pred": "Cuaca sangat bagus.",
            "tgt_lang": "Indonesian"
        },
        {
            "src": "I love programming in Python.",
            "pred": "Saya suka pemrograman di Python.",
            "tgt_lang": "Indonesian"
        },
        {
            "src": "Machine learning is fascinating.",
            "pred": "Pembelajaran mesin adalah menarik.",
            "tgt_lang": "Indonesian"
        },
        {
            "src": "The book is on the table.",
            "pred": "Buku di atas meja.",
            "tgt_lang": "Indonesian"
        }
    ]
    
    # Convert to message format
    for task in sample_tasks:
        messages = [
            {
                "role": "system",
                "content": "You are an expert post-editor specializing in improving machine translation output. Your task is to review and correct machine-translated text to produce high-quality, fluent, and accurate translations in the target language only. You must respond ONLY with a JSON object containing the 'post_edited_text' field with the corrected translation in the target language."
            },
            {
                "role": "user", 
                "content": f"Source text (English): {task['src']}\n\nMachine translation ({task['tgt_lang']}): {task['pred']}\n\nProvide the corrected and improved translation in {task['tgt_lang']} as JSON: {{\"post_edited_text\": \"your corrected translation in {task['tgt_lang']} here\"}}"
            }
        ]
        test_messages.append(messages)
    
    print(f"Testing with {len(test_messages)} sample requests...")
    
    # Test single request first
    print("\n1. Testing single request...")
    single_result = model.generate(test_messages[0])
    print(f"Single result: {single_result}")
    
    # Test batch processing
    print(f"\n2. Testing async batch processing...")
    batch_results = model.generate_batch(test_messages)
    
    print(f"\nResults:")
    for i, (task, result) in enumerate(zip(sample_tasks, batch_results)):
        print(f"Task {i+1}:")
        print(f"  Source: {task['src']}")
        print(f"  Original MT: {task['pred']}")
        print(f"  Post-edited: {result}")
        print()
    
    print("Test completed!")


if __name__ == "__main__":
    test_gemini_async_batch()
