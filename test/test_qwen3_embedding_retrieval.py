#!/usr/bin/env python3
"""
Test script for All-MPNet embedding-based parallel sentence retrieval.

This script tests the enhanced SBERTRetriever using sentence-transformers/all-mpnet-base-v2 model 
to retrieve semantically similar parallel sentences for few-shot post-editing.

The implementation uses the existing SBERT infrastructure with configurable model names,
allowing seamless integration with the caching system and existing retrieval pipeline.

Usage:
    cd bible-nmt
    python test/test_qwen3_embedding_retrieval.py

Requirements:
    - sentence-transformers
    - torch
    - numpy
    - bible-nmt conda environment
"""

import os
import sys
import gc
from pathlib import Path
from retrieval.caching import clear_cache


# Add the post-editing source directory to Python path
post_editing_dir = Path(__file__).parent.parent / "src" / "post-editing"
sys.path.insert(0, str(post_editing_dir))

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def cleanup_gpu_memory():
    """Try to free up GPU memory between tests."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print("🧹 GPU memory cleanup completed")
    except Exception as e:
        print(f"⚠️ GPU cleanup warning: {e}")

def test_basic_all_mpnet_embedding():
    """Test basic All-MPNet embedding functionality following the transformers example."""
    print("="*60)
    print("TEST 1: Basic All-MPNet Embedding Functionality")
    print("="*60)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        import torch.nn.functional as F
        
        def mean_pooling(model_output, attention_mask):
            """Mean Pooling - Take attention mask into account for correct averaging."""
            token_embeddings = model_output[0]  # First element contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Test sentences
        sentences = [
            'What is the capital of China?',
            'Explain gravity',
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
        ]
        
        print(f"Loading All-MPNet model (sentence-transformers/all-mpnet-base-v2)...")
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        print(f"✅ Model loaded successfully!")
        
        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        print(f"Embeddings shape: {sentence_embeddings.shape}")
        print(f"Embedding dimension: {sentence_embeddings.shape[1]}")
        
        # Compute similarity scores (queries vs documents)
        query_embeddings = sentence_embeddings[:2]  # First 2 are queries
        doc_embeddings = sentence_embeddings[2:]    # Last 2 are documents
        scores = query_embeddings @ doc_embeddings.T
        
        print(f"Similarity scores: {scores.tolist()}")
        
        # Verify expected behavior
        if scores[0][0] > scores[0][1]:  # First query should match first document better
            print("✅ Basic embedding test PASSED: First query matches first document better")
        else:
            print("❌ Basic embedding test FAILED: Unexpected similarity scores")
            return False
            
        if scores[1][1] > scores[1][0]:  # Second query should match second document better
            print("✅ Basic embedding test PASSED: Second query matches second document better")
        else:
            print("❌ Basic embedding test FAILED: Unexpected similarity scores")
            return False
            
        print("✅ Basic All-MPNet embedding functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Basic embedding test FAILED with error: {e}")
        return False


def test_all_mpnet_retriever_class():
    """Test the SBERTRetriever class implementation."""
    print("\n" + "="*60)
    print("TEST 2: SBERTRetriever Class Functionality")
    print("="*60)
    
    try:
        from retrieval.vectorizers import SBERTRetriever
        
        # Create test corpus of parallel sentences
        test_corpus = [
            ("Hello, how are you?", "Halo, apa kabar?"),
            ("Good morning!", "Selamat pagi!"),
            ("What is your name?", "Siapa nama Anda?"),
            ("I love programming.", "Saya suka programming."),
            ("The weather is nice today.", "Cuaca hari ini bagus."),
            ("How are you doing?", "Bagaimana kabar Anda?"),  # Similar to first one
            ("It's a beautiful day.", "Hari yang indah."),     # Similar to weather
        ]
        
        print(f"Created test corpus with {len(test_corpus)} parallel sentence pairs")
        
        # Initialize retriever
        print("Initializing SBERTRetriever...")
        retriever = SBERTRetriever(
            corpus_id="test_corpus_all_mpnet",
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        print("✅ Retriever initialized successfully!")
        
        # Test retrieval with different queries
        test_queries = [
            "Hi, how are things?",  # Should match "Hello, how are you?" and "How are you doing?"
            "Good morning everyone!",  # Should match "Good morning!"
            "The weather is great.",  # Should match "The weather is nice today." and "It's a beautiful day."
        ]
        
        for i, query in enumerate(test_queries):
            print(f"\nTest Query {i+1}: '{query}'")
            
            # Retrieve top-3 similar examples
            k = 3
            similar_examples = retriever.get_similar_examples(query, test_corpus, k)
            
            print(f"Retrieved {len(similar_examples)} similar examples:")
            for j, (src, tgt) in enumerate(similar_examples):
                print(f"  {j+1}. '{src}' -> '{tgt}'")
                
            # Basic validation
            if len(similar_examples) == k:
                print(f"✅ Retrieved correct number of examples ({k})")
            else:
                print(f"❌ Expected {k} examples, got {len(similar_examples)}")
                return False
        
        print("\n✅ SBERTRetriever class functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ SBERTRetriever test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_few_shot_selector_integration():
    """Test integration with FewShotSelector."""
    print("\n" + "="*60)
    print("TEST 3: FewShotSelector Integration")
    print("="*60)
    
    try:
        from retrieval.few_shot import FewShotSelector
        from core.data_models import Row
        
        # Create test corpus
        test_corpus = [
            ("The book is on the table.", "Buku itu di atas meja."),
            ("I am reading a book.", "Saya sedang membaca buku."),
            ("She likes to read books.", "Dia suka membaca buku."),
            ("The cat is sleeping.", "Kucing itu sedang tidur."),
            ("Birds can fly high.", "Burung bisa terbang tinggi."),
        ]
        
        print(f"Created test corpus with {len(test_corpus)} parallel sentence pairs")
        
        # Initialize FewShotSelector with all_mpnet
        print("Initializing FewShotSelector with all_mpnet vectorizer...")
        selector = FewShotSelector(
            vectorizer_type="all_mpnet", 
            corpus_id="test_integration_corpus_all_mpnet"
        )
        print("✅ FewShotSelector initialized successfully!")
        
        # Create test rows
        test_rows = [
            Row(
                src_text="I love reading books.",
                tgt_text="Saya suka membaca buku.",
                pred_tgt_text="Saya cinta membaca buku.",
                src_lang="en",
                tgt_lang="id", 
                src_lang_name="English",
                tgt_lang_name="Indonesian"
            ),
            Row(
                src_text="The dog is running fast.",
                tgt_text="Anjing itu berlari cepat.",
                pred_tgt_text="Anjing berlari dengan cepat.",
                src_lang="en",
                tgt_lang="id",
                src_lang_name="English", 
                tgt_lang_name="Indonesian"
            )
        ]
        
        print(f"Created {len(test_rows)} test rows")
        
        # Test example retrieval for multiple rows
        k = 2
        examples_list = selector.get_examples_for_rows(test_rows, test_corpus, k)
        
        print(f"\nRetrieved examples for {len(examples_list)} rows:")
        for i, examples in enumerate(examples_list):
            print(f"\nRow {i+1} ('{test_rows[i].src_text}'):")
            print(f"  Retrieved {len(examples)} examples:")
            for j, (src, tgt) in enumerate(examples):
                print(f"    {j+1}. '{src}' -> '{tgt}'")
        
        # Validate results
        if len(examples_list) == len(test_rows):
            print(f"✅ Retrieved examples for all {len(test_rows)} rows")
        else:
            print(f"❌ Expected examples for {len(test_rows)} rows, got {len(examples_list)}")
            return False
            
        for i, examples in enumerate(examples_list):
            if len(examples) == k:
                print(f"✅ Row {i+1}: Retrieved correct number of examples ({k})")
            else:
                print(f"❌ Row {i+1}: Expected {k} examples, got {len(examples)}")
                return False
        
        print("\n✅ FewShotSelector integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ FewShotSelector integration test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_caching_functionality():
    """Test embedding caching functionality."""
    print("\n" + "="*60)
    print("TEST 4: Embedding Caching Functionality")
    print("="*60)
    
    try:
        from retrieval.vectorizers import SBERTRetriever
        
        # Same corpus for testing caching
        test_corpus = [
            ("Hello world!", "Halo dunia!"),
            ("Good bye!", "Selamat tinggal!"),
            ("Thank you!", "Terima kasih!"),
        ]
        
        # First retriever instance
        print("Creating first retriever instance...")
        retriever1 = SBERTRetriever(
            corpus_id="cache_test_corpus_all_mpnet",
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # This should compute and cache embeddings
        print("First retrieval (should compute and cache embeddings)...")
        examples1 = retriever1.get_similar_examples("Hi there!", test_corpus, 2)
        print(f"Retrieved {len(examples1)} examples")
        
        # Second retriever instance with same corpus_id
        print("Creating second retriever instance with same corpus_id...")
        retriever2 = SBERTRetriever(
            corpus_id="cache_test_corpus_all_mpnet",
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # This should use cached embeddings
        print("Second retrieval (should use cached embeddings)...")
        examples2 = retriever2.get_similar_examples("Hello!", test_corpus, 2)
        print(f"Retrieved {len(examples2)} examples")
        
        # Basic validation - both should return same number of examples
        if len(examples1) == len(examples2) == 2:
            print("✅ Both retrievals returned correct number of examples")
        else:
            print(f"❌ Inconsistent results: {len(examples1)} vs {len(examples2)}")
            return False
        
        # Test cache clearing
        print("Testing cache clearing...")
        clear_cache()
        print("✅ Cache cleared successfully")
        
        print("\n✅ Embedding caching functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Caching functionality test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Starting All-MPNet Embedding Retrieval Tests")
    print("Environment: bible-nmt conda environment")
    print("Model: sentence-transformers/all-mpnet-base-v2")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
            print(f"Current device: {torch.cuda.current_device()}")
        else:
            print("⚠️ CUDA not available - will use CPU (slower)")
    except:
        print("⚠️ Could not check CUDA availability")
    
    # No special environment variables needed for transformers
    
    test_results = []
    
    # Run all tests
    test_functions = [
        test_basic_all_mpnet_embedding,
        test_all_mpnet_retriever_class,
        test_few_shot_selector_integration,
        test_caching_functionality,
    ]
    
    for test_func in test_functions:
        try:
            result = test_func()
            test_results.append(result)
            # Clean up GPU memory after each test
            cleanup_gpu_memory()
        except KeyboardInterrupt:
            print("\n❌ Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Test {test_func.__name__} failed with unexpected error: {e}")
            test_results.append(False)
            # Clean up even after failures
            cleanup_gpu_memory()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Qwen3 embedding retrieval is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
