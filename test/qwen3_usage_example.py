#!/usr/bin/env python3
"""
Simple usage example for the All-MPNet embedding vectorizer in post-editing.

This demonstrates how to use the new all_mpnet vectorizer with the post-editing pipeline.

Usage:
    cd bible-nmt
    python test/qwen3_usage_example.py
"""

import os
import sys
from pathlib import Path

# Add the post-editing source directory to Python path
post_editing_dir = Path(__file__).parent.parent / "src" / "post-editing"
sys.path.insert(0, str(post_editing_dir))

def demonstrate_all_mpnet_vectorizer():
    """Demonstrate basic usage of the All-MPNet vectorizer."""
    print("🚀 All-MPNet Embedding Vectorizer Usage Example")
    print("="*50)
    
    try:
        from retrieval.few_shot import FewShotSelector
        
        # Sample parallel corpus (English-Indonesian)
        corpus = [
            ("Good morning!", "Selamat pagi!"),
            ("How are you?", "Apa kabar?"),
            ("Thank you very much.", "Terima kasih banyak."),
            ("I love programming.", "Saya suka programming."),
            ("The weather is nice today.", "Cuaca hari ini bagus."),
            ("What is your name?", "Siapa nama Anda?"),
            ("Have a great day!", "Semoga harimu menyenangkan!"),
            ("I am learning Indonesian.", "Saya sedang belajar bahasa Indonesia."),
        ]
        
        print(f"Sample corpus: {len(corpus)} parallel sentence pairs")
        
        # Initialize few-shot selector with All-MPNet embeddings
        print("\nInitializing FewShotSelector with all_mpnet vectorizer...")
        # Create a few-shot selector with the all-mpnet vectorizer
        # This uses SBERTRetriever with sentence-transformers/all-mpnet-base-v2 model
        selector = FewShotSelector(
            vectorizer_type="all_mpnet",
            corpus_id="demo_corpus_all_mpnet"
        )
        print("✅ Initialized successfully!")
        
        # Test queries
        test_queries = [
            "Hello! How are things?",
            "Thanks a lot!",
            "What's the weather like?",
            "My name is John.",
        ]
        
        print(f"\nTesting retrieval with {len(test_queries)} queries:")
        print("-" * 50)
        
        for i, query in enumerate(test_queries):
            print(f"\nQuery {i+1}: '{query}'")
            
            # Retrieve top-3 similar examples
            similar_examples = selector.get_examples(query, corpus, k=3)
            
            print("Top 3 similar examples:")
            for j, (src, tgt) in enumerate(similar_examples):
                print(f"  {j+1}. '{src}' → '{tgt}'")
        
        print("\n" + "="*50)
        print("✅ Demo completed successfully!")
        print("\nTo use in post-editing pipeline:")
        print("python run_post_editing.py --vectorizer qwen3_vllm --model_type vllm \\")
        print("  --model_name your_model --csv_path data.csv --few_shot_corpus_path corpus.csv \\")
        print("  --src en --tgt id --src_lang_name English --tgt_lang_name Indonesian \\")
        print("  --output_dir ./results")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set environment variables for vLLM
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['DISABLE_XFORMERS'] = '1'
    os.environ['FLASH_ATTENTION_SKIP_CUDA_BUILD'] = '1'
    os.environ['ENFORCE_EAGER'] = '1'
    
    demonstrate_qwen3_vectorizer()
