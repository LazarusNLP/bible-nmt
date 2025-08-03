#!/usr/bin/env python3
"""
Simple token length analysis - works with any dataset format
"""

import sys
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import json

def analyze_hub_dataset(repo_name, config_name="ind_tet", sample_size=1000):
    """Analyze a dataset from Hugging Face Hub"""
    print(f"🔧 Loading NLLB tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
    
    print(f"📊 Loading dataset: {repo_name}, config: {config_name}")
    try:
        dataset = load_dataset(repo_name, config_name)
        train_data = dataset['train']
        
        if len(train_data) > sample_size:
            train_data = train_data.shuffle(seed=42).select(range(sample_size))
        
        print(f"✅ Loaded {len(train_data)} examples")
        
        # Tokenize
        src_texts = train_data['text_source']
        tgt_texts = train_data['text_target']
        
        print("🔍 Tokenizing texts...")
        src_tokens = tokenizer(src_texts, truncation=False, add_special_tokens=True)
        tgt_tokens = tokenizer(tgt_texts, truncation=False, add_special_tokens=True)
        
        src_lengths = [len(tokens) for tokens in src_tokens['input_ids']]
        tgt_lengths = [len(tokens) for tokens in tgt_tokens['input_ids']]
        
        # Results
        result = analyze_lengths(src_lengths, tgt_lengths, config_name)
        return result
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def analyze_local_texts(src_texts, tgt_texts, name="local_data"):
    """Analyze token lengths from raw text lists"""
    print(f"🔧 Loading NLLB tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
    
    print(f"📊 Analyzing {len(src_texts)} text pairs...")
    
    # Tokenize
    print("🔍 Tokenizing texts...")
    src_tokens = tokenizer(src_texts, truncation=False, add_special_tokens=True)
    tgt_tokens = tokenizer(tgt_texts, truncation=False, add_special_tokens=True)
    
    src_lengths = [len(tokens) for tokens in src_tokens['input_ids']]
    tgt_lengths = [len(tokens) for tokens in tgt_tokens['input_ids']]
    
    return analyze_lengths(src_lengths, tgt_lengths, name)

def analyze_lengths(src_lengths, tgt_lengths, name):
    """Core analysis function"""
    
    print(f"\n{'='*60}")
    print(f"📊 TOKEN LENGTH ANALYSIS: {name}")
    print(f"{'='*60}")
    
    # Basic stats
    src_max = max(src_lengths)
    src_mean = np.mean(src_lengths)
    src_p95 = np.percentile(src_lengths, 95)
    src_p99 = np.percentile(src_lengths, 99)
    
    tgt_max = max(tgt_lengths)
    tgt_mean = np.mean(tgt_lengths)
    tgt_p95 = np.percentile(tgt_lengths, 95)
    tgt_p99 = np.percentile(tgt_lengths, 99)
    
    overall_max = max(src_max, tgt_max)
    
    print(f"📝 Examples analyzed: {len(src_lengths):,}")
    print(f"\n🎯 MAXIMUM TOKEN LENGTHS:")
    print(f"   Source: {src_max:>3} tokens")
    print(f"   Target: {tgt_max:>3} tokens")
    print(f"   Overall: {overall_max:>3} tokens")
    
    print(f"\n📊 AVERAGES:")
    print(f"   Source: {src_mean:>5.1f} tokens")
    print(f"   Target: {tgt_mean:>5.1f} tokens")
    
    print(f"\n📈 PERCENTILES:")
    print(f"   95th - Source: {src_p95:>5.0f}, Target: {tgt_p95:>5.0f}")
    print(f"   99th - Source: {src_p99:>5.0f}, Target: {tgt_p99:>5.0f}")
    
    # Recommendations
    recommended = int(max(src_p99, tgt_p99) * 1.1)
    conservative = int(overall_max * 1.05)
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   Recommended max_length: {recommended} (99th percentile + 10%)")
    print(f"   Conservative max_length: {conservative} (absolute max + 5%)")
    print(f"   Use {recommended} for most training scenarios")
    
    # Show distribution
    print(f"\n📊 LENGTH DISTRIBUTION:")
    bins = [0, 50, 100, 150, 200, 250, 300, 400, 500, float('inf')]
    bin_labels = ['0-50', '51-100', '101-150', '151-200', '201-250', '251-300', '301-400', '401-500', '500+']
    
    all_lengths = src_lengths + tgt_lengths
    hist, _ = np.histogram(all_lengths, bins=bins)
    
    for i, (label, count) in enumerate(zip(bin_labels, hist)):
        pct = count / len(all_lengths) * 100
        print(f"   {label:>8} tokens: {count:>5} examples ({pct:>4.1f}%)")
    
    return {
        'name': name,
        'examples': len(src_lengths),
        'src_max': src_max,
        'src_mean': src_mean,
        'tgt_max': tgt_max,
        'tgt_mean': tgt_mean,
        'overall_max': overall_max,
        'recommended_max': recommended,
        'conservative_max': conservative
    }

def main():
    """Main function with usage examples"""
    
    print("🚀 Simple Token Length Analysis")
    print("Choose your analysis method:\n")
    
    if len(sys.argv) > 1:
        method = sys.argv[1]
    else:
        print("Usage examples:")
        print("  python scripts/simple_token_analysis.py hub <repo_name> [config_name]")
        print("  python scripts/simple_token_analysis.py sample")
        print("\nExamples:")
        print("  python scripts/simple_token_analysis.py hub biblenlp-corpus ind_tet")
        print("  python scripts/simple_token_analysis.py sample")
        return
    
    if method == "hub":
        if len(sys.argv) < 3:
            print("❌ Please provide repository name")
            print("Usage: python scripts/simple_token_analysis.py hub <repo_name> [config_name]")
            return
        
        repo_name = sys.argv[2]
        config_name = sys.argv[3] if len(sys.argv) > 3 else "ind_tet"
        
        result = analyze_hub_dataset(repo_name, config_name)
        if result:
            print(f"\n✅ Analysis complete for {repo_name}:{config_name}")
    
    elif method == "sample":
        # Create sample data for testing
        print("📝 Creating sample data for testing...")
        
        sample_src = [
            "Ini adalah contoh teks bahasa Indonesia yang pendek.",
            "Teks ini sedikit lebih panjang dari yang sebelumnya dan memiliki beberapa kata tambahan.",
            "Ini adalah contoh teks yang sangat panjang dengan banyak kata dan frasa yang bertujuan untuk menguji kemampuan tokenizer dalam menangani teks yang lebih kompleks dan memiliki struktur kalimat yang lebih rumit.",
        ]
        
        sample_tgt = [
            "This is a short Indonesian text example.",
            "This text is slightly longer than the previous one and has several additional words.",
            "This is an example of very long text with many words and phrases that aims to test the tokenizer's ability to handle more complex texts and have more complicated sentence structures.",
        ]
        
        # Replicate to create more samples
        sample_src = sample_src * 100
        sample_tgt = sample_tgt * 100
        
        result = analyze_local_texts(sample_src, sample_tgt, "sample_data")
        print(f"\n✅ Sample analysis complete")
    
    else:
        print(f"❌ Unknown method: {method}")
        print("Available methods: hub, sample")

if __name__ == "__main__":
    main()