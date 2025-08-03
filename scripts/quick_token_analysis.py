#!/usr/bin/env python3
"""
Quick token length analysis script - faster version with just the essentials
"""

import sys
import os
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm

# Configuration
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
SOURCE_LANG = "ind"
TARGET_LANGS = ["aaz", "ptu", "nfa", "heg", "lex", "row", "llg", "rgu", "txq", "tet", "wrs"]

NT_BOOKS = [
    "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", "GAL", "EPH",
    "PHP", "COL", "1TH", "2TH", "1TI", "2TI", "TIT", "PHM", "HEB", "JAS",
    "1PE", "2PE", "1JN", "2JN", "3JN", "JUD", "REV"
]

def load_ebible_corpus(src_lang, tgt_lang):
    """Load and process a single language pair dataset"""
    def process_translations(x):
        languages = x["translation"]["language"]
        translations = x["translation"]["translation"]
        
        lang_translations = {}
        for lang, translation in zip(languages, translations):
            if lang not in lang_translations:
                lang_translations[lang] = []
            lang_translations[lang].append(translation)
        
        src_text = " ".join(lang_translations.get(src_lang, [""]))
        tgt_text = " ".join(lang_translations.get(tgt_lang, [""]))
        return {"text_source": src_text, "text_target": tgt_text}
    
    dataset = load_dataset("bible-nlp/biblenlp-corpus", languages=[src_lang, tgt_lang])
    dataset = dataset.map(process_translations)
    
    def is_nt_book(refs):
        if isinstance(refs, list):
            return any(ref.split()[0] in NT_BOOKS for ref in refs)
        else:
            return refs.split()[0] in NT_BOOKS
    
    train_data = dataset['train']
    train_ds = train_data.filter(lambda x: is_nt_book(x["ref"]))
    test_ds = train_data.filter(lambda x: not is_nt_book(x["ref"]))
    train_val_ds = train_ds.train_test_split(test_size=0.1, seed=41)
    
    return DatasetDict({
        "train": train_val_ds["train"], 
        "validation": train_val_ds["test"], 
        "test": test_ds
    })

def quick_analyze_lengths(sample_size=1000):
    """Quick analysis with sampling for speed"""
    
    print("🔧 Loading NLLB tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"✅ Loaded: {MODEL_NAME}")
    
    all_src_lengths = []
    all_tgt_lengths = []
    results = []
    
    print(f"\n🚀 Quick analysis (sampling {sample_size} examples per language pair)...")
    
    for i, tgt_lang in enumerate(TARGET_LANGS):
        print(f"[{i+1}/{len(TARGET_LANGS)}] {SOURCE_LANG} → {tgt_lang}...")
        
        try:
            dataset = load_ebible_corpus(SOURCE_LANG, tgt_lang)
            
            # Sample from training data for speed
            train_data = dataset['train']
            if len(train_data) > sample_size:
                train_data = train_data.shuffle(seed=42).select(range(sample_size))
            
            # Tokenize all at once
            src_texts = train_data['text_source']
            tgt_texts = train_data['text_target']
            
            src_tokens = tokenizer(src_texts, truncation=False, add_special_tokens=True)
            tgt_tokens = tokenizer(tgt_texts, truncation=False, add_special_tokens=True)
            
            src_lengths = [len(tokens) for tokens in src_tokens['input_ids']]
            tgt_lengths = [len(tokens) for tokens in tgt_tokens['input_ids']]
            
            all_src_lengths.extend(src_lengths)
            all_tgt_lengths.extend(tgt_lengths)
            
            # Calculate stats
            result = {
                'lang_pair': f"{SOURCE_LANG}_{tgt_lang}",
                'examples': len(src_lengths),
                'src_max': max(src_lengths),
                'src_mean': np.mean(src_lengths),
                'tgt_max': max(tgt_lengths),
                'tgt_mean': np.mean(tgt_lengths),
            }
            results.append(result)
            
            print(f"  Src: max={result['src_max']:>3}, avg={result['src_mean']:>5.1f}")
            print(f"  Tgt: max={result['tgt_max']:>3}, avg={result['tgt_mean']:>5.1f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Overall statistics
    print(f"\n{'='*60}")
    print(f"📊 OVERALL RESULTS")
    print(f"{'='*60}")
    
    if not results:
        print("❌ No successful results. Please check dataset availability or try different languages.")
        return 0, 0, []
    
    global_src_max = max(r['src_max'] for r in results)
    global_tgt_max = max(r['tgt_max'] for r in results)
    global_max = max(global_src_max, global_tgt_max)
    
    print(f"🎯 MAXIMUM TOKEN LENGTHS:")
    print(f"   Source (Indonesian): {global_src_max}")
    print(f"   Target (all langs):  {global_tgt_max}")
    print(f"   Overall maximum:     {global_max}")
    
    print(f"\n📈 PERCENTILES:")
    src_p95 = np.percentile(all_src_lengths, 95)
    src_p99 = np.percentile(all_src_lengths, 99)
    tgt_p95 = np.percentile(all_tgt_lengths, 95)
    tgt_p99 = np.percentile(all_tgt_lengths, 99)
    
    print(f"   95th percentile: Src={src_p95:.0f}, Tgt={tgt_p95:.0f}")
    print(f"   99th percentile: Src={src_p99:.0f}, Tgt={tgt_p99:.0f}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    recommended = int(max(src_p99, tgt_p99) * 1.1)
    conservative = int(global_max * 1.05)
    
    print(f"   Recommended max_length: {recommended} (99th percentile + 10%)")
    print(f"   Conservative max_length: {conservative} (absolute max + 5%)")
    
    # Show top languages by max length
    print(f"\n🌍 LONGEST BY LANGUAGE:")
    sorted_results = sorted(results, key=lambda x: x['tgt_max'], reverse=True)
    for i, result in enumerate(sorted_results[:5]):
        lang = result['lang_pair'].split('_')[1]
        print(f"   {i+1}. {lang}: {result['tgt_max']} tokens")
    
    return global_max, recommended, results

if __name__ == "__main__":
    # You can adjust sample size for speed vs accuracy
    sample_size = 1000 if len(sys.argv) < 2 else int(sys.argv[1])
    
    print(f"🚀 Quick Token Length Analysis")
    print(f"📊 Sample size: {sample_size} examples per language")
    
    max_tokens, recommended, results = quick_analyze_lengths(sample_size)
    
    print(f"\n✅ Analysis complete!")
    print(f"🎯 Use max_length={recommended} for training")