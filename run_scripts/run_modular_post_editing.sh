#!/bin/bash

# Unified Post-Editing Script using the new modular system
# This script replaces both run_vllm_post_editing.sh and run_api_post_editing.sh

# Set working directory to project root
cd /data/projects/punim0478/setiawand/bible-nmt

# ====================================================================
# vLLM Model Experiments (equivalent to old run_vllm_post_editing.sh)
# ====================================================================

echo "Starting vLLM post-editing experiments..."

# Experiment 1: TF-IDF vectorizer
python src/post-editing/run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-27b-it-fp8" \
  --csv_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao/aligned-ind-dhao-ot.csv" \
  --few_shot_corpus_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao/aligned-ind-dhao-nt.csv" \
  --src "id" \
  --tgt "nfa" \
  --src_lang_name "Indonesian" \
  --tgt_lang_name "Dhao" \
  --output_dir "results/modular/gemma-3-27b-it-5-shot-tfidf" \
  --prompt "post_editing" \
  --num_few_shot 5 \
  --vectorizer "tfidf" \
  # --max_samples 10 \
  # --debug

# Experiment 2: BM25 vectorizer  
python src/post-editing/run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-27b-it-fp8" \
  --csv_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao/aligned-ind-dhao-ot.csv" \
  --few_shot_corpus_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao/aligned-ind-dhao-nt.csv" \
  --src "id" \
  --tgt "nfa" \
  --src_lang_name "Indonesian" \
  --tgt_lang_name "Dhao" \
  --output_dir "results/modular/gemma-3-27b-it-5-shot-bm25" \
  --prompt "post_editing" \
  --num_few_shot 5 \
  --vectorizer "bm25" \
  # --max_samples 10 \
  # --debug

# Experiment 3: SBERT vectorizer (with caching)
python src/post-editing/run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-27b-it-fp8" \
  --csv_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao/aligned-ind-dhao-ot.csv" \
  --few_shot_corpus_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao/aligned-ind-dhao-nt.csv" \
  --src "id" \
  --tgt "nfa" \
  --src_lang_name "Indonesian" \
  --tgt_lang_name "Dhao" \
  --output_dir "results/modular/gemma-3-27b-it-5-shot-sbert" \
  --prompt "post_editing" \
  --num_few_shot 5 \
  --vectorizer "sbert" \
  # --max_samples 10 \
  # --debug

# ====================================================================
# API Model Experiments (equivalent to old run_api_post_editing.sh)
# ====================================================================

echo "Starting API post-editing experiments..."

# Experiment 4: Gemini API baseline (no few-shot)
python src/post-editing/run_post_editing.py \
  --model_type "gemini" \
  --model_name "gemini-2.5-flash" \
  --csv_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao/aligned-ind-dhao-ot.csv" \
  --src "id" \
  --tgt "nfa" \
  --src_lang_name "Indonesian" \
  --tgt_lang_name "Dhao" \
  --output_dir "results/modular/gemini-baseline" \
  --prompt "post_editing" \
  --num_workers 10 \
  # --max_samples 10 \
  # --debug

# Experiment 5: Gemini API with few-shot examples
python src/post-editing/run_post_editing.py \
  --model_type "gemini" \
  --model_name "gemini-2.5-flash" \
  --csv_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao/aligned-ind-dhao-ot.csv" \
  --few_shot_corpus_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao/aligned-ind-dhao-nt.csv" \
  --src "id" \
  --tgt "nfa" \
  --src_lang_name "Indonesian" \
  --tgt_lang_name "Dhao" \
  --output_dir "results/modular/gemini-5-shot-bm25" \
  --prompt "post_editing" \
  --num_few_shot 5 \
  --vectorizer "bm25" \
  --num_workers 10 \
  --batch_size 50 \
  --batch_timeout 3600 \
  # --max_samples 10 \
  # --debug

# Experiment 6: GPT API with few-shot examples
python src/post-editing/run_post_editing.py \
  --model_type "gpt" \
  --model_name "gpt-4o" \
  --csv_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao/aligned-ind-dhao-ot.csv" \
  --few_shot_corpus_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao/aligned-ind-dhao-nt.csv" \
  --src "id" \
  --tgt "nfa" \
  --src_lang_name "Indonesian" \
  --tgt_lang_name "Dhao" \
  --output_dir "results/modular/gpt-5-shot-sbert" \
  --prompt "post_editing" \
  --num_few_shot 5 \
  --vectorizer "sbert" \
  --num_workers 8 \
  # --max_samples 10 \
  # --debug

echo "All post-editing experiments completed!"
