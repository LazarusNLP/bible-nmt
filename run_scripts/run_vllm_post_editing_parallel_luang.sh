#!/bin/bash

# vLLM Post-Editing using the new modular system
# Direct replacement for run_vllm_post_editing.sh

# Set working directory to project root
cd /data/projects/punim0478/setiawand/bible-nmt

# Change to post-editing directory for imports to work correctly
cd src/post-editing

# Generate timestamp for unique output directories
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT="../../results/all-para/run_${TIMESTAMP}"

echo "Starting vLLM post-editing experiments with timestamp: ${TIMESTAMP}"
echo "Results will be saved to: ${BASE_OUTPUT}"

# Experiment 1: TF-IDF vectorizer (5-shot)
echo "Running Experiment 1: TF-IDF vectorizer..."
python run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
  --csv_path ./ebible-corpus/luang-eng/aligned-eng-luang-ot.csv\
  --few_shot_corpus_path ./ebible-corpus/luang-eng/aligned-eng-luang-nt.csv \
  --src "eng" \
  --tgt "lex" \
  --src_lang_name "English" \
  --tgt_lang_name "Luang" \
  --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-5-shot-tfidf" \
  --prompt "post_editing" \
  --num_few_shot 5 \
  --vectorizer "tfidf" \

# Experiment 2: BM25 vectorizer (5-shot)
# echo "Running Experiment 2: BM25 vectorizer..."
# python run_post_editing.py \
#   --model_type "vllm" \
#   --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
#   --csv_path /data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/luang/aligned-ind-luang-ot.csv \
#   --few_shot_corpus_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/luang/aligned-ind-luang-nt.csv" \
#   --src "id" \
#   --tgt "lex" \
#   --src_lang_name "English" \
#   --tgt_lang_name "Luang" \
#   --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-5-shot-bm25" \
#   --prompt "post_editing" \
#   --num_few_shot 5 \
#   --vectorizer "bm25" \

# # Experiment 3: SBERT vectorizer (5-shot) with caching
# echo "Running Experiment 3: SBERT vectorizer..."
# python run_post_editing.py \
#   --model_type "vllm" \
#   --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
#   --csv_path /data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/luang/aligned-ind-luang-ot.csv \
#   --few_shot_corpus_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/luang/aligned-ind-luang-nt.csv" \
#   --src "id" \
#   --tgt "lex" \
#   --src_lang_name "English" \
#   --tgt_lang_name "Luang" \
#   --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-5-shot-sbert" \
#   --prompt "post_editing" \
#   --num_few_shot 5 \
#   --vectorizer "sbert" \

# echo "Running Experiment 4: Counterweighted CHRF vectorizer..."
#   python run_post_editing.py \
#   --model_type "vllm" \
#   --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
#   --csv_path /data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/luang/aligned-ind-luang-ot.csv \
#   --few_shot_corpus_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/luang/aligned-ind-luang-nt.csv" \
#   --src "id" \
#   --tgt "lex" \
#   --src_lang_name "English" \
#   --tgt_lang_name "Luang" \
#   --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-5-shot-chrf_rag" \
#   --prompt "post_editing" \
#   --num_few_shot 5 \
#   --vectorizer "chrf_rag" \

# echo "Running Experiment 5: Word Longest Common Subsequence vectorizer..."
# python run_post_editing.py \
#   --model_type "vllm" \
#   --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
#   --csv_path /data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/luang/aligned-ind-luang-ot.csv \
#   --few_shot_corpus_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/luang/aligned-ind-luang-nt.csv" \
#   --src "id" \
#   --tgt "lex" \
#   --src_lang_name "English" \
#   --tgt_lang_name "Luang" \
#   --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-5-shot-word_lcs" \
#   --prompt "post_editing" \
#   --num_few_shot 5 \
#   --vectorizer "word_lcs" \

echo "All experiments completed! Results saved to: ${BASE_OUTPUT}"

# Return to project root
cd ../..
