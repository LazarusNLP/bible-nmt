#!/bin/bash
# Set working directory to project root
cd /data/projects/punim0478/setiawand/bible-nmt

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found at $(pwd)/.env"
fi

# Change to post-editing directory for imports to work correctly
cd src/post-editing

BASE_OUTPUT="/data/projects/punim0478/setiawand/bible-nmt/results"
ADDITIONAL_FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/lexical-resource/eng-dhao/dhao-english-parallel-full.csv"
SRC="eng"
TGT="nfa"
SRC_LANG_NAME="English"
TGT_LANG_NAME="Dhao"

# Function to run experiments for a given version and corpus type
run_experiment() {
    local VERSION=$1
    local CORPUS_TYPE=$2  # "all", "nt", etc.
    local CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/${VERSION}/aligned-eng-${VERSION}-${CORPUS_TYPE}.csv"
    local FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/${VERSION}/aligned-eng-${VERSION}-nt.csv"
    
    echo "=========================================="
    echo "Running experiments for ${VERSION} - ${CORPUS_TYPE}"
    echo "=========================================="
    
    # Run with parallel_full (using additional few-shot corpus)
    python run_post_editing.py \
        --model_type "gemini" \
        --model_name "gemini-2.5-flash" \
        --csv_path ${CSV_PATH} \
        --src "eng" \
        --tgt ${TGT} \
        --src_lang_name ${SRC_LANG_NAME} \
        --tgt_lang_name ${TGT_LANG_NAME} \
        --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-${VERSION}-dhao/${CORPUS_TYPE}/parallel_full" \
        --prompt "dhao_post_editing" \
        --few_shot_mode "parallel" \
        --vectorizer "word_parallel" \
        --few_shot_corpus_path ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
        --batch_size 100 \
        --delay_between_batches 5.0 \
        --top_n_per_word 5 \
        --max_samples 500 \
        --debug
    
    # Run with parallel_nt (using NT corpus)
    python run_post_editing.py \
        --model_type "gemini" \
        --model_name "gemini-2.5-flash" \
        --csv_path ${CSV_PATH} \
        --src "eng" \
        --tgt ${TGT} \
        --src_lang_name ${SRC_LANG_NAME} \
        --tgt_lang_name ${TGT_LANG_NAME} \
        --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-${VERSION}-dhao/${CORPUS_TYPE}/parallel_nt" \
        --prompt "dhao_post_editing" \
        --few_shot_mode "parallel" \
        --vectorizer "word_parallel" \
        --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} \
        --batch_size 100 \
        --delay_between_batches 5.0 \
        --top_n_per_word 5 \
        --max_samples 500 \
        --debug
    
    # Run with parallel_full+nt (using both corpora)
    python run_post_editing.py \
        --model_type "gemini" \
        --model_name "gemini-2.5-flash" \
        --csv_path ${CSV_PATH} \
        --src "eng" \
        --tgt ${TGT} \
        --src_lang_name ${SRC_LANG_NAME} \
        --tgt_lang_name ${TGT_LANG_NAME} \
        --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-${VERSION}-dhao/${CORPUS_TYPE}/parallel_full+nt" \
        --prompt "dhao_post_editing" \
        --few_shot_mode "parallel" \
        --vectorizer "word_parallel" \
        --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
        --batch_size 100 \
        --delay_between_batches 5.0 \
        --top_n_per_word 5 \
        --max_samples 500 \
        --debug
    
    echo "Completed experiments for ${VERSION} - ${CORPUS_TYPE}"
    echo ""
}

# ==========================================
# ENGLSV Experiments (perplexity: 102.87)
# ==========================================

echo "Starting experiments for englsv..."

# Run englsv with full Bible (all)
run_experiment "englsv" "all"

# Sleep between corpus types
echo "Sleeping for 2.5 minutes to avoid API rate limiting..."
sleep 150

# Run englsv with New Testament only
run_experiment "englsv" "nt"

# Sleep between versions
echo "Sleeping for 5 minutes before next version..."
sleep 300

# ==========================================
# ENGOJB Experiments (perplexity: 140.66)
# ==========================================

echo "Starting experiments for engojb..."

# Run engojb with full Bible (all)
run_experiment "engojb" "all"

# Sleep between corpus types
echo "Sleeping for 2.5 minutes to avoid API rate limiting..."
sleep 150

# Run engojb with New Testament only
run_experiment "engojb" "nt"

# Sleep between versions
echo "Sleeping for 5 minutes before next version..."
sleep 300

# ==========================================
# T4T Experiments (perplexity: 194.91)
# ==========================================

echo "Starting experiments for t4t..."

# Run t4t with full Bible (all)
run_experiment "t4t" "all"

# Sleep between corpus types
echo "Sleeping for 2.5 minutes to avoid API rate limiting..."
sleep 150

# Run t4t with New Testament only
run_experiment "t4t" "nt"

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

