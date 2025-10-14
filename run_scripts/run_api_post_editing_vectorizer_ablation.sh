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
CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engwebp/aligned-eng-engwebp-ot.csv"
FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engwebp/aligned-eng-engwebp-nt.csv"
ADDITIONAL_FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/lexical-resource/eng-dhao/dhao-english-parallel-full.csv"
SRC="eng"
TGT="nfa"
SRC_LANG_NAME="English"
TGT_LANG_NAME="Dhao"

echo "================================================"
echo "VECTORIZER ABLATION STUDY"
echo "================================================"
echo "English Version: engwebp"
echo "Corpus: OT (test) with NT+full (few-shot)"
echo "Start time: $(date)"
echo "================================================"

# ==========================================
# Random Baseline Vectorizer Experiments
# ==========================================

echo ""
echo "=========================================="
echo "Starting Random Baseline experiments..."
echo "=========================================="

echo ""
echo "Running Random Baseline with k=5..."

python run_post_editing.py \
    --model_type "gemini" \
    --model_name "gemini-2.5-flash" \
    --csv_path ${CSV_PATH} \
    --src ${SRC} \
    --tgt ${TGT} \
    --src_lang_name ${SRC_LANG_NAME} \
    --tgt_lang_name ${TGT_LANG_NAME} \
    --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-webp-dhao/vectorizer_ablation/random_k5" \
    --prompt "dhao_post_editing" \
    --few_shot_mode "parallel" \
    --vectorizer "random" \
    --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} \
    --batch_size 100 \
    --delay_between_batches 5.0 \
    --num_few_shot 5 \
    --max_samples 500 \
    --debug

echo "Completed Random Baseline k=5"

echo ""
echo "Random Baseline experiments completed!"

# ==========================================
# BM25 Vectorizer Experiments (COMMENTED OUT)
# ==========================================

# echo ""
# echo "=========================================="
# echo "Starting BM25 experiments..."
# echo "=========================================="

# for K_VALUE in 5 10 20 40; do
#     echo ""
#     echo "Running BM25 with k=${K_VALUE}..."
#     
#     python run_post_editing.py \
#         --model_type "gemini" \
#         --model_name "gemini-2.5-flash" \
#         --csv_path ${CSV_PATH} \
#         --src ${SRC} \
#         --tgt ${TGT} \
#         --src_lang_name ${SRC_LANG_NAME} \
#         --tgt_lang_name ${TGT_LANG_NAME} \
#         --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-webp-dhao/vectorizer_ablation/bm25_k${K_VALUE}" \
#         --prompt "dhao_post_editing" \
#         --few_shot_mode "parallel" \
#         --vectorizer "bm25" \
#         --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
#         --batch_size 100 \
#         --delay_between_batches 5.0 \
#         --num_few_shot ${K_VALUE} \
#         --max_samples 500 \
#         --debug
#     
#     echo "Completed BM25 k=${K_VALUE}"
#     echo "Sleeping for 2 minutes..."
#     sleep 120
# done

# echo ""
# echo "BM25 experiments completed!"

# ==========================================
# BGE (English Semantic SOTA) Vectorizer Experiments (COMMENTED OUT)
# ==========================================

# echo ""
# echo "=========================================="
# echo "Starting BGE-large-en-v1.5 (SOTA English bi-encoder) experiments..."
# echo "=========================================="

# for K_VALUE in 5 10 20 40; do
#     echo ""
#     echo "Running BGE with k=${K_VALUE}..."
#     
#     python run_post_editing.py \
#         --model_type "gemini" \
#         --model_name "gemini-2.5-flash" \
#         --csv_path ${CSV_PATH} \
#         --src ${SRC} \
#         --tgt ${TGT} \
#         --src_lang_name ${SRC_LANG_NAME} \
#         --tgt_lang_name ${TGT_LANG_NAME} \
#         --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-webp-dhao/vectorizer_ablation/bge_k${K_VALUE}" \
#         --prompt "dhao_post_editing" \
#         --few_shot_mode "parallel" \
#         --vectorizer "bge" \
#         --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
#         --batch_size 100 \
#         --delay_between_batches 5.0 \
#         --num_few_shot ${K_VALUE} \
#         --max_samples 500 \
#         --debug
#     
#     echo "Completed BGE k=${K_VALUE}"
#     echo "Sleeping for 2 minutes..."
#     sleep 120
# done

# echo ""
# echo "BGE experiments completed!"

# ==========================================
# ChrF-RAG (Character n-gram) Vectorizer Experiments (COMMENTED OUT)
# ==========================================

# echo ""
# echo "=========================================="
# echo "Starting ChrF-RAG (character n-gram similarity) experiments..."
# echo "=========================================="

# for K_VALUE in 5 10 20 40; do
#     echo ""
#     echo "Running ChrF-RAG with k=${K_VALUE}..."
#     
#     python run_post_editing.py \
#         --model_type "gemini" \
#         --model_name "gemini-2.5-flash" \
#         --csv_path ${CSV_PATH} \
#         --src ${SRC} \
#         --tgt ${TGT} \
#         --src_lang_name ${SRC_LANG_NAME} \
#         --tgt_lang_name ${TGT_LANG_NAME} \
#         --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-webp-dhao/vectorizer_ablation/chrf_rag_k${K_VALUE}" \
#         --prompt "dhao_post_editing" \
#         --few_shot_mode "parallel" \
#         --vectorizer "chrf_rag" \
#         --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
#         --batch_size 100 \
#         --delay_between_batches 5.0 \
#         --num_few_shot ${K_VALUE} \
#         --max_samples 500 \
#         --debug
#     
#     echo "Completed ChrF-RAG k=${K_VALUE}"
#     echo "Sleeping for 2 minutes..."
#     sleep 120
# done

# echo ""
# echo "ChrF-RAG experiments completed!"

# ==========================================
# Word Parallel Vectorizer Experiments (COMMENTED OUT)
# ==========================================

# echo ""
# echo "=========================================="
# echo "Starting Word Parallel experiments..."
# echo "=========================================="

# for TOP_N in 1 2 3; do
#     echo ""
#     echo "Running Word Parallel with top_n_per_word=${TOP_N}..."
#     
#     python run_post_editing.py \
#         --model_type "gemini" \
#         --model_name "gemini-2.5-flash" \
#         --csv_path ${CSV_PATH} \
#         --src ${SRC} \
#         --tgt ${TGT} \
#         --src_lang_name ${SRC_LANG_NAME} \
#         --tgt_lang_name ${TGT_LANG_NAME} \
#         --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-webp-dhao/vectorizer_ablation/word_parallel_top${TOP_N}" \
#         --prompt "dhao_post_editing" \
#         --few_shot_mode "parallel" \
#         --vectorizer "word_parallel" \
#         --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
#         --batch_size 100 \
#         --delay_between_batches 5.0 \
#         --top_n_per_word ${TOP_N} \
#         --max_samples 500 \
#         --debug
#     
#     echo "Completed Word Parallel top_n_per_word=${TOP_N}"
#     echo "Sleeping for 2 minutes..."
#     sleep 120
# done

# echo ""
# echo "Word Parallel experiments completed!"

echo ""
echo "================================================"
echo "RANDOM BASELINE EXPERIMENT COMPLETED!"
echo "================================================"
echo "End time: $(date)"
echo "Total experiments: 1"
echo "  - Random Baseline: 1 (k=5) [Random selection baseline for comparison]"
echo "Results saved in: results/gemini-2.5-flash-webp-dhao/vectorizer_ablation/"
echo ""
echo "NOTE: Other experiments (BM25, BGE, ChrF-RAG, Word Parallel) have been commented out."
echo "      Uncomment them in the script if you want to run the full ablation study."
echo "================================================"

