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
GLOSSARY_PATH="/data/projects/punim0478/setiawand/bible-nmt/lexical-resource/eng-dhao/dhao-english-dict-full.csv"
SRC="eng"
TGT="wyc"
SRC_LANG_NAME="English"
TGT_LANG_NAME="Wycliffe"

# Function to run experiment for a given Bible version
run_experiment() {
    local VERSION=$1
    local CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/${VERSION}+/aligned-eng-${VERSION}-ot.csv"
    local FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/${VERSION}+/aligned-eng-${VERSION}-nt.csv"
    
    # ==========================================
    # COMMENTED OUT: parallel_full+nt mode
    # ==========================================
    # echo "=========================================="
    # echo "Running experiment for ${VERSION} with parallel_full+nt"
    # echo "Input: OT corpus"
    # echo "Few-shot: NT corpus + full parallel corpus"
    # echo "=========================================="
    
    # Run with parallel_full+nt (using both NT and full parallel corpora)
    # python run_post_editing.py \
    #     --model_type "gemini" \
    #     --model_name "gemini-2.5-flash" \
    #     --csv_path ${CSV_PATH} \
    #     --src "eng" \
    #     --tgt ${TGT} \
    #     --src_lang_name ${SRC_LANG_NAME} \
    #     --tgt_lang_name ${TGT_LANG_NAME} \
    #     --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-${VERSION}/parallel_full+nt" \
    #     --prompt "dhao_post_editing" \
    #     --few_shot_mode "parallel" \
    #     --vectorizer "word_parallel" \
    #     --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
    #     --batch_size 100 \
    #     --delay_between_batches 5.0 \
    #     --top_n_per_word 5 \
    #     --max_samples 500 \
    #     --debug
    
    # echo "Sleeping for 1 minute between experiments..."
    # sleep 60
    
    # ==========================================
    # COMMENTED OUT: glossary_full mode
    # ==========================================
    # echo "=========================================="
    # echo "Running experiment for ${VERSION} with glossary_full"
    # echo "Input: OT corpus"
    # echo "Few-shot: Full glossary"
    # echo "=========================================="
    
    # Run with glossary_full
    # python run_post_editing.py \
    #     --model_type "gemini" \
    #     --model_name "gemini-2.5-flash" \
    #     --csv_path ${CSV_PATH} \
    #     --src "eng" \
    #     --tgt ${TGT} \
    #     --src_lang_name ${SRC_LANG_NAME} \
    #     --tgt_lang_name ${TGT_LANG_NAME} \
    #     --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-${VERSION}/glossary_full" \
    #     --prompt "dhao_post_editing" \
    #     --few_shot_mode "glossary" \
    #     --glossary_mode "full" \
    #     --glossary_path ${GLOSSARY_PATH} \
    #     --batch_size 100 \
    #     --delay_between_batches 5.0 \
    #     --top_n_per_word_glossary 5 \
    #     --max_samples 500 \
    #     --debug
    
    # echo "Sleeping for 1 minute between experiments..."
    # sleep 60
    
    echo "=========================================="
    echo "Running experiment for ${VERSION} with glossary_full+parallel_full+nt"
    echo "Input: OT corpus"
    echo "Few-shot: NT corpus + full parallel corpus + full glossary"
    echo "=========================================="
    
    # Run with glossary_full+parallel_full+nt (combining both approaches)
    python run_post_editing.py \
        --model_type "gemini" \
        --model_name "gemini-2.5-flash" \
        --csv_path ${CSV_PATH} \
        --src "eng" \
        --tgt ${TGT} \
        --src_lang_name ${SRC_LANG_NAME} \
        --tgt_lang_name ${TGT_LANG_NAME} \
        --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-${VERSION}+/glossary_full+parallel_full+nt" \
        --prompt "dhao_post_editing" \
        --few_shot_mode "both" \
        --glossary_mode "full" \
        --vectorizer "word_parallel" \
        --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
        --glossary_path ${GLOSSARY_PATH} \
        --batch_size 100 \
        --delay_between_batches 5.0 \
        --top_n_per_word 5 \
        --max_samples 500 \
        --debug
    
    echo "Completed all experiments for ${VERSION}"
    echo ""
}

# ==========================================
# COMMENTED OUT: webp Experiments (perplexity: 102.87)
# ==========================================

echo "Starting experiment for webp..."
run_experiment "engwebp"

Sleep between versions
echo "Sleeping for 2.5 minutes before next version..."
sleep 150

# ==========================================
# COMMENTED OUT: ENGLSV Experiments (perplexity: 102.87)
# ==========================================

# echo "Starting experiment for englsv..."
# run_experiment "englsv"

# Sleep between versions
# echo "Sleeping for 2.5 minutes before next version..."
# sleep 150

# ==========================================
# COMMENTED OUT: ENGOJB Experiments (perplexity: 140.66)
# ==========================================

# echo "Starting experiment for engojb..."
# run_experiment "engojb"

# Sleep between versions
# echo "Sleeping for 2.5 minutes before next version..."
# sleep 150

# ==========================================
# COMMENTED OUT: T4T Experiments (perplexity: 194.91)
# ==========================================

# echo "Starting experiment for t4t..."
# run_experiment "t4t"

# ==========================================
# ENGWYC2017 Experiments
# ==========================================

# echo "Starting experiment for engwyc2017..."
# run_experiment "engwyc2017"

# # Sleep between versions
# echo "Sleeping for 2.5 minutes before next version..."
# sleep 150

# ==========================================
# ENGWYC2018 Experiments
# ==========================================

# echo "Starting experiment for engwyc2018..."
# run_experiment "engwyc2018"

# echo "=========================================="
# echo "All experiments completed!"
# echo "=========================================="

