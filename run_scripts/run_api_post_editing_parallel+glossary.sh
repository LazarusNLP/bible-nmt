#!/bin/bash
# Ablation experiment for top_n_per_word_glossary parameter
# Testing values: 5, 10, 20, 40, 60, 80, 100
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
GLOSSARY_PATH="/data/projects/punim0478/setiawand/bible-nmt/lexical-resource/eng-dhao/dhao-english-dict-full.csv"
ADDITIONAL_FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/lexical-resource/eng-dhao/dhao-english-parallel-full.csv"
SRC="eng"
TGT="nfa"
SRC_LANG_NAME="English"
TGT_LANG_NAME="Dhao"

# Define versions to process
VERSIONS=("engwebster" "enggnv")

# Loop through each version
for VERSION in "${VERSIONS[@]}"; do
    echo "================================================"
    echo "Processing version: ${VERSION}"
    echo "================================================"
    
    # Set version-specific paths
    CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/${VERSION}/aligned-eng-${VERSION}-ot.csv"
    FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/${VERSION}/aligned-eng-${VERSION}-nt.csv"
    
    # Loop through different top_n_per_word_glossary values
    for n in 5 10 20 40 60 80 100; do
        echo "Running ${VERSION} experiment with top_n_per_word_glossary=${n}..."
        
        # Gemini API post-editing
        python run_post_editing.py \
            --model_type "gemini" \
            --model_name "gemini-2.5-flash" \
            --csv_path ${CSV_PATH} \
            --src "eng" \
            --tgt ${TGT} \
            --src_lang_name ${SRC_LANG_NAME} \
            --tgt_lang_name ${TGT_LANG_NAME} \
            --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-${VERSION}-dhao/glossary_n${n}+parallel_full+nt" \
            --prompt "dhao_post_editing" \
            --few_shot_mode "both" \
            --glossary_mode "word_fuzzy" \
            --vectorizer "word_parallel" \
            --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
            --glossary_path ${GLOSSARY_PATH} \
            --batch_size 50 \
            --delay_between_batches 30.0 \
            --top_n_per_word 5 \
            --top_n_per_word_glossary ${n} \
            --max_samples 500 \
            --debug
        
        echo "Completed ${VERSION} experiment with n=${n}"
        
        # Add delay between experiments except for the last one
        if [ ${n} != 100 ]; then
            echo "Sleeping for 2 minutes to avoid API rate limiting..."
            sleep 120
        fi
    done
    
    echo "Completed processing version: ${VERSION}"
    echo ""
done

echo "================================================"
echo "All versions processed successfully!"
echo "================================================"
