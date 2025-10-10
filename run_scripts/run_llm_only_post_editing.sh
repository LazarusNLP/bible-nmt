#!/bin/bash
# LLM-only post-editing script (no glossary, no parallel sentences)
# Uses --few_shot_mode none to rely solely on LLM capabilities

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

# Configuration
BASE_OUTPUT="/data/projects/punim0478/setiawand/bible-nmt/results"
CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engwebp/aligned-eng-engwebp-ot.csv"
SRC="eng"
TGT="nfa"
SRC_LANG_NAME="English"
TGT_LANG_NAME="Dhao"

# Run LLM-only post-editing (no glossary, no parallel sentences)
echo "Starting LLM-only post-editing for webp English version..."
echo "Input: ${CSV_PATH}"
echo "Output: ${BASE_OUTPUT}/gemini-2.5-flash-webp-dhao/llm_only"
echo "Mode: LLM-only (no few-shot examples, no glossary)"
echo ""

python run_post_editing.py \
    --model_type "gemini" \
    --model_name "gemini-2.5-flash" \
    --csv_path ${CSV_PATH} \
    --src ${SRC} \
    --tgt ${TGT} \
    --src_lang_name ${SRC_LANG_NAME} \
    --tgt_lang_name ${TGT_LANG_NAME} \
    --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-webp-dhao/no_few_shot" \
    --prompt "dhao_post_editing" \
    --few_shot_mode "none" \
    --batch_size 100 \
    --delay_between_batches 5.0 \
    --max_samples 500

echo ""
echo "LLM-only post-editing completed!"
echo "Results saved to: ${BASE_OUTPUT}/gemini-2.5-flash-webp-dhao/no_few_shot"

