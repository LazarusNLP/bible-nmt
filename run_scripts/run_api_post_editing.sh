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

# Gemini API post-editing
python run_post_editing.py \
    --model_type "gemini" \
    --model_name "gemini-2.5-flash" \
    --csv_path /data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/luang/aligned-eng-luang-gen.csv \
    --src "eng" \
    --tgt "lex" \
    --src_lang_name "English" \
    --tgt_lang_name "Luang" \
    --output_dir "../../results/gemini-2.5-flash/all_parallel" \
    --prompt "luang_post_editing" \
    --num_workers 20 \
    --few_shot_corpus_path "/data/projects/punim0478/setiawand/bible-nmt/lexical-resource/eng-luang/luang-english-parallel.csv" \
    --vectorizer "full" \
    --sequential
        # --prompt-only
