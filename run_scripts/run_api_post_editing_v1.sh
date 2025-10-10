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
CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engBBE/aligned-eng-engBBE-ot.csv"
FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engBBE/aligned-eng-engBBE-nt.csv"
ADDITIONAL_FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/lexical-resource/eng-dhao/dhao-english-parallel-full.csv"
SRC="eng"
TGT="nfa"
SRC_LANG_NAME="English"
TGT_LANG_NAME="Dhao"

# Gemini API post-editing
# python run_post_editing.py \
#     --model_type "gemini" \
#     --model_name "gemini-2.5-flash" \
#     --csv_path ${CSV_PATH} \
#     --src "eng" \
#     --tgt ${TGT} \
#     --src_lang_name ${SRC_LANG_NAME} \
#     --tgt_lang_name ${TGT_LANG_NAME} \
#     --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-engBBE-dhao/parallel_full" \
#     --prompt "dhao_post_editing" \
#     --few_shot_mode "parallel" \
#     --vectorizer "word_parallel" \
#     --few_shot_corpus_path ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
#     --batch_size 100 \
#     --delay_between_batches 5.0 \
#     --top_n_per_word 5 \
#     --max_samples 500 \
#     --debug

# python run_post_editing.py \
#     --model_type "gemini" \
#     --model_name "gemini-2.5-flash" \
#     --csv_path ${CSV_PATH} \
#     --src "eng" \
#     --tgt ${TGT} \
#     --src_lang_name ${SRC_LANG_NAME} \
#     --tgt_lang_name ${TGT_LANG_NAME} \
#     --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-engBBE-dhao/parallel_nt" \
#     --prompt "dhao_post_editing" \
#     --few_shot_mode "parallel" \
#     --vectorizer "word_parallel" \
#     --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} \
#     --batch_size 100 \
#     --delay_between_batches 5.0 \
#     --top_n_per_word 5 \
#     --max_samples 500 \
#     --debug

# python run_post_editing.py \
#     --model_type "gemini" \
#     --model_name "gemini-2.5-flash" \
#     --csv_path ${CSV_PATH} \
#     --src "eng" \
#     --tgt ${TGT} \
#     --src_lang_name ${SRC_LANG_NAME} \
#     --tgt_lang_name ${TGT_LANG_NAME} \
#     --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-engBBE-dhao/parallel_full+nt" \
#     --prompt "dhao_post_editing" \
#     --few_shot_mode "parallel" \
#     --vectorizer "word_parallel" \
#     --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
#     --batch_size 100 \
#     --delay_between_batches 5.0 \
#     --top_n_per_word 5 \
#     --max_samples 500 \
#     --debug


# Sleep for 5 minutes to allow API rate limiting to reset
# echo "Sleeping for 5 minutes to avoid API rate limiting..."
# sleep 300
# echo "Resuming execution..."

CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engwebp/aligned-eng-engwebp-ot.csv"
FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engwebp/aligned-eng-engwebp-nt.csv"
ADDITIONAL_FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/lexical-resource/eng-dhao/dhao-english-parallel-full.csv"

python run_post_editing.py \
    --model_type "gemini" \
    --model_name "gemini-2.5-flash" \
    --csv_path ${CSV_PATH} \
    --src "eng" \
    --tgt ${TGT} \
    --src_lang_name ${SRC_LANG_NAME} \
    --tgt_lang_name ${TGT_LANG_NAME} \
    --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-webp-dhao/parallel_full" \
    --prompt "dhao_post_editing" \
    --few_shot_mode "parallel" \
    --vectorizer "word_parallel" \
    --few_shot_corpus_path ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
    --batch_size 100 \
    --delay_between_batches 5.0 \
    --top_n_per_word 5 \
    --max_samples 500 \
    --debug

python run_post_editing.py \
    --model_type "gemini" \
    --model_name "gemini-2.5-flash" \
    --csv_path ${CSV_PATH} \
    --src "eng" \
    --tgt ${TGT} \
    --src_lang_name ${SRC_LANG_NAME} \
    --tgt_lang_name ${TGT_LANG_NAME} \
    --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-webp-dhao/parallel_nt" \
    --prompt "dhao_post_editing" \
    --few_shot_mode "parallel" \
    --vectorizer "word_parallel" \
    --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} \
    --batch_size 100 \
    --delay_between_batches 5.0 \
    --top_n_per_word 5 \
    --max_samples 500 \
    --debug
 
python run_post_editing.py \
    --model_type "gemini" \
    --model_name "gemini-2.5-flash" \
    --csv_path ${CSV_PATH} \
    --src "eng" \
    --tgt ${TGT} \
    --src_lang_name ${SRC_LANG_NAME} \
    --tgt_lang_name ${TGT_LANG_NAME} \
    --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-webp-dhao/parallel_full+nt" \
    --prompt "dhao_post_editing" \
    --few_shot_mode "parallel" \
    --vectorizer "word_parallel" \
    --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
    --batch_size 100 \
    --delay_between_batches 5.0 \
    --top_n_per_word 5 \
    --max_samples 500 \
    --debug