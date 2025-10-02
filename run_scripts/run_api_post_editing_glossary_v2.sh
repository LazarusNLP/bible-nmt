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
CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engwyc2017/aligned-eng-engwyc2017-ot.csv"
GLOSSARY_PATH="/data/projects/punim0478/setiawand/bible-nmt/lexical-resource/eng-dhao/dhao-english-dict-full.csv"
SRC="eng"
TGT="nfa"
SRC_LANG_NAME="English"
TGT_LANG_NAME="Dhao"

python run_post_editing.py \
    --model_type "gemini" \
    --model_name "gemini-2.5-flash" \
    --csv_path ${CSV_PATH} \
    --src "eng" \
    --tgt ${TGT} \
    --src_lang_name ${SRC_LANG_NAME} \
    --tgt_lang_name ${TGT_LANG_NAME} \
    --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-engwyc2017-dhao/glossary" \
    --prompt "dhao_post_editing" \
    --few_shot_mode "glossary" \
    --glossary_mode "word_fuzzy" \
    --glossary_path ${GLOSSARY_PATH} \
    --batch_size 100 \
    --delay_between_batches 5.0 \
    --top_n_per_word_glossary 5 \
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
    --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-engwyc2017-dhao/glossary_full" \
    --prompt "dhao_post_editing" \
    --few_shot_mode "glossary" \
    --glossary_mode "full" \
    --glossary_path ${GLOSSARY_PATH} \
    --batch_size 100 \
    --delay_between_batches 5.0 \
    --top_n_per_word_glossary 5 \
    --max_samples 500 \
    --debug

Sleep for 5 minutes to allow API rate limiting to reset
echo "Sleeping for 5 minutes to avoid API rate limiting..."
sleep 150
echo "Resuming execution..."

CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engwyc2018/aligned-eng-engwyc2018-ot.csv"
python run_post_editing.py \
    --model_type "gemini" \
    --model_name "gemini-2.5-flash" \
    --csv_path ${CSV_PATH} \
    --src "eng" \
    --tgt ${TGT} \
    --src_lang_name ${SRC_LANG_NAME} \
    --tgt_lang_name ${TGT_LANG_NAME} \
    --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-engwyc2018-dhao/glossary" \
    --prompt "dhao_post_editing" \
    --few_shot_mode "glossary" \
    --glossary_mode "word_fuzzy" \
    --glossary_path ${GLOSSARY_PATH} \
    --batch_size 100 \
    --delay_between_batches 5.0 \
    --top_n_per_word_glossary 5 \
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
    --output_dir "${BASE_OUTPUT}/gemini-2.5-flash-engwyc2018-dhao/glossary_full" \
    --prompt "dhao_post_editing" \
    --few_shot_mode "glossary" \
    --glossary_mode "full" \
    --glossary_path ${GLOSSARY_PATH} \
    --batch_size 100 \
    --delay_between_batches 5.0 \
    --top_n_per_word_glossary 5 \
    --max_samples 500 \
    --debug