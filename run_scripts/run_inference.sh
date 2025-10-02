#!/bin/bash
iso_codes=(engBBE engwebp engwyc2017 engwyc2018)

BASE_INPUT_FILE="./ebible-corpus/dhao-eng"
BASE_MODEL_PATH="./nllb-models/english/nllb-200-distilled-600M-eng"

for code in "${iso_codes[@]}"; do
    echo "Running inference for target language: $code"
    MODEL_PATH="${BASE_MODEL_PATH}-${code}-nfa/checkpoint-5000"
    INPUT_FILE="${BASE_INPUT_FILE}/${code}/aligned-eng-${code}-ot.csv"
    
    python src/run_inference.py \
        --input_path "$INPUT_FILE" \
        --model_path "$MODEL_PATH" \
        --target_path "$INPUT_FILE" \
        --src_lang_nllb "ind_Latn" \
        --tgt_lang_nllb "nfa_Latn" \
        --max_length 400 \
        --num_beams 8 \
        --per_device_eval_batch_size 64
done