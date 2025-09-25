#!/bin/bash

INPUT_FILE="./ebible-corpus/luang/aligned-eng-luang-gen.csv"  # Path to your input text file
MODEL_PATH="./nllb-200-distilled-600M-scripture_files-engwebp-lex-5000"  # Path to your model

python src/run_inference.py \
    --input_path "$INPUT_FILE" \
    --model_path "$MODEL_PATH" \
    --target_path "$INPUT_FILE" \
    --src_lang_nllb "ind_Latn" \
    --tgt_lang_nllb "lex_Latn" \
    --max_length 400 \
    --num_beams 8 \
    --per_device_eval_batch_size 64