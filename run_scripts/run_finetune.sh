#!/bin/bash

# Bible NMT Finetuning Script for CSV data
# This script finetunes a pre-trained model on CSV parallel text data

# Target language code (can be modified as needed)
tgt_code="lex"

echo "Running finetuning for target language: $tgt_code"

python src/run_finetune.py \
    --train_csv_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/luang/luang-english-parallel.csv" \
    --test_csv_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/luang/aligned-eng-luang-gen.csv" \
    --src_lang "eng" \
    --tgt_lang "luang" \
    --src_lang_nllb "eng_Latn" \
    --tgt_lang_nllb "${tgt_code}_Latn" \
    --model_name "/data/projects/punim0478/setiawand/bible-nmt/nllb-200-distilled-600M-scripture_files-engwebp-lex-5000/checkpoint-5000" \
    --max_length 300 \
    --num_beams 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --label_smoothing_factor 0.2 \
    --max_steps 5000 \
    --warmup_steps 300 \
    --early_stopping_patience 4 \
    --validation_split 0.05 \
    --save_tokenized_data \
    --torch_dtype "bfloat16" \
    --attn_implementation "sdpa" \
    --seed 114

echo "Finetuning completed!"
