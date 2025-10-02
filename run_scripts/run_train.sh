#!/bin/bash

iso_codes=(nfa)
eng_version=(eng-engBBE eng-engwyc2018 eng-engwebp)
eng_version=(eng-engwyc2017)
for eng_version in "${eng_version[@]}"; do
  for code in "${iso_codes[@]}"; do
    echo "Running for target language: $code" 
    python src/run_translation.py \
        --dataset_name "scripture_files" \
        --source_text_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/eng/corpus/${eng_version}.txt" \
        --target_text_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/nfa-nfa.txt" \
        --verse_text_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/vref.txt" \
        --output_dir "nllb-models/english/nllb-200-distilled-600M-${eng_version}-${code}" \
        --src_lang "eng" \
        --tgt_lang "$code" \
        --csv_source_col "source_text" \
        --csv_target_col "target_text" \
        --src_lang_nllb "eng_Latn" \
        --tgt_lang_nllb "${code}_Latn" \
        --model_name "facebook/nllb-200-distilled-600M" \
        --max_length 400 \
        --num_beams 2 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-4 \
        --label_smoothing_factor 0.2 \
        --max_steps 5000 \
        --warmup_steps 1000 \
        --early_stopping_patience 4 \
        --save_tokenized_data \
        --torch_dtype "bfloat16" \
        --attn_implementation "sdpa"
  done
done

# for eng_version in "${eng_version[@]}"; do
#   for code in "${iso_codes[@]}"; do
#     echo "Running for target language: $code" 
#     python src/run_translation.py \
#         --dataset_name "scripture_files" \
#         --source_text_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/eng/corpus/${eng_version}.txt" \
#         --target_text_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/nfa-nfa.txt" \
#         --verse_text_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/vref.txt" \
#         --output_dir "nllb-models/english/nllb-200-distilled-600M-${eng_version}-${code}" \
#         --src_lang "eng" \
#         --tgt_lang "$code" \
#         --additional_csv_path "lexical-resource/eng-dhao/dhao-english-parallel-full.csv" \
#         --csv_source_col "source_text" \
#         --csv_target_col "target_text" \
#         --src_lang_nllb "eng_Latn" \
#         --tgt_lang_nllb "${code}_Latn" \
#         --model_name "facebook/nllb-200-distilled-600M" \
#         --max_length 400 \
#         --num_beams 2 \
#         --per_device_train_batch_size 16 \
#         --per_device_eval_batch_size 16 \
#         --gradient_accumulation_steps 4 \
#         --learning_rate 2e-4 \
#         --label_smoothing_factor 0.2 \
#         --max_steps 5000 \
#         --warmup_steps 1000 \
#         --early_stopping_patience 4 \
#         --save_tokenized_data \
#         --torch_dtype "float32" \
#         --attn_implementation "sdpa"
#   done
# done