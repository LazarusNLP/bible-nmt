iso_codes=(tet row nfa aaz)
iso_codes=(lex)

for code in "${iso_codes[@]}"; do
  echo "Running for target language: $code" 
  python src/run_translation.py \
      --dataset_name "scripture_files" \
      --source_text_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/eng-webp/eng-webp-norm.txt" \
      --target_text_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/luang/lex-norm.txt" \
      --verse_text_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/vref.txt" \
      --src_lang "eng" \
      --tgt_lang "$code" \
      --additional_csv_path "ebible-corpus/luang/luang-english-parallel.csv" \
      --csv_source_col "source_text" \
      --csv_target_col "target_text" \
      --src_lang_nllb "eng_Latn" \
      --tgt_lang_nllb "${code}_Latn" \
      --model_name "facebook/nllb-200-distilled-600M" \
      --max_length 300 \
      --num_beams 2 \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 16 \
      --gradient_accumulation_steps 4 \
      --learning_rate 2e-4 \
      --label_smoothing_factor 0.2 \
      --max_steps 5000 \
      --warmup_steps 1000 \
      --gradient_accumulation_steps 1 \
      --early_stopping_patience 4 \
      --save_tokenized_data \
      --torch_dtype "float32" \
      --attn_implementation "sdpa"
done

# for code in "${iso_codes[@]}"; do
#   echo "Running for target language: $code" 
#   python src/run_translation.py \
#       --dataset_name "bible-nlp/biblenlp-corpus" \
#       --src_lang "eng" \
#       --tgt_lang "$code" \
#       --src_lang_nllb "eng_Latn" \
#       --tgt_lang_nllb "${code}_Latn" \
#       --model_name "facebook/nllb-200-distilled-600M" \
#       --max_length 256 \
#       --num_beams 2 \
#       --per_device_train_batch_size 16 \
#       --per_device_eval_batch_size 16 \
#       --gradient_accumulation_steps 4 \
#       --learning_rate 2e-4 \
#       --label_smoothing_factor 0.2 \
#       --max_steps 5000 \
#       --warmup_steps 1000 \
#       --gradient_accumulation_steps 1 \
#       --early_stopping_patience 4 \
#       --save_tokenized_data \
#       --torch_dtype "float32" \
#       --attn_implementation "sdpa"
# done