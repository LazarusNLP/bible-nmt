iso_codes=(row nfa lex aaz tet ptu wrs)
iso_codes=(lex)
# for code in "${iso_codes[@]}"; do
#     echo "Running evaluation for target language: $code"
#     python src/run_evaluation.py \
#         --model_name "nllb-200-distilled-600M-scripture_files-engwebp-${code}-5000/checkpoint-5000" \
#         --dataset_name "scripture_files" \
#         --verse_text_path "/data/projects/punim0478/setiawand/ebible/metadata/vref.txt" \
#         --src_lang eng \
#         --tgt_lang "$code" \
#         --src_lang_nllb eng_Latn \
#         --tgt_lang_nllb "${code}_Latn" \
#         --max_length 400 \
#         --num_beams 8 \
#         --per_device_eval_batch_size 16 || exit 1

for code in "${iso_codes[@]}"; do
    echo "Running evaluation for target language: $code"
    python src/run_evaluation.py \
        --model_name "checkpoint-5000-finetune-eng-luang-3000/checkpoint-3000" \
        --dataset_name "scripture_files" \
        --verse_text_path "/data/projects/punim0478/setiawand/ebible/metadata/vref.txt" \
        --src_lang eng \
        --tgt_lang "$code" \
        --src_lang_nllb eng_Latn \
        --tgt_lang_nllb "${code}_Latn" \
        --max_length 400 \
        --num_beams 8 \
        --per_device_eval_batch_size 16 || exit 1
done

