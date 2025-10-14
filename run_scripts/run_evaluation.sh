iso_codes=(engwebp)

for code in "${iso_codes[@]}"; do
    echo "Running evaluation for target language: $code"
    python src/run_evaluation.py \
        --model_name "nllb-models/english/nllb-200-distilled-600M-eng-${code}-nfa-7000-epochs/checkpoint-7000" \
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

