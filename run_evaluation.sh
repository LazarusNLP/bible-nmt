iso_codes=(jav sun mad bvz ban bug)

for code in "${iso_codes[@]}"; do
    echo "Running for target language: $code"
    python src/run_evaluation.py \
        --model_name "nllb-200-distilled-1.3B-alkitab-sabda-mt-ind-${code}/checkpoint-5000" \
        --dataset_name LazarusNLP/alkitab-sabda-mt \
        --src_lang ind \
        --tgt_lang "$code" \
        --src_lang_nllb ind_Latn \
        --tgt_lang_nllb "${code}_Latn" \
        --max_length 256 \
        --num_beams 8 \
        --per_device_eval_batch_size 8 || exit 1
done