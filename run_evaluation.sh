iso_codes=(aaz ptu nfa heg lex row llg rgu txq tet wrs)

for code in "${iso_codes[@]}"; do
    echo "Running for target language: $code"
    python src/run_evaluation.py \
        --model_name "nllb-200-distilled-1.3B-biblenlp-corpus-ind-${code}/checkpoint-5000" \
        --dataset_name bible-nlp/biblenlp-corpus \
        --src_lang ind \
        --tgt_lang "$code" \
        --src_lang_nllb ind_Latn \
        --tgt_lang_nllb "${code}_Latn" \
        --max_length 256 \
        --num_beams 8 \
        --per_device_eval_batch_size 8 || exit 1
done