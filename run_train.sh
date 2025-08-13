# iso_codes=(aaz ptu nfa heg lex row llg rgu txq tet wrs)
iso_codes=(aaz)

for code in "${iso_codes[@]}"; do
    echo "Running for target language: $code"
    python src/run_translation.py \
        --model_name facebook/nllb-200-distilled-1.3B \
        --dataset_name bible-nlp/biblenlp-corpus \
        --src_lang ind \
        --tgt_lang "$code" \
        --src_lang_nllb ind_Latn \
        --tgt_lang_nllb "${code}_Latn" \
        --max_length 512 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate 2e-4 \
        --max_steps 10000 || { echo "Failed for $code, continuing..."; continue; }
done