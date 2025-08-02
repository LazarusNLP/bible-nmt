iso_codes=(jav sun mad bvz ban bug mak sda btx bts bbc mqj)

for code in "${iso_codes[@]}"; do
    echo "Running for target language: $code"
    python src/run_translation.py \
        --model_name facebook/nllb-200-distilled-1.3B \
        --dataset_name LazarusNLP/alkitab-sabda-mt \
        --src_lang ind \
        --tgt_lang "$code" \
        --src_lang_nllb ind_Latn \
        --tgt_lang_nllb "${code}_Latn" \
        --max_length 256 \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 16 \
        --learning_rate 2e-4 \
        --max_steps 5000 || { echo "Failed for $code, continuing..."; continue; }
done