iso_codes=(ptu heg lex tet wrs)
# iso_codes=(ptu)

for code in "${iso_codes[@]}"; do
    echo "Running for target language: $code"
    python src/run_translation.py \
        --model_name facebook/nllb-200-distilled-600M \
        --dataset_name Davidsamuel101/ebible_local_ind_corpus \
        --src_lang ind \
        --tgt_lang "$code" \
        --src_lang_nllb ind_Latn \
        --tgt_lang_nllb "${code}_Latn" \
        --max_length 256 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-4 \
        --lr_scheduler_type cosine \
        --warmup_steps 1000 \
        --num_beams 2 \
        --max_steps 5000 || { echo "Failed for $code, continuing..."; continue; }
done

# src_lang="hau"
# tgt_lang="daa"

# python src/run_translation.py \
#     --model_name facebook/nllb-200-distilled-600M \
#     --dataset_name bible-nlp/biblenlp-corpus \
#     --src_lang $src_lang \
#     --tgt_lang $tgt_lang \
#     --src_lang_nllb "${src_lang}_Latn" \
#     --tgt_lang_nllb "${tgt_lang}_Latn" \
#     --max_length 200 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 5e-5 \
#     --warmup_steps 4000 \
#     --lr_scheduler_type linear \
#     --early_stopping_patience 4 \
#     --early_stopping_threshold 0.1 \
#     --label_smoothing_factor 0.2 \
#     --num_beams 2 \
#     --max_steps 100000 
