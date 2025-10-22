iso_codes=(engULB)

BASE_INPUT_FILE="./ebible-corpus/dhao-eng"

for code in "${iso_codes[@]}"; do
    echo "Running evaluation for target language: $code"
    CSV_FILE="${BASE_INPUT_FILE}/${code}/aligned-eng-${code}-ot.csv"
    
    python src/run_evaluation.py \
        --model_name "nllb-models/english/nllb-200-distilled-600M-eng-${code}-nfa/checkpoint-5000" \
        --dataset_name "csv_file" \
        --csv_file_path "$CSV_FILE" \
        --src_lang eng \
        --tgt_lang "$code" \
        --src_lang_nllb "ind_Latn" \
        --tgt_lang_nllb "nfa_Latn" \
        --max_length 400 \
        --num_beams 2 \
        --per_device_eval_batch_size 8 || exit 1
done

 