iso_codes=(tet ptu wrs row nfa aaz lex)
iso_codes=(lex wrs ptu)

# Choose evaluation dataset mode:
# Option 1: Use ebible_local_ind_corpus from HuggingFace (default)
# Option 2: Use scripture_files with local text files (uncomment below)

# Option 1: Using HuggingFace dataset
# for code in "${iso_codes[@]}"; do
#     echo "Running for target language: $code"
#     python src/run_evaluation.py \
#         --model_name "nllb-200-distilled-600M-biblenlp-corpus-eng-$code-v2-5000/checkpoint-5000" \
#         --dataset_name bible-nlp/biblenlp-corpus \
#         --src_lang eng \
#         --tgt_lang "$code" \
#         --src_lang_nllb eng_Latn \
#         --tgt_lang_nllb "${code}_Latn" \
#         --max_length 512 \
#         --num_beams 2 \
#         --per_device_eval_batch_size 16 || exit 1
# done

# Option 2: Using scripture_files pre-split data saved with the model (preferred)
# The script will automatically read validation.* and test.* from the model directory.
for code in "${iso_codes[@]}"; do
    echo "Running evaluation for target language: $code"
    python src/run_evaluation.py \
        --model_name "nllb-200-distilled-600M-scripture_files-engwebp-${code}-5000/checkpoint-5000" \
        --dataset_name "scripture_files" \
        --verse_text_path "/data/projects/punim0478/setiawand/ebible/metadata/vref.txt" \
        --src_lang eng \
        --tgt_lang "$code" \
        --src_lang_nllb eng_Latn \
        --tgt_lang_nllb "${code}_Latn" \
        --max_length 512 \
        --num_beams 8 \
        --per_device_eval_batch_size 16 || exit 1
done

