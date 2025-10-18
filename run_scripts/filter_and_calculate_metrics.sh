CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engwebp/aligned-eng-engwebp-ot.csv"
POST_EDITED_CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/results/gemini-2.5-flash-webp-dhao/glossary_full+parallel_full+nt_n10/gemini-2.5-flash_aligned-eng-engwebp-ot_eng_nfa.csv"
FILTERED_OUTPUT="${POST_EDITED_CSV_PATH%.csv}-filtered.csv"

python scripts/filter_and_calculate_metrics.py \
    --input_csv ${CSV_PATH} \
    --output_csv ${POST_EDITED_CSV_PATH} \
    --filtered_output ${FILTERED_OUTPUT} \
    --max_rows 500