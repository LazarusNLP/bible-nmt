#!/bin/bash
# Set working directory to project root
cd /data/projects/punim0478/setiawand/bible-nmt

# Define versions to process
# VERSIONS=("engfbv" "enggnv" "kjv2006" "asv" "engwebster" "engbsb")
VERSIONS=("enggnv" "engwebster")

# Loop through each version
for VARIANT in "${VERSIONS[@]}"; do
    echo "================================================"
    echo "Processing version: ${VARIANT}"
    echo "================================================"
    
    # Set version-specific paths
    CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/${VARIANT}/aligned-eng-${VARIANT}-ot.csv"
    POST_EDITED_CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/results/gemini-2.5-flash-${VARIANT}-dhao/glossary_full+parallel_full+nt/gemini-2.5-flash_aligned-eng-${VARIANT}-ot_eng_nfa.csv"
    FILTERED_OUTPUT="${POST_EDITED_CSV_PATH%.csv}-filtered.csv"
    
    # Check if input files exist
    if [ ! -f "${CSV_PATH}" ]; then
        echo "Warning: Input file not found: ${CSV_PATH}"
        continue
    fi
    
    if [ ! -f "${POST_EDITED_CSV_PATH}" ]; then
        echo "Warning: Post-edited file not found: ${POST_EDITED_CSV_PATH}"
        continue
    fi
    
    echo "  Input CSV: ${CSV_PATH}"
    echo "  Post-edited CSV: ${POST_EDITED_CSV_PATH}"
    echo "  Filtered output: ${FILTERED_OUTPUT}"
    
    python scripts/filter_and_calculate_metrics.py \
        --input_csv ${CSV_PATH} \
        --output_csv ${POST_EDITED_CSV_PATH} \
        --filtered_output ${FILTERED_OUTPUT} \
        --max_rows 500
    
    echo "Completed processing version: ${VARIANT}"
    echo ""
done

echo "================================================"
echo "All versions processed successfully!"
echo "================================================"