#!/bin/bash

# Script to create Indonesian Bible datasets for all target languages
# Automatically infers target filenames from language codes

# Use Python from david conda environment
PYTHON="/data/projects/punim0478/setiawand/.cache/conda-envs/david/bin/python"

# Set source configuration
SOURCE_NAME="ind-indayt.txt"  # or ind-indayt.txt or ind-indags.txt
SOURCE_LANG="ind"
OUTPUT_DIR="./data"

# Define target languages
TARGET_LANGS=("aaz" "ptu" "nfa" "heg" "lex" "row" "llg" "rgu" "txq" "tet" "wrs")

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "Creating Indonesian Bible datasets"
echo "Source: $SOURCE_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="
echo ""

# Process each target language
for target_lang in "${TARGET_LANGS[@]}"; do
    echo "----------------------------------------------"
    echo "Processing target language: $target_lang"
    echo "----------------------------------------------"
    
    # Infer target filename based on language code
    # Special case for heg which uses hegNTpo
    if [ "$target_lang" = "heg" ]; then
        target_name="heg-hegNTpo.txt"
    else
        target_name="${target_lang}-${target_lang}.txt"
    fi
    
    # Check if target file exists
    target_path="/data/gpfs/projects/punim0478/setiawand/ebible/corpus/$target_name"
    if [ ! -f "$target_path" ]; then
        echo "WARNING: Target file not found: $target_name"
        echo "Skipping $target_lang..."
        echo ""
        continue
    fi
    
    echo "Target file: $target_name"
    echo "Creating dataset: ${SOURCE_LANG}_${target_lang}"
    
    # Run the Python script
    $PYTHON scripts/create_indonesian_dataset.py \
        --source_name "$SOURCE_NAME" \
        --target_name "$target_name" \
        --source_lang "$SOURCE_LANG" \
        --target_lang "$target_lang" \
        --output_dir "$OUTPUT_DIR"
    
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "✓ Successfully created dataset for $target_lang"
    else
        echo "✗ Failed to create dataset for $target_lang"
    fi
    
    echo ""
done

echo "=============================================="
echo "Dataset creation complete!"
echo "=============================================="

# List created datasets
echo ""
echo "Created datasets:"
ls -la $OUTPUT_DIR/