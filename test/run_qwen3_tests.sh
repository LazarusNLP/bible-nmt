#!/bin/bash

# Simple script to run All-MPNet embedding tests
# Usage: cd bible-nmt && bash test/run_all_mpnet_tests.sh

echo "🚀 Running All-MPNet Embedding Tests"
echo "====================================="

# No special environment variables needed for transformers

# Check if we're in the correct directory
if [ ! -f "src/post-editing/run_post_editing.py" ]; then
    echo "❌ Error: Please run this script from the bible-nmt directory"
    echo "Usage: cd bible-nmt && bash test/run_all_mpnet_tests.sh"
    exit 1
fi

# Activate conda environment (if not already active)
if [ "$CONDA_DEFAULT_ENV" != "bible-nmt" ]; then
    echo "Activating bible-nmt conda environment..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate bible-nmt
fi

echo "Current environment: $CONDA_DEFAULT_ENV"
echo ""

# Run the comprehensive test
echo "Running comprehensive test suite..."
python test/test_qwen3_embedding_retrieval.py

test_exit_code=$?

if [ $test_exit_code -eq 0 ]; then
    echo ""
    echo "🎉 All tests passed! You can now use the all_mpnet vectorizer."
    echo ""
    echo "Example usage in post-editing pipeline:"
    echo "python src/post-editing/run_post_editing.py \\"
    echo "  --vectorizer all_mpnet \\"
    echo "  --model_type vllm \\"
    echo "  --model_name your_model \\"
    echo "  --csv_path data.csv \\"
    echo "  --few_shot_corpus_path corpus.csv \\"
    echo "  --src en --tgt id \\"
    echo "  --src_lang_name English \\"
    echo "  --tgt_lang_name Indonesian \\"
    echo "  --output_dir ./results"
else
    echo ""
    echo "❌ Some tests failed. Check the output above for details."
    echo "Make sure you have transformers and torch installed."
fi

exit $test_exit_code

