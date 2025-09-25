#!/bin/bash

# Test Symmetric Alignment Scoring
# This script runs a quick test with a few English Bible versions to verify
# the pipeline works before running the full analysis.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration for testing
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Test with a small subset of English versions
TEST_VERSIONS=(
    "eng-kjv.txt"
    "eng-web.txt"
    "eng-asv.txt"
)

# Paths
ENGLISH_CORPUS_DIR="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/eng/corpus"
DHAO_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/nfa-nfa.txt"
VREF_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/vref.txt"
OUTPUT_DIR="./test_results/symmetric_alignment"
TEST_TEMP_DIR="./test_temp"

log_info "=== Testing Symmetric Alignment Pipeline ==="
echo ""

# Check if test versions exist
log_info "Checking test corpus files..."
missing_files=()
for version in "${TEST_VERSIONS[@]}"; do
    if [[ ! -f "$ENGLISH_CORPUS_DIR/$version" ]]; then
        missing_files+=("$version")
    fi
done

if [[ ${#missing_files[@]} -gt 0 ]]; then
    log_error "Missing test corpus files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

log_success "All test corpus files found"
echo ""

# Create a temporary directory with just the test versions
log_info "Setting up test environment..."
mkdir -p "$TEST_TEMP_DIR/eng_subset"

for version in "${TEST_VERSIONS[@]}"; do
    cp "$ENGLISH_CORPUS_DIR/$version" "$TEST_TEMP_DIR/eng_subset/"
    log_info "  Copied $version"
done

echo ""

# Run the pipeline with the subset
log_info "Running symmetric alignment scoring on test subset..."

python3 scripts/symmetric_alignment_scorer.py \
    --english_corpus_dir "$TEST_TEMP_DIR/eng_subset" \
    --dhao_corpus_path "$DHAO_CORPUS_PATH" \
    --vref_path "$VREF_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --output_file "test_scores.csv" \
    --temp_dir "$TEST_TEMP_DIR/giza_work" \
    --debug \
    --keep_temp

# Check results
if [[ -f "$OUTPUT_DIR/test_scores.csv" ]]; then
    log_success "Test completed successfully!"
    echo ""
    
    log_info "Test results:"
    cat "$OUTPUT_DIR/test_scores.csv" | column -t -s ','
    echo ""
    
    # Count successful results
    result_count=$(tail -n +2 "$OUTPUT_DIR/test_scores.csv" | wc -l)
    log_info "Successfully processed $result_count/${#TEST_VERSIONS[@]} versions"
    
    if [[ $result_count -eq ${#TEST_VERSIONS[@]} ]]; then
        log_success "All test versions processed successfully!"
        log_info "Pipeline is ready for full run. Execute:"
        echo "  ./run_scripts/run_symmetric_alignment.sh"
    else
        log_warning "Some versions failed - check logs and fix issues before full run"
    fi
    
else
    log_error "Test failed - no output file generated"
    exit 1
fi

# Show temp directory for inspection
log_info "Temporary files kept at: $TEST_TEMP_DIR"
log_info "To clean up: rm -rf $TEST_TEMP_DIR"

echo ""
log_success "Test completed!"
