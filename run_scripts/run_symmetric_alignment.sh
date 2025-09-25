#!/bin/bash

# Symmetric Alignment Scoring Pipeline
# This script runs the complete pipeline to score alignment quality between
# English Bible versions and Dhao translation using GIZA++ symmetric alignment.

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default paths (can be overridden by environment variables)
ENGLISH_CORPUS_DIR="${ENGLISH_CORPUS_DIR:-/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/eng/corpus}"
DHAO_CORPUS_PATH="${DHAO_CORPUS_PATH:-/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/nfa-nfa.txt}"
VREF_PATH="${VREF_PATH:-/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/vref.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/symmetric_alignment}"
TEMP_DIR="${TEMP_DIR:-}"  # Empty means use system default
OUTPUT_FILE="${OUTPUT_FILE:-symmetric_alignment_scores.csv}"

# Options
KEEP_TEMP="${KEEP_TEMP:-false}"
DEBUG="${DEBUG:-false}"
MAX_VERSIONS="${MAX_VERSIONS:-}"  # Empty means process all versions

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Function to check if required tools are available
check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing_tools=()
    
    # Check for Python
    if ! command -v python3 &> /dev/null; then
        missing_tools+=("python3")
    fi
    
    # Check for GIZA++ tools
    if ! command -v GIZA++ &> /dev/null; then
        missing_tools+=("GIZA++")
    fi
    
    if ! command -v plain2snt &> /dev/null; then
        missing_tools+=("plain2snt")
    fi
    
    if ! command -v mkcls &> /dev/null; then
        missing_tools+=("mkcls")
    fi
    
    # Check for required Python scripts
    local required_scripts=("scripts/symmetric_alignment_scorer.py" "scripts/normalize_text.py" "scripts/create_aligned_nt_ot.py")
    
    for script in "${required_scripts[@]}"; do
        if [[ ! -f "$script" ]]; then
            missing_tools+=("$script")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools/scripts:"
        for tool in "${missing_tools[@]}"; do
            echo "  - $tool"
        done
        
        log_info "To install GIZA++ tools, make sure they are compiled and in your PATH."
        log_info "Required Python packages: pandas, numpy, scikit-learn"
        exit 1
    fi
    
    log_success "All dependencies are available"
}

# Function to check if input files exist
check_input_files() {
    log_info "Checking input files..."
    
    if [[ ! -d "$ENGLISH_CORPUS_DIR" ]]; then
        log_error "English corpus directory not found: $ENGLISH_CORPUS_DIR"
        exit 1
    fi
    
    if [[ ! -f "$DHAO_CORPUS_PATH" ]]; then
        log_error "Dhao corpus file not found: $DHAO_CORPUS_PATH"
        exit 1
    fi
    
    if [[ ! -f "$VREF_PATH" ]]; then
        log_error "Verse reference file not found: $VREF_PATH"
        exit 1
    fi
    
    # Count English corpus files
    local eng_count=$(find "$ENGLISH_CORPUS_DIR" -name "eng-*.txt" | wc -l)
    log_info "Found $eng_count English corpus files"
    
    if [[ $eng_count -eq 0 ]]; then
        log_error "No English corpus files found in $ENGLISH_CORPUS_DIR"
        exit 1
    fi
    
    log_success "All input files are available"
}

# Function to display configuration
show_config() {
    log_info "Configuration:"
    echo "  English corpus directory: $ENGLISH_CORPUS_DIR"
    echo "  Dhao corpus file: $DHAO_CORPUS_PATH"
    echo "  Verse reference file: $VREF_PATH"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Output file: $OUTPUT_FILE"
    echo "  Temporary directory: ${TEMP_DIR:-<system default>}"
    echo "  Keep temporary files: $KEEP_TEMP"
    echo "  Debug mode: $DEBUG"
    echo "  Max versions to process: ${MAX_VERSIONS:-<all>}"
}

# Function to run the symmetric alignment scorer
run_scorer() {
    log_info "Starting symmetric alignment scoring..."
    
    # Build command
    local cmd_args=(
        "python3" "scripts/symmetric_alignment_scorer.py"
        "--english_corpus_dir" "$ENGLISH_CORPUS_DIR"
        "--dhao_corpus_path" "$DHAO_CORPUS_PATH"
        "--vref_path" "$VREF_PATH"
        "--output_dir" "$OUTPUT_DIR"
        "--output_file" "$OUTPUT_FILE"
    )
    
    # Add optional arguments
    if [[ -n "$TEMP_DIR" ]]; then
        cmd_args+=("--temp_dir" "$TEMP_DIR")
    fi
    
    if [[ "$KEEP_TEMP" == "true" ]]; then
        cmd_args+=("--keep_temp")
    fi
    
    if [[ "$DEBUG" == "true" ]]; then
        cmd_args+=("--debug")
    fi
    
    # Run the scorer
    log_info "Executing: ${cmd_args[*]}"
    
    if "${cmd_args[@]}"; then
        log_success "Symmetric alignment scoring completed successfully!"
        
        # Display results if output file exists
        local output_path="$OUTPUT_DIR/$OUTPUT_FILE"
        if [[ -f "$output_path" ]]; then
            log_info "Results saved to: $output_path"
            
            # Show top 5 results if CSV has content
            if [[ $(wc -l < "$output_path") -gt 1 ]]; then
                log_info "Top 5 best aligned English versions:"
                echo ""
                head -6 "$output_path" | column -t -s ',' | while IFS= read -r line; do
                    echo "  $line"
                done
                echo ""
            fi
        fi
        
    else
        log_error "Symmetric alignment scoring failed!"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run symmetric alignment scoring between English Bible versions and Dhao.

OPTIONS:
    -h, --help              Show this help message
    -d, --debug             Enable debug mode
    -k, --keep-temp         Keep temporary files for debugging
    -o, --output-dir DIR    Output directory (default: ./results/symmetric_alignment)
    -f, --output-file FILE  Output CSV filename (default: symmetric_alignment_scores.csv)
    -t, --temp-dir DIR      Temporary directory (default: system temp)
    -m, --max-versions N    Maximum number of versions to process (for testing)

ENVIRONMENT VARIABLES:
    ENGLISH_CORPUS_DIR      Directory containing English corpus files
    DHAO_CORPUS_PATH        Path to Dhao corpus file
    VREF_PATH               Path to verse reference file
    OUTPUT_DIR              Output directory
    TEMP_DIR                Temporary directory
    OUTPUT_FILE             Output CSV filename
    KEEP_TEMP               Keep temporary files (true/false)
    DEBUG                   Enable debug mode (true/false)
    MAX_VERSIONS            Maximum number of versions to process

EXAMPLES:
    # Run with default settings
    $0
    
    # Run in debug mode and keep temp files
    $0 --debug --keep-temp
    
    # Run with custom output location
    $0 --output-dir ./my_results --output-file my_scores.csv
    
    # Process only first 5 versions for testing
    MAX_VERSIONS=5 $0 --debug
    
    # Use custom corpus locations
    ENGLISH_CORPUS_DIR=/path/to/eng/corpus \\
    DHAO_CORPUS_PATH=/path/to/dhao.txt \\
    $0

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -d|--debug)
            DEBUG="true"
            shift
            ;;
        -k|--keep-temp)
            KEEP_TEMP="true"
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -f|--output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -t|--temp-dir)
            TEMP_DIR="$2"
            shift 2
            ;;
        -m|--max-versions)
            MAX_VERSIONS="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log_info "=== Symmetric Alignment Scoring Pipeline ==="
    echo ""
    
    show_config
    echo ""
    
    check_dependencies
    echo ""
    
    check_input_files
    echo ""
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Run the scoring pipeline
    run_scorer
    
    log_success "Pipeline completed successfully!"
    
    # Final summary
    echo ""
    log_info "=== SUMMARY ==="
    local output_path="$OUTPUT_DIR/$OUTPUT_FILE"
    if [[ -f "$output_path" ]]; then
        local result_count=$(tail -n +2 "$output_path" | wc -l)
        log_success "Successfully processed $result_count Bible versions"
        log_info "Results available at: $output_path"
    else
        log_warning "Output file not found - check logs for errors"
    fi
}

# Run main function
main "$@"
