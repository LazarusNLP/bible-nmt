# Symmetric Alignment Scoring for Bible Translations

This system evaluates the alignment quality between English Bible versions and Dhao translation using GIZA++ symmetric alignment scoring.

## Overview

The pipeline performs the following steps for each English Bible version:

1. **Text Normalization**: Normalizes both English and Dhao texts using the existing `normalize_text.py` script with `--preserve-case` option
2. **Verse Alignment**: Creates aligned verse corpora using `create_aligned_nt_ot.py`  
3. **Forward Alignment**: Runs GIZA++ from English to Dhao and extracts log-likelihood score
4. **Reverse Alignment**: Runs GIZA++ from Dhao to English and extracts log-likelihood score
5. **Scoring**: Computes average of forward and reverse scores

## Files Created

### Scripts (`/scripts/`)
- `symmetric_alignment_scorer.py` - Main orchestration script
- `giza_alignment_runner.py` - Helper script for running GIZA++ and extracting scores

### Shell Scripts (`/run_scripts/`)
- `run_symmetric_alignment.sh` - Full pipeline execution
- `test_symmetric_alignment.sh` - Test with subset of versions

## Prerequisites

### Required Tools
- Python 3.6+
- GIZA++ toolkit (GIZA++, plain2snt, mkcls) - already installed in your `bible-nmt` conda environment
- Existing scripts: `normalize_text.py`, `create_aligned_nt_ot.py`

### Input Files
- English corpus files: `/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/eng/corpus/eng-*.txt`
- Dhao corpus: `/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/nfa-nfa.txt`
- Verse references: `/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/vref.txt`

## Usage

### Quick Test (3 versions)
```bash
cd /data/projects/punim0478/setiawand/bible-nmt
conda activate bible-nmt
./run_scripts/test_symmetric_alignment.sh
```

### Full Analysis (all versions)
```bash
cd /data/projects/punim0478/setiawand/bible-nmt
conda activate bible-nmt
./run_scripts/run_symmetric_alignment.sh
```

### Custom Configuration
```bash
# Use custom paths
ENGLISH_CORPUS_DIR=/custom/path/eng \
DHAO_CORPUS_PATH=/custom/path/dhao.txt \
./run_scripts/run_symmetric_alignment.sh

# Debug mode with temp file preservation
./run_scripts/run_symmetric_alignment.sh --debug --keep-temp

# Process limited number of versions
./run_scripts/run_symmetric_alignment.sh --max-versions 10
```

## Output

The main output is a CSV file with columns:
- `english_version`: Name of English Bible version (e.g., "kjv", "web", "asv")
- `target_language`: Always "dhao" 
- `forward_score`: Log-likelihood score for English→Dhao alignment
- `reverse_score`: Log-likelihood score for Dhao→English alignment  
- `average_score`: Average of forward and reverse scores

**Higher scores indicate better alignment quality.**

### Sample Output
```csv
english_version,target_language,forward_score,reverse_score,average_score
kjv,dhao,-156.7234,-159.2341,-157.9788
web,dhao,-154.9876,-158.1234,-156.5555
asv,dhao,-155.4321,-157.8765,-156.6543
```

## Command Line Options

### run_symmetric_alignment.sh
```bash
Usage: ./run_scripts/run_symmetric_alignment.sh [OPTIONS]

OPTIONS:
    -h, --help              Show help message
    -d, --debug             Enable debug mode
    -k, --keep-temp         Keep temporary files for debugging
    -o, --output-dir DIR    Output directory (default: ./results/symmetric_alignment)
    -f, --output-file FILE  Output CSV filename (default: symmetric_alignment_scores.csv)
    -t, --temp-dir DIR      Temporary directory (default: system temp)
    -m, --max-versions N    Maximum number of versions to process
```

### Environment Variables
- `ENGLISH_CORPUS_DIR` - Directory containing English corpus files
- `DHAO_CORPUS_PATH` - Path to Dhao corpus file
- `VREF_PATH` - Path to verse reference file
- `OUTPUT_DIR` - Output directory
- `KEEP_TEMP` - Keep temporary files (true/false)
- `DEBUG` - Enable debug mode (true/false)

## Algorithm Details

### Scoring Methodology
1. **Log-likelihood Extraction**: Extracts final training perplexity from GIZA++ `.perp` files
2. **Score Conversion**: Converts perplexity to log-likelihood (higher = better alignment)
3. **Symmetric Scoring**: Averages forward and reverse alignment scores
4. **Ranking**: Ranks English versions by average alignment score

### GIZA++ Configuration
- IBM Model 1: 5 iterations
- HMM Model: 5 iterations  
- IBM Model 3: 3 iterations
- IBM Model 4: 3 iterations
- No dump files created (space optimization)

## Troubleshooting

### Common Issues

1. **Missing GIZA++ tools**
   ```
   ERROR: Missing required tools: GIZA++, plain2snt, mkcls
   ```
   Solution: Ensure tools are in PATH (already installed in bible-nmt environment)

2. **Memory issues with large corpora**
   - Use `--max-versions` to limit processing
   - Monitor temp directory disk usage
   - Consider processing in batches

3. **Alignment failures**
   - Check input text encoding (should be UTF-8)
   - Verify verse alignment quality
   - Enable debug mode for detailed logs

### Debug Mode
```bash
./run_scripts/run_symmetric_alignment.sh --debug --keep-temp
```
- Enables verbose logging
- Preserves all temporary files
- Shows detailed GIZA++ output

### File Locations
- Main results: `./results/symmetric_alignment/symmetric_alignment_scores.csv`
- Test results: `./test_results/symmetric_alignment/test_scores.csv`
- Temporary files: System temp directory (unless specified)

## Performance Notes

- Processing ~50 English versions takes several hours
- Each version requires ~10-20 minutes (normalization + alignment + 2x GIZA++)
- Disk space: ~1-2GB temporary files per version (cleaned automatically)
- Memory: ~2-4GB RAM recommended

## Expected Results

The system will identify which English Bible translations have the best structural/lexical alignment with the Dhao translation, which can inform:
- Translation quality assessment
- Choice of source text for further translation work
- Understanding of translation consistency patterns

Higher average scores indicate better bidirectional alignment between the English version and Dhao translation.
