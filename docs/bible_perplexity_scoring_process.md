# Bible Perplexity Scoring Process

This document explains the automated pipeline for finding the best English Bible version for translation alignment with target languages (specifically Dhao) based on GIZA++ statistical alignment perplexity scores.

## 🎯 **Purpose**

The goal is to identify which English Bible translation provides the best alignment with a target language translation. Better alignment (lower perplexity) indicates:
- More consistent verse structure
- Better lexical correspondence  
- More reliable word-to-word mappings
- Higher quality parallel corpus for machine translation training

## 📋 **Process Overview**

The pipeline automatically processes multiple English Bible versions against a single target language Bible and ranks them by alignment quality:

```
English Versions → Verse Alignment → Normalization → GIZA++ → Perplexity Scores → CSV Report
```

## 🔄 **Detailed Pipeline Steps**

### 1. **Input Discovery**
- Scans `/ebible-corpus/eng/corpus/` for all English Bible versions (`eng-*.txt`)
- Uses target language file: `/ebible-corpus/dhao-eng/nfa-nfa.txt`
- Uses verse reference file: `/ebible-corpus/vref.txt`

### 2. **Verse Alignment** (`create_aligned_nt_ot.py`)
For each English version:
- Aligns verses between English and target language using verse references
- Handles verse ranges (where multiple verses are combined)
- Creates parallel corpus in CSV format with columns: `verse`, `source_text`, `target_text`
- Skips empty or missing verses

### 3. **Text Normalization** (`normalize_for_giza.py`)
Prepares texts for statistical alignment:
- Converts to lowercase for better alignment
- Removes brackets while preserving content: `[text]` → `text`
- Handles numbers with verbalized forms: `99 (ninety-nine)` → `ninety-nine`
- Removes most punctuation except apostrophes in contractions
- Normalizes quotation marks and whitespace
- Filters very short sentences (< 3 tokens)
- Creates parallel training files: `train.en-{version}` and `train.nfa`

### 4. **GIZA++ Statistical Alignment**
Uses the `giza-py` wrapper to run GIZA++ with symmetric alignment:
```bash
python3 giza.py \
  --source train.en-{version} \
  --target train.nfa \
  --alignments output_dir \
  --sym-heuristic intersection
```

GIZA++ performs:
- IBM Model 1-4 statistical word alignment
- HMM alignment modeling
- Symmetric alignment using intersection heuristic
- Perplexity calculation based on translation probabilities

### 5. **Perplexity Extraction**
- Parses GIZA++ output to extract final perplexity scores
- Lower perplexity = better alignment quality
- Handles various GIZA++ output formats and error cases

### 6. **Results Compilation**
Creates CSV report with:
- `english_version`: Version identifier (e.g., "webp", "kjv", "asv")
- `target_language`: Target language ("dhao")  
- `perplexity_score`: GIZA++ perplexity (lower is better)

## 🚀 **Usage**

### **Quick Start**
```bash
cd /data/projects/punim0478/setiawand/bible-nmt
./run_scripts/run_bible_perplexity_scoring.sh
```

### **Script Configuration**
Edit `/run_scripts/run_bible_perplexity_scoring.sh` to modify parameters:

```bash
python scripts/bible_perplexity_scorer.py \
    --english_corpus_dir "/path/to/english/corpus" \
    --target_corpus_path "/path/to/target.txt" \
    --vref_path "/path/to/vref.txt" \
    --giza_py_path "/path/to/giza-py" \
    --output_dir "./results/perplexity_scoring" \
    --output_file "bible_perplexity_scores.csv" \
    --max_versions 10 \      # Limit versions for testing
    --debug \                # Enable detailed logging
    --keep_temp              # Keep intermediate files
```

## 📊 **Output Interpretation**

### **CSV Results Format**
```csv
english_version,target_language,perplexity_score
webp,dhao,45.2341
kjv,dhao,47.8932
asv,dhao,48.1245
```

### **Perplexity Score Meaning**
- **Lower scores = Better alignment**
- **Typical range**: 30-100+ 
- **Score < 50**: Generally good alignment
- **Score > 80**: Poor alignment, may indicate structural differences

### **Interpretation Guidelines**
1. **Best Version**: Lowest perplexity score
2. **Top Candidates**: Consider top 3-5 versions for ensemble approaches
3. **Outliers**: Very high scores may indicate incompatible verse structures
4. **Confidence**: Large score differences suggest clear preference

## 🔧 **Technical Details**

### **File Structure**
```
bible-nmt/
├── scripts/
│   ├── bible_perplexity_scorer.py    # Main pipeline script
│   ├── create_aligned_nt_ot.py       # Verse alignment
│   └── normalize_for_giza.py         # Text normalization
├── run_scripts/
│   └── run_bible_perplexity_scoring.sh  # Simple runner script
└── results/perplexity_scoring/
    ├── bible_perplexity_scores.csv   # Final results
    └── temp_*/                       # Temporary processing files
```

### **Dependencies**
- **Conda Environment**: `bible-nmt`
- **GIZA++ Tools**: `GIZA++`, `plain2snt`, `mkcls`
- **giza-py**: Python wrapper for GIZA++
- **Python Packages**: `pandas`, `pathlib`, `subprocess`, `csv`, `re`

### **Processing Time**
- **Per version**: ~2-5 minutes depending on corpus size
- **Full pipeline**: 2-4 hours for ~50 English versions
- **Memory usage**: ~1-2GB during GIZA++ alignment

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **"Expected CSV file not found"**
   - Check that `create_aligned_nt_ot.py` completed successfully
   - Verify verse reference file exists and is readable
   - Ensure English and target corpus files have same verse count

2. **"GIZA++ alignment failed"**
   - Verify GIZA++ tools are installed and in PATH
   - Check that `giza-py` directory exists and contains `giza.py`
   - Ensure normalized text files have sufficient content

3. **"Could not extract perplexity"**
   - Check GIZA++ output format
   - Enable debug mode to see full GIZA++ output
   - Verify GIZA++ completed all training iterations

### **Debug Mode**
Run with `--debug --keep_temp` to:
- See detailed processing logs
- Examine intermediate files in temp directory
- Debug alignment and normalization issues

### **Validation Steps**
1. **Check input files exist and are readable**
2. **Verify verse counts match between source and target**
3. **Examine normalized text quality**
4. **Review GIZA++ output for errors**
5. **Validate perplexity extraction patterns**

## 📈 **Expected Results**

### **Typical Best Performers**
- **Modern translations**: ESV, NIV, NASB often align well
- **Formal equivalence**: More literal translations may score better
- **Contemporary language**: Matches modern target language patterns

### **Factors Affecting Scores**
- **Translation philosophy**: Formal vs. dynamic equivalence
- **Verse structure**: Consistent vs. reorganized verses
- **Vocabulary complexity**: Simple vs. archaic language
- **Cultural adaptation**: Literal vs. culturally adapted translations

## 🔄 **Next Steps After Scoring**

1. **Select Best Version**: Use lowest perplexity score
2. **Validate Choice**: Manually inspect a few verse alignments
3. **Create Training Corpus**: Use best version for MT training
4. **Consider Ensemble**: Combine top versions for robustness
5. **Domain Adaptation**: Fine-tune on best-aligned parallel data

## 📝 **Example Workflow**

```bash
# 1. Run perplexity scoring
./run_scripts/run_bible_perplexity_scoring.sh

# 2. Review results
cat results/perplexity_scoring/bible_perplexity_scores.csv

# 3. Select best version (e.g., "webp" with score 45.23)
# 4. Use for downstream translation training:
python src/run_translation.py \
    --source_text_path "ebible-corpus/eng/corpus/eng-webp.txt" \
    --target_text_path "ebible-corpus/dhao-eng/nfa-nfa.txt" \
    # ... other training parameters
```

---

*This documentation covers the complete Bible perplexity scoring process for finding optimal English-target language alignment pairs for machine translation training.*
