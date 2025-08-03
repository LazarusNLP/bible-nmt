# Token Length Analysis Scripts

These scripts analyze the token lengths in your Bible translation datasets using the NLLB tokenizer.

## Scripts Available

### 1. `quick_token_analysis.py` - Fast Analysis ⚡
**Recommended for quick results**

```bash
# Quick analysis (1000 samples per language)
python scripts/quick_token_analysis.py

# Custom sample size for speed vs accuracy trade-off
python scripts/quick_token_analysis.py 2000
```

**Features:**
- ✅ Fast execution (2-5 minutes)
- ✅ Samples data for speed
- ✅ Shows max token lengths
- ✅ Provides recommendations
- ✅ No dependencies on matplotlib

### 2. `analyze_token_lengths.py` - Complete Analysis 📊
**Use for comprehensive analysis**

```bash
# Full analysis of ALL data
python scripts/analyze_token_lengths.py
```

**Features:**
- ✅ Analyzes ALL examples (slower but comprehensive)
- ✅ Creates detailed visualizations
- ✅ Saves results to CSV files
- ✅ Generates plots and charts
- ❗ Requires matplotlib, seaborn

## Required Dependencies

```bash
# Install if not already available
pip install transformers datasets numpy pandas tqdm

# For full analysis only:
pip install matplotlib seaborn
```

## Expected Output

Both scripts will show you:

```
🎯 MAXIMUM TOKEN LENGTHS:
   Source (Indonesian): 245
   Target (all langs):  312
   Overall maximum:     312

💡 RECOMMENDATIONS:
   Recommended max_length: 289 (99th percentile + 10%)
   Conservative max_length: 328 (absolute max + 5%)
```

## What This Tells You

- **Recommended max_length**: Use this for training (handles 99% of examples)
- **Conservative max_length**: Use this if you need to handle ALL examples
- **Language-specific max lengths**: See which languages need more tokens

## Usage in Training

After running the analysis, use the recommended `max_length` in your training config:

```python
# Example for NLLB training
max_length = 289  # From analysis results

training_args = {
    "max_source_length": max_length,
    "max_target_length": max_length,
    # ... other args
}
```

## Files Generated (Full Analysis Only)

- `token_analysis_results/token_length_summary.csv`
- `token_analysis_results/detailed_token_stats.csv`
- `token_analysis_plots/token_length_distributions.png`