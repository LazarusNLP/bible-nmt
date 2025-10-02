# Post-Editing Pipeline

A modular and unified post-editing system for machine translation that supports multiple model types and retrieval methods.

## Features

- **Multiple Model Support**: GPT, Gemini, and vLLM models
- **Retrieval Methods**: BM25, TF-IDF, SBERT, CHRF-RAG, and Word-based LCS for few-shot example selection
- **Glossary/Lexicon Support**: Terminological assistance using glossary files with POS tags
- **Flexible Few-shot Modes**: Parallel examples, glossary-based, or combined approaches
- **Efficient Processing**: Batch processing, caching, and resumption capabilities
- **Comprehensive Evaluation**: BLEU, SacreBLEU, SPBLEU, chrF variants
- **Modular Design**: Clean separation of concerns with reusable components

## Quick Start

### Basic Usage

```bash
# Using vLLM with a local model
python run_post_editing.py --model_type vllm --model_name microsoft/DialoGPT-medium \
  --csv_path data.csv --src en --tgt id \
  --src_lang_name English --tgt_lang_name Indonesian \
  --output_dir ./results

# Using GPT API with few-shot examples
python run_post_editing.py --model_type gpt --model_name gpt-4o \
  --csv_path data.csv --few_shot_corpus_path corpus.csv --vectorizer sbert \
  --src en --tgt id --src_lang_name English --tgt_lang_name Indonesian \
  --output_dir ./results

# Using Gemini API with async batch processing (recommended for speed)
python run_post_editing.py --model_type gemini --model_name gemini-2.5-flash \
  --csv_path data.csv \
  --src en --tgt id --src_lang_name English --tgt_lang_name Indonesian \
  --output_dir ./results

# Using Gemini API with custom batch size
python run_post_editing.py --model_type gemini --model_name gemini-2.5-flash \
  --csv_path data.csv --batch_size 50 \
  --src en --tgt id --src_lang_name English --tgt_lang_name Indonesian \
  --output_dir ./results

# Using glossary-only mode for terminological consistency
python run_post_editing.py --model_type vllm --model_name microsoft/DialoGPT-medium \
  --csv_path data.csv --glossary_path glossary.csv \
  --few_shot_mode glossary --max_glossary_entries 10 \
  --src en --tgt id --src_lang_name English --tgt_lang_name Indonesian \
  --output_dir ./results

# Using word-fuzzy glossary mode for per-word fuzzy matching
python run_post_editing.py --model_type vllm --model_name microsoft/DialoGPT-medium \
  --csv_path data.csv --glossary_path glossary.csv \
  --few_shot_mode glossary --glossary_mode word_fuzzy --top_n_per_word_glossary 5 \
  --src en --tgt id --src_lang_name English --tgt_lang_name Indonesian \
  --output_dir ./results

# Using combined parallel examples + glossary
python run_post_editing.py --model_type vllm --model_name microsoft/DialoGPT-medium \
  --csv_path data.csv --few_shot_corpus_path corpus.csv --glossary_path glossary.csv \
  --few_shot_mode both --num_few_shot 3 --max_glossary_entries 5 \
  --src en --tgt id --src_lang_name English --tgt_lang_name Indonesian \
  --output_dir ./results

# Using LLM-only mode (no few-shot examples or glossary)
python run_post_editing.py --model_type gemini --model_name gemini-2.5-flash \
  --csv_path data.csv --few_shot_mode none \
  --src en --tgt id --src_lang_name English --tgt_lang_name Indonesian \
  --output_dir ./results

# Using translation mode with few-shot examples
python run_post_editing.py --model_type gemini --model_name gemini-2.5-flash \
  --csv_path data.csv --few_shot_corpus_path corpus.csv --glossary_path glossary.csv \
  --few_shot_mode both --translation-mode \
  --src en --tgt id --src_lang_name English --tgt_lang_name Indonesian \
  --output_dir ./results

# Using sequential processing (no concurrency)
python run_post_editing.py --model_type gemini --model_name gemini-2.5-flash \
  --csv_path data.csv --sequential \
  --src en --tgt id --src_lang_name English --tgt_lang_name Indonesian \
  --output_dir ./results
```

### Environment Variables

Set the following environment variables for API-based models:

```bash
export OPENAI_API_KEY="your-openai-api-key"  # For GPT models
export GEMINI_API_KEY="your-gemini-api-key"  # For Gemini models
```

### Dependencies for Enhanced Glossary Matching

For advanced glossary matching with lemmatization support:

```bash
pip install spacy
```

**Note**: SpaCy is optional. Without it, the system will use basic string matching only.

## Module Structure

```
post-editing/
├── core/               # Core data models and utilities
│   ├── data_models.py  # Row class and data structures
│   ├── data_io.py      # File I/O operations
│   ├── metrics.py      # Evaluation metrics calculation
│   └── constants.py    # Shared constants and schemas
├── retrieval/          # Few-shot retrieval systems
│   ├── base.py         # Abstract retriever interface
│   ├── vectorizers.py  # BM25, TF-IDF, SBERT implementations
│   ├── caching.py      # SBERT model and embedding caching
│   └── few_shot.py     # Few-shot selection utilities
├── models/             # Model interfaces
│   ├── base.py         # Abstract LLM interface
│   ├── api_models.py   # GPT and Gemini implementations
│   ├── vllm_models.py  # vLLM implementation
│   └── tokenization.py # Token counting utilities
├── utils/              # Processing utilities
│   ├── processing.py   # Main processing engine
│   ├── validation.py   # Input validation
│   └── logging.py      # Logging utilities
├── config.py           # Configuration management
├── prompt.py           # Prompt templates
└── run_post_editing.py # Main unified script
```

## Arguments

### Required Arguments

- `--model_type`: Model type (`gpt`, `gemini`, `vllm`)
- `--model_name`: Model name or path
- `--csv_path`: Path to input CSV file
- `--src`: Source language code (e.g., `en`)
- `--tgt`: Target language code (e.g., `id`)
- `--src_lang_name`: Source language name (e.g., `English`)
- `--tgt_lang_name`: Target language name (e.g., `Indonesian`)
- `--output_dir`: Output directory

### Optional Arguments

- `--few_shot_corpus_path`: Path to few-shot corpus
- `--glossary_path`: Path to glossary file (CSV with source_word, target_word, pos columns)
- `--few_shot_mode`: Few-shot mode (`parallel`, `glossary`, `both`, `none`) (default: `parallel`)
- `--max_samples`: Maximum number of samples to process
- `--prompt`: Prompt template key (default: `default`)
- `--num_few_shot`: Number of few-shot examples (default: 5)
- `--max_glossary_entries`: Maximum glossary entries per input (default: all available)
- `--vectorizer`: Similarity method for parallel mode. Options: `bm25`, `tfidf`, `sbert`, `all_mpnet`, `chrf_rag`, `word_parallel`, `full`
- `--glossary_mode`: Glossary selection mode. Options: `smart` (default), `full`, `word_fuzzy`
- `--top_n_per_word_glossary`: For word_fuzzy glossary mode, number of top glossary matches per word (default: 3)
- `--batch_size`: Batch size for processing
- `--num_workers`: Number of worker threads (default: 8)
- `--batch_timeout`: Timeout for batch jobs in seconds
- `--sequential`: Process requests sequentially without concurrency (disables batch processing and threading)
- `--debug`: Enable debug mode
- `--log_file`: Optional log file path

## Input Format

### CSV File
The input CSV must contain these columns:
- `source_text`: Source language text
- `target_text`: Ground truth target language text
- `pred_text`: Machine translation prediction

### Few-shot Corpus
Can be either:
1. **CSV file** with `source_text` and `target_text` columns
2. **Text file** with one sentence per line (used for both source and target)

### Glossary File
**CSV file** with the following columns:
- `source_word`: Word or phrase in source language
- `target_word`: Translation in target language  
- `pos`: Part-of-speech tag (optional, e.g., NOUN, VERB, ADJ)

Example:
```csv
source_word,target_word,pos
love,kasih,NOUN
to love,mengasihi,VERB
beautiful,indah,ADJ
quickly,dengan cepat,ADV
itu (jamak),nala,PRON
cabut (rambut),hau,VERB
```

**Note**: Entries with parentheses (like `'itu (jamak)'`) will match input words without parentheses (`'itu'`), while preserving the contextual information for the LLM.

## Output

- **Results CSV**: Same format as input with added `post_edited_tgt_txt` column
- **Metrics JSON**: Comprehensive evaluation metrics including improvements
- **Logs**: Processing logs and statistics

## Key Features

### 1. Model Flexibility
- **API Models**: GPT and Gemini with async batch processing support
- **Local Models**: vLLM for high-performance local inference
- **Unified Interface**: Same API across all model types
- **Async Processing**: Gemini models support native async batch processing for maximum throughput

### 2. Few-shot Modes

The system supports three few-shot approaches:

#### **Parallel Mode** (`--few_shot_mode parallel`)
Uses similar translation examples from parallel corpus. **Requires**: `--few_shot_corpus_path`

**Available vectorizers** (`--vectorizer`):
- **`bm25`**: Fast keyword-based similarity using Okapi BM25 algorithm
- **`tfidf`**: Term frequency-inverse document frequency vectorization with cosine similarity  
- **`sbert`**: Semantic similarity with embedding caching (uses Indonesian-optimized `LazarusNLP/all-indo-e5-small-v4`)
- **`all_mpnet`**: High-quality multilingual embeddings (uses `sentence-transformers/all-mpnet-base-v2`)
- **`chrf_rag`**: Character n-gram similarity with diversity-aware selection to avoid redundancy
- **`word_parallel`**: Retrieves top-n matches per word in the query with configurable hyperparameters
- **`full`**: Returns the entire corpus as few-shot examples (useful for small corpora)

**Vectorizer Selection Guide:**
- **Indonesian tasks**: `sbert` (Indonesian-optimized, faster)
- **Multilingual/high-quality**: `all_mpnet` (state-of-the-art multilingual)
- **Large corpora**: `bm25` or `tfidf` (keyword-based, very fast)
- **Character-level**: `chrf_rag` (good for morphologically rich languages)
- **Fuzzy matching**: `word_parallel` (handles typos and variations)

#### **Glossary Mode** (`--few_shot_mode glossary`)
Uses relevant terminology from glossary/lexicon. **Requires**: `--glossary_path`

**Available matching modes** (`--glossary_mode`):
- **`smart`** (default): Intelligently selects most relevant glossary entries based on input text matching
- **`full`**: Returns all available glossary entries (up to `--max_glossary_entries` limit)
- **`word_fuzzy`**: Fuzzy matching per word - finds top-N glossary entries for each word in the input text using similarity matching

#### **Both Mode** (`--few_shot_mode both`)  
Combines parallel examples with glossary entries for comprehensive assistance. **Requires**: Both `--few_shot_corpus_path` and `--glossary_path`
- Uses the specified `--vectorizer` for parallel examples
- Uses the specified `--glossary_mode` for terminology

#### **None Mode** (`--few_shot_mode none`)  
Uses LLM capabilities only without any few-shot examples or glossary. **Requires**: No additional resources
- Pure LLM-based translation or post-editing
- Fastest processing (no retrieval overhead)
- Useful for baseline comparisons

#### Glossary Matching Features:
- **Optimized Lookup**: Fast dictionary-based matching (O(n) vs O(n×m))
- **Multi-Strategy Matching**: Exact → Normalized → Lemmatized (in priority order)
- **Parenthetical Handling**: Matches base words in entries like `'itu (jamak)'` when input contains `'itu'`
- **Normalization**: Handles dashes and spacing variations (`'self-driving'` ↔ `'self driving'`)
- **Lemmatization**: Automatic lemmatization for inflected forms (requires spaCy)
- **No Duplicate Removal**: Returns all matching entries (preserves multiple senses)
- **Debug Mode**: Detailed matching statistics with `--debug` flag

### 3. Efficient Processing
- **Async Batch Processing**: Native async support for Gemini API with up to 50 concurrent requests
- **Smart Batch Sizing**: Automatic optimization for different model types (e.g., 200 requests for Gemini)
- **Rate Limiting**: Built-in rate limiting to respect API limits (500 RPM for Gemini)
- **Progress Tracking**: Real-time progress bars for async operations
- **Resumption**: Automatically resume from where processing left off
- **Caching**: SBERT embeddings cached for reuse across runs
- **Multi-threaded Fallback**: Thread-based parallel processing for non-async models

### 4. Comprehensive Evaluation
- **Multiple Metrics**: BLEU, SacreBLEU, SPBLEU, chrF, chrF3, chrF++
- **Improvement Tracking**: Shows gains from post-editing
- **Statistical Analysis**: Token counts and processing statistics

## Development

### Adding New Models
1. Inherit from `BaseLLM` in `models/base.py`
2. Implement `generate()` and optionally `generate_batch()`
3. Add model creation logic to `run_post_editing.py`

### Adding New Retrievers
1. Inherit from `BaseRetriever` in `retrieval/base.py`
2. Implement `get_similar_examples()`
3. Add to `FewShotSelector` in `retrieval/few_shot.py`

### Testing
```bash
# Test with debug mode
python run_post_editing.py --debug --max_samples 10 [other args]

# Test different components
python -m pytest tests/  # If you add tests
```

## Migration from Legacy Scripts

The modular design maintains compatibility with existing workflows:

1. **Same Arguments**: Most arguments remain unchanged
2. **Same Output Format**: CSV and metrics files use same format
3. **Same Performance**: Equivalent or better performance
4. **Additional Features**: More model types and processing options

## Performance Tips

1. **Use Gemini for Maximum Speed**: Gemini 2.5 Flash with async batch processing provides the fastest throughput
2. **Let Auto-Batch Sizing Work**: Don't set `--batch_size` for Gemini - it will automatically use optimal batch sizes
3. **Use Batch Processing**: Set `--batch_size` for large datasets with non-async models
4. **Cache SBERT**: Reuse same `corpus_id` across runs for caching
5. **Optimize Workers**: Adjust `--num_workers` based on API rate limits (irrelevant for Gemini async)
6. **Resume Processing**: Interrupted runs automatically resume
7. **Use vLLM**: For local models, vLLM provides best performance

### Gemini Performance Expectations

- **Async Processing**: Up to 50 concurrent requests with built-in rate limiting
- **Speed**: 8.33 requests per second (500 RPM) conservative limit
- **Throughput**: Process 200+ translations in ~24 seconds instead of 10+ minutes sequential
- **Reliability**: Automatic retry and error handling for failed requests

## Troubleshooting

### Common Issues

1. **CUDA Errors**: Set `CUDA_VISIBLE_DEVICES=0` for vLLM
2. **API Rate Limits**: Reduce `--num_workers` or increase `--batch_size`
3. **Memory Issues**: Use smaller `--batch_size` or `--max_samples`
4. **Import Errors**: Check if all required packages are installed

### Debug Mode
Use `--debug` flag for detailed logging and message inspection.

**Debug output includes:**
- Complete prompt content for first sample
- Glossary matching details (direct matches, lemmatization results)
- Word extraction and matching statistics
- Sample glossary entries found
