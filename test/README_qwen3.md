# All-MPNet Embedding Vectorizer for Post-Editing

This directory contains test scripts and documentation for the All-MPNet embedding-based vectorizer integration in the Bible NMT post-editing pipeline.

## Overview

The `all_mpnet` vectorizer uses the enhanced **SBERTRetriever** with **sentence-transformers/all-mpnet-base-v2** model to compute semantic embeddings for few-shot parallel sentence retrieval. This provides more accurate semantic similarity matching compared to traditional keyword-based methods like BM25, while leveraging the existing caching infrastructure for optimal performance.

## Features

- **Semantic Similarity**: Uses state-of-the-art all-mpnet-base-v2 model for better semantic understanding
- **Efficient Caching**: Leverages existing SBERT caching infrastructure for optimal performance
- **Embedding Caching**: Caches corpus embeddings to avoid recomputation across runs
- **Model Flexibility**: Easy to switch between different sentence-transformer models
- **Seamless Integration**: Fully integrated with existing post-editing pipeline

## Requirements

- **sentence-transformers** (for embedding models)
- **PyTorch** (for tensor operations)
- **CUDA-compatible GPU** (recommended for performance)
- **bible-nmt conda environment**

## Installation

Ensure your bible-nmt environment has the required dependencies:

```bash
# sentence-transformers should already be installed in bible-nmt environment
# If needed, install it:
conda activate bible-nmt
pip install sentence-transformers

# Or install with conda:
conda install -c conda-forge sentence-transformers
```

## Test Scripts

### 1. `test_qwen3_embedding_retrieval.py`

**Comprehensive test suite for All-MPNet embedding functionality via enhanced SBERTRetriever**

This test suite validates:
- Basic All-MPNet embedding functionality via SBERTRetriever
- Enhanced SBERTRetriever with sentence-transformers/all-mpnet-base-v2 model
- Integration with FewShotSelector using all_mpnet vectorizer
- Embedding caching mechanism with model-specific caching

```bash
cd bible-nmt
python test/test_qwen3_embedding_retrieval.py
```

### 2. `qwen3_usage_example.py`

Simple demonstration of how to use the Qwen3 vectorizer:

```bash
cd bible-nmt  
python test/qwen3_usage_example.py
```

## Usage in Post-Editing Pipeline

To use the All-MPNet vectorizer in the main post-editing pipeline:

```bash
python src/post-editing/run_post_editing.py \
  --vectorizer all_mpnet \
  --model_type vllm \
  --model_name your_translation_model \
  --csv_path input_data.csv \
  --few_shot_corpus_path parallel_corpus.csv \
  --src en --tgt id \
  --src_lang_name English \
  --tgt_lang_name Indonesian \
  --output_dir ./results
```

## How It Works

### 1. Embedding Computation

- **Queries**: Formatted with task instructions: `"Instruct: {task_description}\nQuery: {query}"`
- **Documents**: Used as-is without instruction formatting
- **Model**: Qwen/Qwen3-Embedding-0.6B via vLLM with `task="embed"`

### 2. Similarity Calculation

- Uses dot product similarity: `corpus_embeddings @ query_embedding.T`
- Returns top-k most similar parallel sentence pairs
- Maintains original corpus order for deterministic results

### 3. Caching Strategy

- Corpus embeddings are computed once and cached by `corpus_id`
- Query embeddings are computed fresh for each request
- Cache can be cleared with `retriever.clear_cache()`

## Configuration Options

When initializing `Qwen3VLLMRetriever`:

```python
retriever = Qwen3VLLMRetriever(
    corpus_id="my_corpus_id",  # For caching (optional)
    task_description="Custom task description"  # Default: semantic similarity
)
```

## Performance Notes

- **First Run**: Will download and load the Qwen3-Embedding-0.6B model (~600MB)
- **Corpus Processing**: Computes embeddings for entire corpus on first use
- **Subsequent Runs**: Uses cached embeddings for same `corpus_id`
- **GPU Memory**: Requires ~2-4GB GPU memory depending on corpus size

## Troubleshooting

### Common Issues

1. **CUDA Errors**: Set `CUDA_VISIBLE_DEVICES=0` if you have multiple GPUs
2. **vLLM Version**: Ensure vLLM >= 0.8.5 for embedding task support  
3. **Memory Issues**: Reduce corpus size or use CPU inference if GPU memory insufficient
4. **Import Errors**: Ensure you're running from correct directory with proper Python path

### Debug Mode

Run with environment variables for detailed debugging:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DISABLE_XFORMERS=1
export FLASH_ATTENTION_SKIP_CUDA_BUILD=1
export ENFORCE_EAGER=1
```

## Comparison with Other Vectorizers

| Vectorizer | Type | Speed | Quality | GPU Required |
|------------|------|--------|---------|--------------|
| BM25 | Keyword | Fast | Good | No |
| TF-IDF | Keyword | Fast | Good | No |  
| SBERT | Embedding | Medium | Very Good | Optional |
| **Qwen3-vLLM** | **Embedding** | **Medium** | **Excellent** | **Recommended** |

## Contributing

When modifying the Qwen3 integration:

1. Update test scripts to cover new functionality
2. Ensure backward compatibility with existing pipeline
3. Test with different corpus sizes and languages
4. Update this documentation

## References

- [Qwen3-Embedding Model](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Post-Editing Pipeline Documentation](../src/post-editing/README.md)

