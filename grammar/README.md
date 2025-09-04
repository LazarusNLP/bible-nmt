# Luang Grammar Book Text Processing Pipeline

This document outlines the steps used to clean and process the Luang grammar book text from its original PDF format to a clean text file.

## Processing Steps

### 1. PDF to LaTeX Conversion
- **Source**: Luang Grammar and Phonology Sketch (PDF format)
- **Tool**: Mathpix
- **Action**: Convert the PDF document to LaTeX format
- **Output**: `2025_09_03_818715438e96f16321bbg.tex`

### 2. LaTeX to Plain Text Conversion
- **Tool**: `opendetex` library
- **Command**: 
  ```bash
  detex -n 2025_09_03_818715438e96f16321bbg.tex > luang_grammarbook.txt
  ```
- **Output**: `luang_grammarbook.txt` (raw text with formatting artifacts)

### 3. Text Cleaning and Preprocessing
- **Tool**: `nlpretext` library
- **Installation**: Install nlpretext for text cleaning capabilities
- **Command**:
  ```bash
  nlpretext preprocess run --input luang_grammar_book.txt --output luang_clean_grammarbook
  ```
- **Output**: `luang_clean_grammarbook.parquet` (cleaned text in Parquet format)

### 4. Parquet to Text Conversion
- **Method**: Python script using pandas
- **Script**: `convert_parquet_into_text.ipynb`
- **Code**:
  ```python
  from pathlib import Path
  import pandas as pd

  df = pd.read_parquet("luang_clean_grammarbook.parquet")
  Path("luang_clean_grammarbook.txt").write_text('\n'.join(z for z in df.values.squeeze() if z != ''))
  ```
- **Final Output**: `luang_clean_grammarbook.txt` (clean, processed text file)

## Files in this Directory

- `luang_grammarbook.txt` - Raw text extracted from LaTeX
- `luang_clean_grammarbook.parquet` - Cleaned text in Parquet format
- `luang_clean_grammarbook.txt` - Final cleaned text file
- `convert_parquet_into_text.ipynb` - Notebook for Parquet to text conversion

## Tools Required

- **Mathpix**: For PDF to LaTeX conversion
- **opendetex**: For LaTeX to plain text conversion (`detex` command)
- **nlpretext**: For text preprocessing and cleaning
- **pandas**: For handling Parquet files
