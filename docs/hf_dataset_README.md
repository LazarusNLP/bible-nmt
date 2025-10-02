---
language:
- ind
- aaz
- heg
- lex
- llg
- nfa
- ptu
- rgu
- row
- tet
- txq
- wrs
task_categories:
- translation
pretty_name: eBible Indonesian Local Language Corpus
size_categories:
- 10K<n<100K
configs:
- config_name: ind_aaz
  data_files:
  - split: train
    path: ind_aaz/train-*
- config_name: ind_heg
  data_files:
  - split: train
    path: ind_heg/train-*
- config_name: ind_lex
  data_files:
  - split: train
    path: ind_lex/train-*
- config_name: ind_llg
  data_files:
  - split: train
    path: ind_llg/train-*
- config_name: ind_nfa
  data_files:
  - split: train
    path: ind_nfa/train-*
- config_name: ind_ptu
  data_files:
  - split: train
    path: ind_ptu/train-*
- config_name: ind_rgu
  data_files:
  - split: train
    path: ind_rgu/train-*
- config_name: ind_row
  data_files:
  - split: train
    path: ind_row/train-*
- config_name: ind_tet
  data_files:
  - split: train
    path: ind_tet/train-*
- config_name: ind_txq
  data_files:
  - split: train
    path: ind_txq/train-*
- config_name: ind_wrs
  data_files:
  - split: train
    path: ind_wrs/train-*
dataset_info:
  features:
  - name: source_text
    dtype: string
  - name: target_text
    dtype: string
  - name: source_lang
    dtype: string
  - name: target_lang
    dtype: string
  - name: verse
    dtype: string
---

# eBible Indonesian Local Language Corpus

This dataset contains parallel Bible translations between Indonesian and various local languages from Indonesia, particularly from Eastern Indonesia regions.

## Dataset Description

This dataset is created from the eBible corpus, containing verse-aligned translations between Indonesian and multiple local languages. The dataset properly handles verse ranges where multiple verses are combined in the translation.

## Available Language Pairs

The dataset contains the following language pairs as different configurations/subsets:

| Configuration | Source | Target | Language Name | Examples | Description |
|--------------|--------|--------|---------------|----------|-------------|
| `ind_aaz` | ind | aaz | Amarasi | 9,068 | Amarasi Bible translation |
| `ind_heg` | ind | heg | Helong | 9,073 | Helong Bible translation |
| `ind_lex` | ind | lex | Luang | 9,674 | Luang Bible translation |
| `ind_llg` | ind | llg | Lole | 9,071 | Lole Bible translation |
| `ind_nfa` | ind | nfa | Dhao | 9,070 | Dhao Bible translation |
| `ind_ptu` | ind | ptu | Bambam | 9,345 | Bambam Bible translation |
| `ind_rgu` | ind | rgu | Rikou | 8,223 | Rikou Bible translation |
| `ind_row` | ind | row | Dela-Oenale | 9,070 | Dela-Oenale Bible translation |
| `ind_tet` | ind | tet | Tetun | 9,069 | Tetun Bible translation |
| `ind_txq` | ind | txq | Tii | 9,069 | Tii Bible translation |
| `ind_wrs` | ind | wrs | Waris | 8,981 | Waris Bible translation |

**Total examples across all configurations:** 99,713

## Dataset Structure

Each configuration contains a 'train' split with the following features:

- `source_text`: Text in the source language (Indonesian)
- `target_text`: Text in the target language
- `source_lang`: Source language code (always 'ind')
- `target_lang`: Target language code  
- `verse`: Bible verse reference (e.g., "GEN 1:1" or "GEN 1:14-15" for verse ranges)

## Usage

### Loading a specific language pair

```python
from datasets import load_dataset

# Load a specific language pair (configuration)
dataset = load_dataset("Davidsamuel101/ebible_local_ind_corpus", "ind_aaz")
train_data = dataset['train']

# Example: access first example
print(train_data[0])
```

### Loading multiple configurations

```python
from datasets import load_dataset

# Load specific configurations
target_langs = ["aaz", "ptu", "nfa", "heg", "lex", "row", "llg", "rgu", "txq", "tet", "wrs"]
for lang in target_langs:
    dataset = load_dataset("Davidsamuel101/ebible_local_ind_corpus", f"ind_{lang}")
    print(f"{lang}: {len(dataset['train'])} examples")
```

### Using for machine translation

```python
from datasets import load_dataset
from transformers import pipeline

# Load Indonesian to Tetun dataset
dataset = load_dataset("Davidsamuel101/ebible_local_ind_corpus", "ind_tet")['train']

# Example: prepare for training
def preprocess_function(examples):
    inputs = [ex for ex in examples["source_text"]]
    targets = [ex for ex in examples["target_text"]]
    return {"translation": [{"ind": i, "tet": t} for i, t in zip(inputs, targets)]}

# Apply preprocessing
processed_dataset = dataset.map(preprocess_function, batched=True)
```

## Data Processing Notes

The dataset was created with special handling for:
- **Verse ranges**: Marked with `<range>` tokens in the original corpus
- **Empty verses**: Automatic skipping of verses where either source or target text is blank
- **Verse concatenation**: Proper concatenation of source verses when targets combine multiple verses

## Languages

### Source Language
- **Indonesian** (`ind`): Bahasa Indonesia, the national language of Indonesia

### Target Languages
- **Amarasi** (`aaz`): Uab Meto language spoken in West Timor
- **Helong** (`heg`): Austronesian language of West Timor
- **Luang** (`lex`): Language spoken in the Luang islands
- **Lole** (`llg`): Language of Central Rote
- **Dhao** (`nfa`): Language of Ndao island
- **Bambam** (`ptu`): Language of Central Sulawesi
- **Rikou** (`rgu`): Language of Central Rote
- **Dela-Oenale** (`row`): Language of West Rote
- **Tetun** (`tet`): Language of East Timor
- **Tii** (`txq`): Language of West Rote
- **Waris** (`wrs`): Papuan language

## Citation

If you use this dataset, please cite:

```bibtex
@misc{ebible_indonesian_2024,
  title={eBible Indonesian Local Language Corpus},
  author={Dataset created from eBible corpus},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/Davidsamuel101/ebible_local_ind_corpus}
}
```

## License

Please refer to the original eBible corpus license terms. The texts are from Bible translations which may have their own usage terms.

## Acknowledgments

This dataset is derived from the eBible corpus. We acknowledge the translators and communities who created these Bible translations in local languages.
