#!/usr/bin/env python3
"""
Script to push the combined eBible dataset to HuggingFace Hub.
This approach creates a single dataset with all language pairs as different configurations.
"""

import os
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict, load_from_disk
from huggingface_hub import HfApi, login
import json


def create_combined_dataset():
    """Create a combined dataset with all language pairs as different configurations."""
    
    # Language pair configurations
    language_configs = {
        "ind_aaz": "Indonesian to Amarasi (Uab Meto) Bible translation",
        "ind_heg": "Indonesian to Helong Bible translation", 
        "ind_lex": "Indonesian to Luang Bible translation",
        "ind_llg": "Indonesian to Lole Bible translation",
        "ind_nfa": "Indonesian to Dhao Bible translation",
        "ind_ptu": "Indonesian to Bambam Bible translation",
        "ind_rgu": "Indonesian to Rikou Bible translation",
        "ind_row": "Indonesian to Dela-Oenale Bible translation",
        "ind_tet": "Indonesian to Tetun Bible translation",
        "ind_txq": "Indonesian to Tii Bible translation",
        "ind_wrs": "Indonesian to Waris Bible translation"
    }
    
    combined_data = {}
    
    print("Loading all language pair datasets...")
    for config_name in language_configs.keys():
        # Use absolute path
        data_dir = Path(f"/data/projects/punim0478/setiawand/bible-nmt/data/{config_name}")
        if data_dir.exists():
            try:
                dataset = load_from_disk(str(data_dir))
                if 'train' in dataset:
                    combined_data[config_name] = dataset['train']
                    print(f"  ✓ Loaded {config_name}: {len(dataset['train'])} examples")
                else:
                    print(f"  ⚠ No train split found in {config_name}")
            except Exception as e:
                print(f"  ✗ Error loading {config_name}: {e}")
        else:
            print(f"  ✗ Data directory not found: {data_dir}")
    
    if not combined_data:
        raise ValueError("No datasets could be loaded!")
    
    print(f"\nSuccessfully loaded {len(combined_data)} language pairs")
    total_examples = sum(len(dataset) for dataset in combined_data.values())
    print(f"Total examples across all pairs: {total_examples:,}")
    
    return DatasetDict(combined_data)


def push_combined_dataset_to_hub(dataset: DatasetDict, repo_id: str, private: bool = False):
    """
    Push the combined dataset to HuggingFace Hub.
    
    Args:
        dataset: Combined DatasetDict with all language pairs
        repo_id: HuggingFace repository ID
        private: Whether to make the dataset private
    """
    print(f"\nPushing combined dataset to HuggingFace Hub: {repo_id}")
    
    try:
        # Push to hub - this will create all configurations automatically
        # Using data_dir parameter to ensure proper structure
        print("Pushing all language pair configurations to Hub...")
        
        # Push each configuration separately to ensure proper structure
        for config_name, config_dataset in dataset.items():
            print(f"  Pushing {config_name}...")
            single_config = DatasetDict({"train": config_dataset})
            single_config.push_to_hub(
                repo_id,
                config_name=config_name,
                private=private,
                commit_message=f"Upload {config_name} configuration"
            )
            print(f"  ✓ Pushed {config_name}")
        
        print(f"✓ Successfully pushed all configurations to: https://huggingface.co/datasets/{repo_id}")
        return True
        
    except Exception as e:
        print(f"✗ Error pushing to hub: {e}")
        return False


def create_dataset_card(repo_id: str, dataset: DatasetDict):
    """Create a comprehensive dataset card."""
    
    # Get language codes
    source_langs = ["ind"]
    target_langs = []
    for config_name in dataset.keys():
        parts = config_name.split('_')
        if len(parts) == 2:
            target_langs.append(parts[1])
    
    # Language names mapping
    language_names = {
        'aaz': 'Amarasi',
        'heg': 'Helong',
        'lex': 'Luang',
        'llg': 'Lole',
        'nfa': 'Dhao',
        'ptu': 'Bambam',
        'rgu': 'Rikou',
        'row': 'Dela-Oenale',
        'tet': 'Tetun',
        'txq': 'Tii',
        'wrs': 'Waris'
    }
    
    card_content = f"""---
language:
- ind
{chr(10).join(f'- {lang}' for lang in sorted(set(target_langs)))}
task_categories:
- translation
pretty_name: eBible Indonesian Local Language Corpus
size_categories:
- 10K<n<100K
configs:
{chr(10).join(f'- config_name: {name}' + chr(10) + '  data_files:' + chr(10) + '  - split: train' + chr(10) + f'    path: {name}/train-*' for name in sorted(dataset.keys()))}
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
{chr(10).join(f"| `{name}` | {name.split('_')[0]} | {name.split('_')[1]} | {language_names.get(name.split('_')[1], 'Unknown')} | {len(dataset[name]):,} | {language_names.get(name.split('_')[1], 'Unknown')} Bible translation |" for name in sorted(dataset.keys()))}

**Total examples across all configurations:** {sum(len(ds) for ds in dataset.values()):,}

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
dataset = load_dataset("{repo_id}", "ind_aaz")
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
    dataset = load_dataset("{repo_id}", f"ind_{{lang}}")
    print(f"{{lang}}: {{len(dataset['train'])}} examples")
```

### Using for machine translation

```python
from datasets import load_dataset
from transformers import pipeline

# Load Indonesian to Tetun dataset
dataset = load_dataset("{repo_id}", "ind_tet")['train']

# Example: prepare for training
def preprocess_function(examples):
    inputs = [ex for ex in examples["source_text"]]
    targets = [ex for ex in examples["target_text"]]
    return {{"translation": [{{"ind": i, "tet": t}} for i, t in zip(inputs, targets)]}}

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
@misc{{ebible_indonesian_2024,
  title={{eBible Indonesian Local Language Corpus}},
  author={{Dataset created from eBible corpus}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/{repo_id}}}
}}
```

## License

Please refer to the original eBible corpus license terms. The texts are from Bible translations which may have their own usage terms.

## Acknowledgments

This dataset is derived from the eBible corpus. We acknowledge the translators and communities who created these Bible translations in local languages.
"""
    
    return card_content


def delete_existing_loading_script(repo_id: str):
    """Delete the existing loading script from the repository if it exists."""
    print("\nChecking for existing loading script...")
    api = HfApi()
    
    try:
        # List files in the repository
        files = api.list_repo_files(repo_id, repo_type="dataset")
        
        # Look for Python loading scripts
        loading_scripts = [f for f in files if f.endswith('.py') and not f.startswith('scripts/')]
        
        if loading_scripts:
            print(f"Found loading script(s): {loading_scripts}")
            for script in loading_scripts:
                try:
                    api.delete_file(
                        path_in_repo=script,
                        repo_id=repo_id,
                        repo_type="dataset",
                        commit_message=f"Remove broken loading script {script}"
                    )
                    print(f"  ✓ Deleted {script}")
                except Exception as e:
                    print(f"  ⚠ Could not delete {script}: {e}")
        else:
            print("No loading scripts found.")
            
    except Exception as e:
        print(f"Could not check repository files: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Push combined eBible dataset to HuggingFace Hub with all language pairs'
    )
    parser.add_argument('--repo_id', type=str, default='Davidsamuel101/ebible_local_ind_corpus',
                        help='HuggingFace repository ID')
    parser.add_argument('--private', action='store_true',
                        help='Make the dataset private')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace token (optional, will use cached token if not provided)')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test the dataset creation, do not push to hub')
    parser.add_argument('--no-confirm', action='store_true',
                        help='Skip confirmation prompt and push directly')
    parser.add_argument('--delete-loading-script', action='store_true',
                        help='Delete existing loading script from the repository')
    
    args = parser.parse_args()
    
    # Login to HuggingFace if token provided
    if args.token:
        print("Logging in to HuggingFace Hub...")
        login(token=args.token)
    else:
        print("Using cached HuggingFace credentials (run 'huggingface-cli login' if needed)")
    
    # Delete existing loading script if requested
    if args.delete_loading_script and not args.test_only:
        delete_existing_loading_script(args.repo_id)
    
    # Create combined dataset
    print("\n1. Creating combined dataset with all language pairs")
    print("=" * 60)
    try:
        dataset = create_combined_dataset()
    except Exception as e:
        print(f"Error creating combined dataset: {e}")
        return
    
    if args.test_only:
        print("\nTest completed successfully. Exiting without pushing to hub.")
        # Test loading locally
        print("\nTesting local dataset structure...")
        for config_name in dataset.keys():
            print(f"  Config: {config_name}, Examples: {len(dataset[config_name])}")
        return
    
    # Create dataset card
    print(f"\n2. Creating dataset card")
    print("=" * 60)
    card_content = create_dataset_card(args.repo_id, dataset)
    
    # Save dataset card locally for review
    card_path = "hf_dataset_README.md"
    with open(card_path, 'w') as f:
        f.write(card_content)
    print(f"Dataset card saved to {card_path}")
    
    # Push to hub
    print(f"\n3. Pushing to HuggingFace Hub")
    print("=" * 60)
    
    # Ask for confirmation unless --no-confirm is set
    if args.no_confirm:
        response = 'y'
        print(f"\nPushing combined dataset with all configurations to {args.repo_id} (--no-confirm flag set)")
    else:
        print(f"\nReady to push combined dataset with all configurations to {args.repo_id}")
        print("This will push each configuration separately to ensure proper structure.")
        response = input("Continue? (y/n): ")
    
    if response.lower() == 'y':
        if push_combined_dataset_to_hub(dataset, args.repo_id, args.private):
            # Also push the dataset card
            print("\nPushing dataset card...")
            api = HfApi()
            api.upload_file(
                path_or_fileobj=card_path,
                path_in_repo="README.md",
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message="Update dataset card with configuration details"
            )
            print("✓ Dataset card uploaded")
            
            print("\n" + "=" * 60)
            print("Upload complete!")
            print(f"View your dataset at: https://huggingface.co/datasets/{args.repo_id}")
            
            # Test loading from hub
            print("\n4. Testing dataset loading from HuggingFace Hub")
            print("=" * 60)
            test_loading_from_hub(args.repo_id)
        else:
            print("Failed to push dataset to hub.")
    else:
        print("Upload cancelled.")


def test_loading_from_hub(repo_id: str):
    """Test loading the dataset from HuggingFace Hub."""
    from datasets import load_dataset
    
    print("Testing dataset loading...")
    target_langs = ["aaz", "ptu", "nfa", "heg", "lex", "row", "llg", "rgu", "txq", "tet", "wrs"]
    
    success_count = 0
    for lang in target_langs[:3]:  # Test first 3 to save time
        try:
            print(f"  Loading ind_{lang}...")
            dataset = load_dataset(repo_id, f"ind_{lang}")
            print(f"    ✓ Successfully loaded ind_{lang}: {len(dataset['train'])} examples")
            success_count += 1
        except Exception as e:
            print(f"    ✗ Error loading ind_{lang}: {e}")
    
    if success_count > 0:
        print(f"\n✓ Dataset loading test successful! ({success_count}/3 configs loaded)")
        print("\nYou can now load the dataset with:")
        print(f'  dataset = load_dataset("{repo_id}", "ind_<lang>")')
        print(f"  Where <lang> is one of: {', '.join(target_langs)}")
    else:
        print("\n⚠ Dataset loading test failed. The dataset may need time to process on HuggingFace servers.")
        print("Try again in a few minutes.")


if __name__ == "__main__":
    main()