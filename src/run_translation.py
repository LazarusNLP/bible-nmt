#!/usr/bin/env python
"""
Improved Bible translation training script v2.
Combines HuggingFace data loading with critical improvements from silnlp analysis:
- forced_bos_token_id for target language
- Better hyperparameters (learning rate, label smoothing)
- Proper model configuration (dtype, attention)
"""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
import numpy as np
import evaluate
import torch
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
import logging
import wandb
from sklearn.model_selection import train_test_split
import random
import pandas as pd

# Import text normalization
from text_normalization import normalize_text, normalize_parallel_texts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NT_BOOKS = [
    "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", 
    "GAL", "EPH", "PHP", "COL", "1TH", "2TH", "1TI", "2TI", 
    "TIT", "PHM", "HEB", "JAB", "JAS", "1PE", "2PE", "1JN", 
    "2JN", "3JN", "JUD", "REV",
]


def parse_args():
    parser = ArgumentParser()
    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bible-nlp/biblenlp-corpus",
        choices=["bible-nlp/biblenlp-corpus", "LazarusNLP/alkitab-sabda-mt", "Davidsamuel101/ebible_local_ind_corpus", "scripture_files"],
    )
    parser.add_argument("--src_lang", type=str, default="hau")
    parser.add_argument("--tgt_lang", type=str, default="daa")
    parser.add_argument("--src_lang_nllb", type=str, default="hau_Latn")
    parser.add_argument("--tgt_lang_nllb", type=str, default="daa_Latn")
    # Scripture file paths (for dataset_name="scripture_files")
    parser.add_argument("--source_text_path", type=str, default=None, help="Path to source language scripture text file")
    parser.add_argument("--target_text_path", type=str, default=None, help="Path to target language scripture text file")
    parser.add_argument("--verse_text_path", type=str, default=None, help="Path to verse reference file (e.g., GEN 1:1)")
    # Additional CSV data arguments
    parser.add_argument("--additional_csv_path", type=str, default=None, help="Path to additional CSV training data")
    parser.add_argument("--csv_source_col", type=str, default="source_text", help="Name of source text column in CSV")
    parser.add_argument("--csv_target_col", type=str, default="target_text", help="Name of target text column in CSV")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--torch_dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    
    # Training arguments
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--label_smoothing_factor", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=114)
    
    # Output arguments
    parser.add_argument("--save_tokenized_data", action="store_true", help="Save tokenized data files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (auto-generated if not specified)")
    
    return parser.parse_args()


def process_translations(x, src_lang: str, tgt_lang: str):
    """
    Select a single source/target translation per row and record chosen files.
    - Source is `src_lang`; if multiple versions exist, prefer '{src_lang}-{src_lang}.txt'
    - Target is `tgt_lang`; assumed unique in the row.
    """
    tr = x.get("translation") or {}
    languages = list(tr.get("language") or [])
    texts = list(tr.get("translation") or [])

    files_info = x.get("files") or {}
    file_names = files_info.get("file") if isinstance(files_info, dict) else None

    # Select source (prefer '{src}-{src}.txt' when multiple)
    src_indices = [i for i, lang in enumerate(languages) if lang == src_lang]
    selected_source_idx = None
    if src_indices:
        if len(src_indices) == 1:
            selected_source_idx = src_indices[0]
        else:
            preferred_idx = None
            if isinstance(file_names, list) and len(file_names) == len(languages):
                preferred_filename = f"{src_lang}-{src_lang}-web.txt"
                for idx in src_indices:
                    if file_names[idx] == preferred_filename:
                        preferred_idx = idx
                        break
            selected_source_idx = preferred_idx if preferred_idx is not None else src_indices[0]

    # Select target (first occurrence of tgt_lang)
    tgt_indices = [i for i, lang in enumerate(languages) if lang == tgt_lang]
    selected_target_idx = tgt_indices[0] if tgt_indices else None

    source_text = texts[selected_source_idx] if selected_source_idx is not None and selected_source_idx < len(texts) else ""
    target_text = texts[selected_target_idx] if selected_target_idx is not None and selected_target_idx < len(texts) else ""

    # Apply text normalization
    source_text = normalize_text(source_text) if source_text else ""
    target_text = normalize_text(target_text) if target_text else ""

    return {
        "source": source_text,
        "target": target_text,
    }


def load_ebible_corpus(src_lang, tgt_lang):
    """Load Bible corpus from HuggingFace."""
    dataset = load_dataset("bible-nlp/biblenlp-corpus", languages=[src_lang, tgt_lang], trust_remote_code=True)
    dataset = dataset.map(process_translations, fn_kwargs={"src_lang": src_lang, "tgt_lang": tgt_lang})
    
    # OT books for testing, NT books for training and validation
    def is_nt_book(refs):
        if isinstance(refs, list):
            return any(ref.split()[0] in NT_BOOKS for ref in refs)
        else:
            return refs.split()[0] in NT_BOOKS
    
    train_data = dataset['train']
    valid_data = dataset['validation']
    train_data = concatenate_datasets([train_data, valid_data])
    train_ds = train_data.filter(lambda x: is_nt_book(x["ref"]))
    test_ds = train_data.filter(lambda x: not is_nt_book(x["ref"]))
    train_val_ds = train_ds.train_test_split(test_size=0.05, seed=41)
    
    dataset = DatasetDict({
        "train": train_val_ds["train"], 
        "validation": train_val_ds["test"], 
        "test": test_ds
    })
    
    logger.info(f"Dataset splits - Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
    return dataset


def load_alkitab_sabda(src_lang, tgt_lang):
    """Load Alkitab SABDA dataset."""
    dataset = load_dataset(
        "LazarusNLP/alkitab-sabda-mt",
        f"{src_lang}-{tgt_lang}",
        split="train+validation+test",
        trust_remote_code=True,
    )
    dataset = dataset.map(lambda x: {"book": x["verse_id"].split("_")[0]})
    dataset = dataset.rename_column("source_text", "source")
    dataset = dataset.rename_column("target_text", "target")

    # OT books for testing, NT books for training and validation
    train_ds = dataset.filter(lambda x: x["book"] in NT_BOOKS)
    test_ds = dataset.filter(lambda x: x["book"] not in NT_BOOKS)
    train_val_ds = train_ds.train_test_split(test_size=0.1, seed=41)
    
    dataset = DatasetDict({
        "train": train_val_ds["train"], 
        "validation": train_val_ds["test"], 
        "test": test_ds
    })
    
    logger.info(f"Dataset splits - Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
    return dataset


def load_ebible_local_ind_corpus(src_lang, tgt_lang):
    """Load local Indonesian Bible corpus."""
    dataset = load_dataset("Davidsamuel101/ebible_local_ind_corpus", f"{src_lang}_{tgt_lang}")
    
    def is_nt_book(refs):
        if isinstance(refs, list):
            return any(ref.split()[0] in NT_BOOKS for ref in refs)
        else:
            return refs.split()[0] in NT_BOOKS
    
    train_data = dataset['train']
    train_data = train_data.rename_column("source_text", "source")
    train_data = train_data.rename_column("target_text", "target")
    
    train_ds = train_data.filter(lambda x: is_nt_book(x["verse"]))
    test_ds = train_data.filter(lambda x: not is_nt_book(x["verse"]))
    train_val_ds = train_ds.train_test_split(test_size=0.05, seed=41)
    
    dataset = DatasetDict({
        "train": train_val_ds["train"], 
        "validation": train_val_ds["test"], 
        "test": test_ds
    })
    
    logger.info(f"Dataset splits - Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
    return dataset


def load_csv_data(csv_path, source_col="source_text", target_col="target_text"):
    """Load parallel translation data from CSV file.
    
    Args:
        csv_path: Path to CSV file
        source_col: Name of source text column (default: "source_text")
        target_col: Name of target text column (default: "target_text")
    
    Returns:
        Dataset with 'source' and 'target' columns
    """
    logger.info(f"Loading CSV data from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check if columns exist
    if source_col not in df.columns:
        raise ValueError(f"Source column '{source_col}' not found in CSV. Available columns: {list(df.columns)}")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Filter out empty/null rows
    df = df.dropna(subset=[source_col, target_col])
    df = df[(df[source_col].str.strip() != '') & (df[target_col].str.strip() != '')]
    
    # Create list of dictionaries for Dataset.from_list with text normalization
    data_pairs = []
    for _, row in df.iterrows():
        source_text = str(row[source_col]).strip()
        target_text = str(row[target_col]).strip()
        
        # Apply text normalization
        source_text = normalize_text(source_text) if source_text else ""
        target_text = normalize_text(target_text) if target_text else ""
        
        data_pairs.append({
            "source": source_text,
            "target": target_text
        })
    
    logger.info(f"Loaded {len(data_pairs)} parallel translation pairs from CSV")
    
    return Dataset.from_list(data_pairs)


def load_scripture_files(source_path, target_path, verse_path=None, seed=114):
    """Load parallel scripture text files handling <range> markers (like silnlp does).
    
    Files should have the same number of lines, with each line being a verse.
    <range> markers indicate verse continuations that should be merged with previous verse.
    Empty lines indicate verses that don't exist in translation.
    
    If verse_path is provided, uses OT/NT split like load_ebible_local_ind_corpus:
    - OT for test, NT for train/val
    - 5% random sample from NT for validation
    Otherwise uses random verse sampling.
    """
    logger.info(f"Loading scripture files: {source_path} and {target_path}")
    if verse_path:
        logger.info(f"Using verse references from: {verse_path}")
    
    with open(source_path, 'r', encoding='utf-8') as f:
        source_lines = [line.strip() for line in f]
    
    with open(target_path, 'r', encoding='utf-8') as f:
        target_lines = [line.strip() for line in f]
    
    assert len(source_lines) == len(target_lines), f"Line count mismatch: {len(source_lines)} vs {len(target_lines)}"
    
    # Load verse references if provided
    verse_refs = None
    if verse_path:
        with open(verse_path, 'r', encoding='utf-8') as f:
            verse_refs = [line.strip() for line in f]
        assert len(verse_refs) == len(source_lines), f"Verse ref count mismatch: {len(verse_refs)} vs {len(source_lines)}"
    
    # Process lines handling <range> markers and empty lines
    data_pairs = []
    
    i = 0
    while i < len(source_lines):
        src_line = source_lines[i]
        trg_line = target_lines[i]
        
        # Track verse reference
        current_verse = verse_refs[i] if verse_refs else None
        
        # Skip completely empty pairs
        if not src_line and not trg_line:
            i += 1
            continue
            
        # Skip pairs with "..." (incomplete translations)
        if src_line == "..." or trg_line == "...":
            i += 1
            continue
        
        # Check for <range> markers and accumulate text
        accumulated_src = []
        accumulated_trg = []
        verse_span_refs = []
        
        # Process current line and any following <range> lines
        while i < len(source_lines):
            src = source_lines[i]
            trg = target_lines[i]
            
            if verse_refs:
                verse_span_refs.append(verse_refs[i])
            
            # If both are <range>, continue to next
            if src == "<range>" and trg == "<range>":
                i += 1
                continue
            
            # If source is <range>, only add target if not empty
            if src == "<range>":
                if trg and trg != "<range>":
                    accumulated_trg.append(trg)
            # If target is <range>, only add source if not empty
            elif trg == "<range>":
                if src and src != "<range>":
                    accumulated_src.append(src)
            # Normal content - add both
            else:
                if src and src != "<range>":
                    accumulated_src.append(src)
                if trg and trg != "<range>":
                    accumulated_trg.append(trg)
            
            # Check if next line is a <range> marker
            i += 1
            if i >= len(source_lines) or (source_lines[i] != "<range>" and target_lines[i] != "<range>"):
                break
        
        # Create verse pair if we have both source and target content
        if accumulated_src and accumulated_trg:
            src_text = " ".join(accumulated_src)
            trg_text = " ".join(accumulated_trg)
            
            # Apply text normalization
            src_text = normalize_text(src_text) if src_text else ""
            trg_text = normalize_text(trg_text) if trg_text else ""
            
            # Create verse reference (span if multiple verses)
            if verse_refs and verse_span_refs:
                if len(verse_span_refs) == 1:
                    verse_ref = verse_span_refs[0]
                else:
                    # Create span like "GEN 1:1-3"
                    first_verse = verse_span_refs[0]
                    last_verse = verse_span_refs[-1]
                    # Extract book and chapter from first verse
                    parts = first_verse.split()
                    if len(parts) == 2 and ':' in parts[1]:
                        book = parts[0]
                        chapter_verse = parts[1].split(':')
                        if len(chapter_verse) == 2:
                            chapter = chapter_verse[0]
                            first_verse_num = chapter_verse[1]
                            # Get last verse number
                            last_parts = last_verse.split()
                            if len(last_parts) == 2 and ':' in last_parts[1]:
                                last_verse_num = last_parts[1].split(':')[1]
                                verse_ref = f"{book} {chapter}:{first_verse_num}-{last_verse_num}"
                            else:
                                verse_ref = f"{first_verse}-{last_verse}"
                        else:
                            verse_ref = f"{first_verse}-{last_verse}"
                    else:
                        verse_ref = f"{first_verse}-{last_verse}"
            else:
                verse_ref = None
            
            pair = {"source": src_text, "target": trg_text}
            if verse_ref:
                pair["verse"] = verse_ref
            data_pairs.append(pair)
    
    logger.info(f"Loaded {len(data_pairs)} parallel verses (after handling <range> and filtering empty)")
    
    # Set random seed
    import random
    random.seed(seed)
    
    if verse_refs:
        # Use NT/OT split like load_ebible_local_ind_corpus
        def is_nt_book(verse_ref):
            """Check if verse reference is from NT book."""
            if not verse_ref:
                return False
            # Handle verse spans like "MAT 1:1-3"
            book = verse_ref.split()[0] if verse_ref else ""
            return book in NT_BOOKS
        
        # Split by NT/OT
        nt_data = [pair for pair in data_pairs if is_nt_book(pair.get("verse", ""))]
        ot_data = [pair for pair in data_pairs if not is_nt_book(pair.get("verse", ""))]
        
        logger.info(f"NT verses: {len(nt_data)}, OT verses: {len(ot_data)}")
        
        # OT for test, NT for train/val with 5% random split
        test_data = ot_data
        
        # Split NT data: 95% train, 5% validation
        if len(nt_data) > 0:
            val_size = max(1, int(len(nt_data) * 0.05))
            indices = list(range(len(nt_data)))
            random.shuffle(indices)
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            
            val_data = [nt_data[i] for i in val_indices]
            train_data = [nt_data[i] for i in train_indices]
        else:
            train_data = []
            val_data = []
        
        logger.info(f"NT/OT split - Train: {len(train_data)} (NT), Val: {len(val_data)} (5% of NT), Test: {len(test_data)} (OT)")
    else:
        # Original random sampling (keeping for backward compatibility)
        total_verses = len(data_pairs)
        all_indices = list(range(total_verses))
        
        # Default test/val sizes for random sampling
        test_size = 250
        val_size = 250
        
        # Randomly sample test verses
        test_sample_size = min(test_size, total_verses)
        test_indices = set(random.sample(all_indices, test_sample_size))
        
        # Remove test indices from available pool
        remaining_indices = [i for i in all_indices if i not in test_indices]
        
        # Sample validation verses from remaining
        val_sample_size = min(val_size, len(remaining_indices))
        val_indices = set(random.sample(remaining_indices, val_sample_size))
        
        # Everything else is training
        train_indices = [i for i in remaining_indices if i not in val_indices]
        
        # Create splits
        train_data = [data_pairs[i] for i in train_indices]
        val_data = [data_pairs[i] for i in val_indices]
        test_data = [data_pairs[i] for i in test_indices]
        
        logger.info(f"Random verse sampling - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Convert to datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    return dataset


def save_tokenized_data(tokenizer, dataset, output_dir, split_name, src_lang_nllb, tgt_lang_nllb):
    """Save tokenized and detokenized data files."""
    output_dir = Path(output_dir)
    
    # Get the data for this split
    split_data = dataset[split_name]
    
    # Save detokenized (raw text) files
    with open(output_dir / f"{split_name}.src.detok.txt", 'w', encoding='utf-8') as f:
        for example in split_data:
            f.write(example['source'] + '\n')
    
    with open(output_dir / f"{split_name}.trg.detok.txt", 'w', encoding='utf-8') as f:
        for example in split_data:
            f.write(example['target'] + '\n')
    
    # Tokenize and save tokenized files
    tokenized_src = []
    tokenized_trg = []
    
    for example in split_data:
        # Tokenize source (add source language prefix)
        src_tokens = tokenizer.tokenize(example['source'])
        src_with_lang = [src_lang_nllb] + src_tokens + ['</s>']
        tokenized_src.append(' '.join(src_with_lang))
        
        # Tokenize target (add target language prefix)
        trg_tokens = tokenizer.tokenize(example['target'])
        trg_with_lang = [tgt_lang_nllb] + trg_tokens + ['</s>']
        tokenized_trg.append(' '.join(trg_with_lang))
    
    # Save tokenized files
    with open(output_dir / f"{split_name}.src.txt", 'w', encoding='utf-8') as f:
        for line in tokenized_src:
            f.write(line + '\n')
    
    with open(output_dir / f"{split_name}.trg.txt", 'w', encoding='utf-8') as f:
        for line in tokenized_trg:
            f.write(line + '\n')
    
    logger.info(f"Saved {split_name} data: {len(split_data)} examples")


class FixedLabelSmoothingSeq2SeqTrainer(Seq2SeqTrainer):
    """Workaround for label smoothing."""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        if labels is not None and self.label_smoother is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss


def main(args):
    # Set up output directory
    if args.output_dir is None:
        dataset_name = args.dataset_name.split("/")[-1]
        model_name = args.model_name.split("/")[-1]
        output_dir = f"{model_name}-{dataset_name}-{args.src_lang}webp-{args.tgt_lang}-{args.max_steps}"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="bible-nmt",
        name=output_dir,
        config=vars(args)
    )
    
    # Load main dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    if args.dataset_name == "scripture_files":
        if not args.source_text_path or not args.target_text_path:
            raise ValueError("When using scripture_files, you must provide --source_text_path and --target_text_path")
        dataset = load_scripture_files(args.source_text_path, args.target_text_path, args.verse_text_path, seed=args.seed)
    elif "biblenlp-corpus" in args.dataset_name:
        dataset = load_ebible_corpus(args.src_lang, args.tgt_lang)
    elif "alkitab-sabda-mt" in args.dataset_name:
        dataset = load_alkitab_sabda(args.src_lang, args.tgt_lang)
    elif "ebible_local_ind_corpus" in args.dataset_name:
        dataset = load_ebible_local_ind_corpus(args.src_lang, args.tgt_lang)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    # Load and concatenate additional CSV data if provided
    if args.additional_csv_path:
        logger.info(f"Loading additional CSV data from: {args.additional_csv_path}")
        additional_data = load_csv_data(args.additional_csv_path, args.csv_source_col, args.csv_target_col)
        
        # Concatenate additional data with existing training data only
        logger.info(f"Original train dataset size: {len(dataset['train'])}")
        dataset['train'] = concatenate_datasets([dataset['train'], additional_data])
        logger.info(f"New train dataset size after adding CSV data: {len(dataset['train'])}")
        
        # Update dataset info for logging
        logger.info(f"Final dataset splits - Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
    
    # Set torch dtype
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    torch_dtype = dtype_map[args.torch_dtype]
    
    # Load model with specific configuration
    logger.info(f"Loading model: {args.model_name} with dtype={args.torch_dtype}, attn={args.attn_implementation}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Add special tokens for source and target languages if they are not already present
    if args.src_lang_nllb not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": [args.src_lang_nllb]})

    if args.tgt_lang_nllb not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": [args.tgt_lang_nllb]})

    tokenizer.src_lang = args.src_lang_nllb
    tokenizer.tgt_lang = args.tgt_lang_nllb

    # Initialize new token embeddings with the mean of old embeddings
    old_embeddings = model.get_input_embeddings()
    old_num_tokens = old_embeddings.weight.size(dim=0)
    old_embeddings_mean = old_embeddings.weight.mean(dim=0, keepdim=True)

    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    embeddings.weight.data[old_num_tokens:, :] = old_embeddings_mean
    model.tie_weights()
    
    # CRITICAL: Set forced_bos_token_id for target language
    tgt_lang_token_id = tokenizer.convert_tokens_to_ids(args.tgt_lang_nllb)
    logger.info(f"{args.tgt_lang_nllb} token ID: {tgt_lang_token_id}")
    
    model.generation_config.forced_bos_token_id = tgt_lang_token_id
    logger.info(f"✅ Set forced_bos_token_id to {tgt_lang_token_id} ({args.tgt_lang_nllb}) - CRITICAL for performance!")
    
    # Save tokenized data if requested
    if args.save_tokenized_data:
        logger.info("Saving tokenized and detokenized data files...")
        for split_name in ['train', 'validation', 'test']:
            save_tokenized_data(tokenizer, dataset, output_dir, split_name, args.src_lang_nllb, args.tgt_lang_nllb)
    
    def preprocess_function(examples):
        return tokenizer(
            examples["source"],
            text_target=examples["target"],
            max_length=args.max_length,
            padding="max_length",
            truncation=True,
        )
    
    # Process datasets
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=args.num_proc,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Metrics
    sacrebleu = evaluate.load("sacrebleu", trust_remote_code=True)
    chrf = evaluate.load("chrf", trust_remote_code=True)
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        tokenized_preds = [word_tokenize(pred.lower().strip()) for pred in decoded_preds]
        tokenized_refs = [[word_tokenize(ref[0].lower().strip())] for ref in decoded_labels]
        bleu_score = corpus_bleu(tokenized_refs, tokenized_preds)
        
        sacrebleu_result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        chrf3_result = chrf.compute(predictions=decoded_preds, references=decoded_labels, beta=3)
        
        result = {"bleu": bleu_score * 100, "sacrebleu": sacrebleu_result["score"], "chrf3": chrf3_result["score"]}
        return {k: round(v, 4) for k, v in result.items()}
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        optim="adamw_torch",
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        label_smoothing_factor=args.label_smoothing_factor,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        predict_with_generate=True,
        generation_num_beams=args.num_beams,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        group_by_length=True,
        logging_strategy="steps",
        logging_steps=500,
        report_to="wandb",
        logging_dir=f"{output_dir}/logs",
        run_name=output_dir,
        seed=args.seed,
    )
    
    # Save training configuration
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        config = {
            "args": vars(args),
            "training_args": training_args.to_dict(),
            "forced_bos_token_id": tgt_lang_token_id,
            "dataset_splits": {
                "train": len(dataset["train"]),
                "validation": len(dataset["validation"]),
                "test": len(dataset["test"]),
            }
        }
        json.dump(config, f, indent=2, default=str)
    
    # Callbacks
    callbacks = []
    if args.early_stopping_patience:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    
    # Trainer
    trainer = FixedLabelSmoothingSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks if callbacks else None,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model and configs
    trainer.save_model()
    trainer.create_model_card()
    tokenizer.save_pretrained(output_dir)
    
    # Explicitly save the generation config with forced_bos_token_id
    model.generation_config.save_pretrained(output_dir)
    logger.info(f"Saved generation config with forced_bos_token_id: {tgt_lang_token_id}")
    
    logger.info(f"Training completed. Model saved to {output_dir}")
    
    # Finish wandb
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)