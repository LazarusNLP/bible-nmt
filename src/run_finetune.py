#!/usr/bin/env python
"""
Bible translation finetuning script for CSV files.
Based on run_translation.py but adapted to load training data from one CSV file 
and testing data from another CSV file.
"""

from argparse import ArgumentParser
import json
import os
import pandas as pd
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset, DatasetDict
import numpy as np
import evaluate
import torch
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
import logging
import wandb
from sklearn.model_selection import train_test_split
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    # Data arguments
    parser.add_argument("--train_csv_path", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--test_csv_path", type=str, required=True, help="Path to testing CSV file")
    parser.add_argument("--src_lang", type=str, default="eng")
    parser.add_argument("--tgt_lang", type=str, default="luang")
    parser.add_argument("--src_lang_nllb", type=str, default="eng_Latn")
    parser.add_argument("--tgt_lang_nllb", type=str, default="lex_Latn")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--torch_dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    
    # Training arguments
    parser.add_argument("--max_length", type=int, default=300)
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
    parser.add_argument("--validation_split", type=float, default=0.1, help="Fraction of training data to use for validation")
    
    # Output arguments
    parser.add_argument("--save_tokenized_data", action="store_true", help="Save tokenized data files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (auto-generated if not specified)")
    
    return parser.parse_args()


def load_csv_dataset(train_csv_path, test_csv_path, validation_split=0.1, seed=114):
    """Load parallel text data from CSV files.
    
    Args:
        train_csv_path: Path to training CSV with source_text and target_text columns
        test_csv_path: Path to testing CSV with source_text and target_text columns  
        validation_split: Fraction of training data to use for validation
        seed: Random seed for train/validation split
    
    Returns:
        DatasetDict with train, validation, and test splits
    """
    logger.info(f"Loading training data from: {train_csv_path}")
    logger.info(f"Loading testing data from: {test_csv_path}")
    
    # Load training CSV
    train_df = pd.read_csv(train_csv_path)
    logger.info(f"Training CSV columns: {list(train_df.columns)}")
    
    # Load testing CSV  
    test_df = pd.read_csv(test_csv_path)
    logger.info(f"Testing CSV columns: {list(test_df.columns)}")
    
    # Ensure required columns exist
    required_cols = ['source_text', 'target_text']
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Training CSV missing required column: {col}")
        if col not in test_df.columns:
            raise ValueError(f"Testing CSV missing required column: {col}")
    
    # Clean and filter data
    def clean_data(df):
        # Remove rows with missing values
        df = df.dropna(subset=['source_text', 'target_text'])
        # Remove empty strings
        df = df[(df['source_text'].str.strip() != '') & (df['target_text'].str.strip() != '')]
        # Remove very long sequences (over 500 chars)
        df = df[(df['source_text'].str.len() <= 500) & (df['target_text'].str.len() <= 500)]
        return df
    
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)
    
    logger.info(f"After cleaning - Training: {len(train_df)} pairs, Testing: {len(test_df)} pairs")
    
    # Convert to the format expected by the training script
    train_data = [
        {"source": row['source_text'], "target": row['target_text']}
        for _, row in train_df.iterrows()
    ]
    
    test_data = [
        {"source": row['source_text'], "target": row['target_text']}
        for _, row in test_df.iterrows()
    ]
    
    # Split training data into train and validation
    if validation_split > 0:
        train_pairs, val_pairs = train_test_split(
            train_data, 
            test_size=validation_split, 
            random_state=seed
        )
    else:
        train_pairs = train_data
        val_pairs = train_data[:min(50, len(train_data))]  # Use small subset for validation
    
    # Convert to datasets
    train_dataset = Dataset.from_list(train_pairs)
    val_dataset = Dataset.from_list(val_pairs)
    test_dataset = Dataset.from_list(test_data)
    
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    logger.info(f"Dataset splits - Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
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
        model_name = args.model_name.split("/")[-1]
        output_dir = f"{model_name}-finetune-{args.src_lang}-{args.tgt_lang}-{args.max_steps}"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="bible-nmt-finetune",
        name=output_dir,
        config=vars(args)
    )
    
    # Load dataset from CSV files
    logger.info("Loading dataset from CSV files...")
    dataset = load_csv_dataset(
        args.train_csv_path, 
        args.test_csv_path, 
        args.validation_split, 
        args.seed
    )
    
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
    
    # Check if language tokens exist in vocabulary (proper way for NLLB)
    # First check if tokens are in the tokenizer's vocabulary
    src_token_id = tokenizer.convert_tokens_to_ids(args.src_lang_nllb)
    tgt_token_id = tokenizer.convert_tokens_to_ids(args.tgt_lang_nllb)
    
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    logger.info(f"Source token '{args.src_lang_nllb}' -> ID: {src_token_id}")
    logger.info(f"Target token '{args.tgt_lang_nllb}' -> ID: {tgt_token_id}")
    logger.info(f"UNK token ID: {tokenizer.unk_token_id}")
    
    # Check if tokens exist in additional_special_tokens
    additional_special = getattr(tokenizer, 'additional_special_tokens', [])
    logger.info(f"Additional special tokens: {additional_special}")
    
    # For proper finetuning: tokens should already exist and NOT need to be added
    tokens_to_add = []
    
    # Check source token
    if src_token_id != tokenizer.unk_token_id:
        logger.info(f"✅ Source language token {args.src_lang_nllb} already exists (ID: {src_token_id})")
    else:
        tokens_to_add.append(args.src_lang_nllb)
        logger.info(f"⚠️ Source language token {args.src_lang_nllb} not found, will add it")
        
    # Check target token  
    if tgt_token_id != tokenizer.unk_token_id:
        logger.info(f"✅ Target language token {args.tgt_lang_nllb} already exists (ID: {tgt_token_id})")
    else:
        tokens_to_add.append(args.tgt_lang_nllb)
        logger.info(f"⚠️ Target language token {args.tgt_lang_nllb} not found, will add it")

    # Only add tokens and resize embeddings if needed
    if tokens_to_add:
        logger.info(f"Adding new tokens: {tokens_to_add}")
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        
        # Initialize new token embeddings with the mean of old embeddings
        old_embeddings = model.get_input_embeddings()
        old_num_tokens = old_embeddings.weight.size(dim=0)
        old_embeddings_mean = old_embeddings.weight.mean(dim=0, keepdim=True)

        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        embeddings.weight.data[old_num_tokens:, :] = old_embeddings_mean
        model.tie_weights()
        logger.info(f"Initialized {len(tokens_to_add)} new token embeddings")
    else:
        logger.info("✅ No new tokens needed - using existing embeddings from checkpoint")

    tokenizer.src_lang = args.src_lang_nllb
    tokenizer.tgt_lang = args.tgt_lang_nllb
    
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
    logger.info("Starting finetuning...")
    trainer.train()
    
    # Save model and configs
    trainer.save_model()
    trainer.create_model_card()
    tokenizer.save_pretrained(output_dir)
    
    # Explicitly save the generation config with forced_bos_token_id
    model.generation_config.save_pretrained(output_dir)
    logger.info(f"Saved generation config with forced_bos_token_id: {tgt_lang_token_id}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=processed_dataset["test"])
    
    # Save test results
    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"Test Results: {test_results}")
    logger.info(f"Finetuning completed. Model saved to {output_dir}")
    
    # Log final results to wandb
    wandb.log({"test_bleu": test_results.get("eval_bleu", 0)})
    wandb.log({"test_sacrebleu": test_results.get("eval_sacrebleu", 0)})
    wandb.log({"test_chrf3": test_results.get("eval_chrf3", 0)})
    
    # Finish wandb
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
