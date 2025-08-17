from argparse import ArgumentParser
import json

from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset, DatasetDict, concatenate_datasets
import numpy as np
import evaluate
import os
import wandb
import torch

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        print("No .env file found, proceeding without loading environment variables")
except ImportError:
    print("python-dotenv not installed, skipping .env file loading")

NT_BOOKS = [
    "MAT",
    "MRK",
    "LUK",
    "JHN",
    "ACT",
    "ROM",
    "1CO",
    "2CO",
    "GAL",
    "EPH",
    "PHP",
    "COL",
    "1TH",
    "2TH",
    "1TI",
    "2TI",
    "TIT",
    "PHM",
    "HEB",
    "JAB",
    "JAS",
    "1PE",
    "2PE",
    "1JN",
    "2JN",
    "3JN",
    "JUD",
    "REV",
]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-distilled-1.3B")
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["bible-nlp/biblenlp-corpus", "LazarusNLP/alkitab-sabda-mt", "Davidsamuel101/ebible_local_ind_corpus"],
    )
    parser.add_argument("--src_lang", type=str, default="ind")
    parser.add_argument("--tgt_lang", type=str, default="ptu")
    parser.add_argument("--src_lang_nllb", type=str, default="ind_Latn")
    parser.add_argument("--tgt_lang_nllb", type=str, default="ptu_Latn")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--label_smoothing_factor", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    return parser.parse_args()


def process_translations(x, src_lang: str, tgt_lang: str):
    """
    Select a single source/target translation per row and record chosen files.
    - Source is `src_lang`; if multiple versions exist, prefer '{src_lang}-{src_lang}.txt' (e.g., 'ind-ind.txt').
    - Target is `tgt_lang`; assumed unique in the row.
    Returns: source_text, target_text, source_file, target_file
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
                preferred_filename = f"{src_lang}-{src_lang}ags.txt"
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

    source_file = ""
    target_file = ""
    if isinstance(file_names, list) and len(file_names) == len(languages):
        if selected_source_idx is not None and selected_source_idx < len(file_names):
            source_file = file_names[selected_source_idx]
        if selected_target_idx is not None and selected_target_idx < len(file_names):
            target_file = file_names[selected_target_idx]

    return {
        "source_text": source_text,
        "target_text": target_text,
        "source_file": source_file,
        "target_file": target_file,
    }
    

def load_ebible_corpus(src_lang, tgt_lang):
    dataset = load_dataset("bible-nlp/biblenlp-corpus", languages=[src_lang, tgt_lang], trust_remote_code=True)
    dataset = dataset.map(process_translations, fn_kwargs={"src_lang": src_lang, "tgt_lang": tgt_lang})
    # OT books for testing, NT books for training and validation
    def is_nt_book(refs):
        if isinstance(refs, list):
            # Check if any ref belongs to NT books
            return any(ref.split()[0] in NT_BOOKS for ref in refs)
        else:
            # Single reference
            return refs.split()[0] in NT_BOOKS
    
    # The dataset is a DatasetDict, so we need to access the 'train' split
    train_data = dataset['train']
    valid_data = dataset['validation']
    train_data = concatenate_datasets([train_data, valid_data])
    train_ds = train_data.filter(lambda x: is_nt_book(x["ref"]))
    test_ds = train_data.filter(lambda x: not is_nt_book(x["ref"]))
    train_val_ds = train_ds.train_test_split(test_size=0.05, seed=41)
    dataset = DatasetDict({"train": train_val_ds["train"], "validation": train_val_ds["test"], "test": test_ds})
    print(dataset)
    return dataset


def load_alkitab_sabda(src_lang, tgt_lang):
    dataset = load_dataset(
        "LazarusNLP/alkitab-sabda-mt",
        f"{src_lang}-{tgt_lang}",
        split="train+validation+test",
        trust_remote_code=True,
    )
    dataset = dataset.map(lambda x: {"book": x["verse_id"].split("_")[0]})

    # OT books for testing, NT books for training and validation
    train_ds = dataset.filter(lambda x: x["book"] in NT_BOOKS)
    test_ds = dataset.filter(lambda x: x["book"] not in NT_BOOKS)
    train_val_ds = train_ds.train_test_split(test_size=0.1, seed=41)
    dataset = DatasetDict({"train": train_val_ds["train"], "validation": train_val_ds["test"], "test": test_ds})
    return dataset

def load_ebible_local_ind_corpus(src_lang, tgt_lang):
    dataset = load_dataset("Davidsamuel101/ebible_local_ind_corpus", f"{src_lang}_{tgt_lang}")
    # OT books for testing, NT books for training and validation
    def is_nt_book(refs):
        if isinstance(refs, list):
            # Check if any ref belongs to NT books
            return any(ref.split()[0] in NT_BOOKS for ref in refs)
        else:
            # Single reference
            return refs.split()[0] in NT_BOOKS
    
    # The dataset is a DatasetDict, so we need to access the 'train' split
    train_data = dataset['train']
    train_ds = train_data.filter(lambda x: is_nt_book(x["verse"]))
    test_ds = train_data.filter(lambda x: not is_nt_book(x["verse"]))
    train_val_ds = train_ds.train_test_split(test_size=0.05, seed=41)
    dataset = DatasetDict({"train": train_val_ds["train"], "validation": train_val_ds["test"], "test": test_ds})
    print(dataset)
    return dataset


class FixedLabelSmoothingSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Workaround for an HF Trainer interaction where enabling label smoothing
    pops labels before the forward pass, which can lead to the model receiving
    both decoder_input_ids and decoder_inputs_embeds under some configs.

    This subclass keeps labels in the model inputs, runs the forward pass,
    and then applies the label smoother on the returned logits.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        if labels is not None and self.label_smoother is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss


def main(args):
    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    src_lang_nllb, tgt_lang_nllb = args.src_lang_nllb, args.tgt_lang_nllb
    dataset_name = args.dataset_name.split("/")[-1]
    output_dir = f"{args.model_name.split('/')[-1]}-{dataset_name}-{src_lang}-{tgt_lang}-{args.max_steps}"
    os.makedirs(output_dir, exist_ok=True)

    if dataset_name == "biblenlp-corpus":
        dataset = load_ebible_corpus(src_lang, tgt_lang)
    elif dataset_name == "alkitab-sabda-mt":
        dataset = load_alkitab_sabda(src_lang, tgt_lang)
    elif dataset_name == "ebible_local_ind_corpus":
        dataset = load_ebible_local_ind_corpus(src_lang, tgt_lang)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Add special tokens for source and target languages if they are not already present
    if src_lang_nllb not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": [src_lang_nllb]})

    if tgt_lang_nllb not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": [tgt_lang_nllb]})

    tokenizer.src_lang = src_lang_nllb
    tokenizer.tgt_lang = tgt_lang_nllb

    # Initialize new token embeddings with the mean of old embeddings
    old_embeddings = model.get_input_embeddings()
    old_num_tokens = old_embeddings.weight.size(dim=0)
    old_embeddings_mean = old_embeddings.weight.mean(dim=0, keepdim=True)

    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    embeddings.weight.data[old_num_tokens:, :] = old_embeddings_mean
    model.tie_weights()

    def preprocess_function(examples):
        return tokenizer(
            examples["source_text"],
            text_target=examples["target_text"],
            max_length=args.max_length,
            padding="max_length",
            truncation=True,
        )

    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=args.num_proc,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in the labels as we can't decode them.
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        sacrebleu_result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        chrf3_result = chrf.compute(predictions=decoded_preds, 
                                   references=decoded_labels,
                                   char_order=3,
                                   word_order=0)
        result = {"bleu": sacrebleu_result["score"], "chrf3": chrf3_result["score"]}
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    wandb.init(project="bible-nmt", name=output_dir)
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
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        predict_with_generate=True,
        generation_num_beams=args.num_beams,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=500,
        report_to="wandb",  # Logs to W&B
        logging_dir=f"{output_dir}/logs",
        run_name=output_dir,
    )

    # Save training configuration to output directory
    try:
        with open(os.path.join(output_dir, "training_args.json"), "w") as f:
            json.dump(training_args.to_dict(), f, indent=2, sort_keys=True)
        with open(os.path.join(output_dir, "cli_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True, default=str)
    except Exception as e:
        print(f"Warning: failed to save training config JSONs: {e}")

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        )
    ] if args.early_stopping_patience else None

    trainer = FixedLabelSmoothingSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()
  
    trainer.save_model()
    trainer.create_model_card()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
