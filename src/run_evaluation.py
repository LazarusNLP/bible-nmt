from argparse import ArgumentParser
import json
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datasets import load_dataset, DatasetDict, concatenate_datasets
import evaluate
import torch

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


def process_translations(x, src_lang: str, tgt_lang: str):
    """
    Select a single source/target translation per row and record chosen files.
    - Source is `src_lang`; if multiple versions exist, prefer '{src_lang}-{src_lang}.txt' (e.g., 'ind-ind.txt').
    - Target is `tgt_lang`; assumed unique in the row.
    Returns: text_source, text_target, source_file, target_file
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
        "text_source": source_text,
        "text_target": target_text,
        "source_file": source_file,
        "target_file": target_file,
    }
    

def load_ebible_corpus(src_lang, tgt_lang):
    dataset = load_dataset("bible-nlp/biblenlp-corpus", languages=[src_lang, tgt_lang], trust_remote_code=True)
    dataset = dataset.map(process_translations, fn_kwargs={"src_lang": src_lang, "tgt_lang": tgt_lang})
    dataset = dataset.map(lambda x: {"verse_id": str(x["ref"])})
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
    train_val_ds = train_ds.train_test_split(test_size=0.05, seed=41)
    dataset = DatasetDict({"train": train_val_ds["train"], "validation": train_val_ds["test"], "test": test_ds})
    return dataset


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["bible-nlp/biblenlp-corpus", "LazarusNLP/alkitab-sabda-mt"],
    )
    parser.add_argument("--src_lang", type=str, default="ind")
    parser.add_argument("--tgt_lang", type=str, default="btx")
    parser.add_argument("--src_lang_nllb", type=str, default="ind_Latn")
    parser.add_argument("--tgt_lang_nllb", type=str, default="btx_Latn")
    parser.add_argument("--dataset_split_name", type=str, default="test")
    parser.add_argument("--book_name", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--baseline", action="store_true", help="If set, use source text as predictions and write combined baseline.json")
    return parser.parse_args()


def main(args):
    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    src_lang_nllb, tgt_lang_nllb = args.src_lang_nllb, args.tgt_lang_nllb
    dataset_name = args.dataset_name.split("/")[-1]
    output_dir = "baseline" if args.baseline else args.model_name.split("/")[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset_name == "biblenlp-corpus":
        dataset = load_ebible_corpus(src_lang, tgt_lang)
    elif dataset_name == "alkitab-sabda-mt":
        dataset = load_alkitab_sabda(src_lang, tgt_lang)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    translator = None
    if not args.baseline:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        # Move model to GPU before creating the pipeline
        model = model.to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.src_lang = src_lang_nllb
        tokenizer.tgt_lang = tgt_lang_nllb

        translator = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            src_lang=src_lang_nllb,
            tgt_lang=tgt_lang_nllb,
        )

    # Original BLEU (Papineni et al., 2002)
    bleu = evaluate.load("bleu")
    # chrF++ score (character n-grams with word n-grams)
    chrf = evaluate.load("chrf")
    # SacreBLEU with SPM tokenizer for FLORES-200 evaluation (Goyal et al., 2022)
    sacrebleu = evaluate.load("sacrebleu")
    spbleu = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels, verse_ids = eval_preds["prediction"], eval_preds["target"], eval_preds["verse_id"]
        cleaned_preds, cleaned_labels = postprocess_text(preds, labels)

        # Original BLEU (Papineni et al., 2002)
        bleu_result = bleu.compute(predictions=cleaned_preds, references=cleaned_labels)
        sacrebleu_result = sacrebleu.compute(predictions=cleaned_preds, references=cleaned_labels)
        spbleu_result = spbleu.compute(predictions=cleaned_preds, references=cleaned_labels, tokenize="flores200")
        
        # chrF++ score
        chrf_result = chrf.compute(
            predictions=cleaned_preds,
            references=cleaned_labels,
            word_order=2,
        )
        
        # chrF3: set beta=3 and include word n-grams (word_order=2) like chrF++
        chrf3_result = chrf.compute(
            predictions=cleaned_preds,
            references=cleaned_labels,
            char_order=3,
            word_order=0
        )
        
        # spBLEU (Goyal et al., 2022) using SentencePiece tokenization
        eval_result = {
            "bleu": bleu_result["bleu"]*100,
            "sacrebleu": sacrebleu_result["score"],
            "spbleu": spbleu_result["score"],
            "chrf3": chrf3_result["score"],
            "chrf++": chrf_result["score"],
        }
        eval_result = {k: round(v, 4) for k, v in eval_result.items()}

        results = [
            {"verse_id": verse_id, "prediction": pred, "target": label}
            for verse_id, pred, label in zip(verse_ids, preds, labels)
        ]
        return {"eval_metrics": eval_result, "results": results}

    def infer(batch):
        try:
            if len(batch["text_source"]) == 0:
                return {
                    "verse_id": [],
                    "prediction": [],
                    "target": [],
                }
            if args.baseline:
                predictions = list(batch["text_source"])  # Echo source as prediction
            else:
                predictions = [
                    out["translation_text"]
                    for out in translator(
                        batch["text_source"],
                        batch_size=args.per_device_eval_batch_size,
                        max_length=args.max_length,
                        num_beams=2,
                    )
                ]
            # verse_id is now consistently named across both datasets
            return {
                "verse_id": batch["verse_id"],
                "prediction": predictions,
                "target": batch["text_target"],
            }
        except Exception as e:
            print(f"Error in inference: {e}")
            print(f"Batch contents: {batch}")  # Print batch contents on error
            batch_size = len(batch["text_source"]) if "text_source" in batch else 0
            empty_strings = [""] * batch_size
            return {
                "verse_id": empty_strings,
                "prediction": empty_strings,
                "target": empty_strings,
            }
        
    
    print(f"Evaluating {args.model_name} on {dataset_name} for {src_lang} to {tgt_lang}")
    combined_baseline = {}
    for split in ["validation", "test"]:
        split_ds = dataset[split]
        if len(split_ds) == 0:
            print(f"Warning: Dataset is empty for split {split} for this language code.")
            continue
        if args.book_name:
            split_ds = split_ds.filter(lambda x: x["book"] == args.book_name)
        results = split_ds.map(infer, batched=True, batch_size=args.per_device_eval_batch_size)
        output = compute_metrics(results)

        with open(f"{output_dir}/{split}-results.json", "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        if args.baseline:
            combined_baseline[split] = output

    if args.baseline and len(combined_baseline) > 0:
        with open(f"{output_dir}/{tgt_lang}-baseline.json", "w") as f:
            json.dump(combined_baseline, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
