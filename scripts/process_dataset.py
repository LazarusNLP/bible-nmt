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

source_lang = "ind" 
target_lang = ["aaz", "ptu", "nfa", "heg", "lex", "row", "llg", "rgu", "txq", "tet", "wrs"]

from datasets import load_dataset, DatasetDict, concatenate_datasets


def load_ebible_corpus(src_lang, tgt_lang):
    def process_translations(x):
        """Process translations to handle multiple references and concatenate by language"""
        languages = x["translation"]["language"]
        translations = x["translation"]["translation"]
        
        # Group translations by language
        lang_translations = {}
        for lang, translation in zip(languages, translations):
            if lang not in lang_translations:
                lang_translations[lang] = []
            lang_translations[lang].append(translation)
        
        # Concatenate translations for each language with space separator
        src_text = " ".join(lang_translations.get(src_lang, [""]))
        tgt_text = " ".join(lang_translations.get(tgt_lang, [""]))
        return {"text_source": src_text, "text_target": tgt_text}
    
    dataset = load_dataset("bible-nlp/biblenlp-corpus", languages=[src_lang, tgt_lang], trust_remote_code=True)
    dataset = dataset.map(process_translations)
    # OT books for testing, NT books for training and validation
    # Handle both single refs and multiple refs
    def is_nt_book(refs):
        if isinstance(refs, list):
            # Check if any ref belongs to NT books
            return any(ref.split()[0] in NT_BOOKS for ref in refs)
        else:
            # Single reference
            return refs.split()[0] in NT_BOOKS
    
    # The dataset is a DatasetDict, so we need to access the 'train' split
    ds = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    train_ds = ds.filter(lambda x: is_nt_book(x["ref"]))
    test_ds = ds.filter(lambda x: not is_nt_book(x["ref"]))
    train_val_ds = train_ds.train_test_split(test_size=0.1, seed=41)
    dataset = DatasetDict({"train": train_val_ds["train"], "validation": train_val_ds["test"], "test": test_ds})
    return dataset

# Process and push each language pair as separate configs (subsets) to ONE repo
REPO_NAME = "biblenlp-corpus"

print(f"🏗️  Creating dataset with {len(target_lang)} subsets in ONE repository: {REPO_NAME}")
print(f"📊 Each subset will have train/validation/test splits")
print(f"🔧 Subsets: {[f'{source_lang}_{lang}' for lang in target_lang]}")
print()

for i, tgt_lang in enumerate(target_lang):
    print(f"\n[{i+1}/{len(target_lang)}] Processing {source_lang} -> {tgt_lang}...")
    
    try:
        dataset = load_ebible_corpus(source_lang, tgt_lang)
        
        # Create config name (use underscore, not hyphen!)
        config_name = f"{source_lang}_{tgt_lang}"
        
        print(f"  Dataset structure:")
        for split_name, split_data in dataset.items():
            print(f"    {split_name}: {len(split_data):,} examples")
        
        # Push to hub with config name - this avoids split name issues!
        dataset.push_to_hub(
            repo_id=REPO_NAME,
            config_name=config_name,  # Each language pair = separate config
            private=False
        )
        
        print(f"  ✅ Successfully pushed {config_name}")
        
    except Exception as e:
        print(f"  ❌ Error processing {tgt_lang}: {e}")
        continue

print(f"\n🎉 All done! Dataset available at: https://huggingface.co/datasets/{REPO_NAME}")
print(f"\n📋 FINAL STRUCTURE:")
print(f"   Repository: {REPO_NAME}")
print(f"   Subsets: {len(target_lang)} language pairs")
print(f"   Each subset has: train, validation, test splits")
print(f"\n💡 Usage examples:")
print(f'   dataset = load_dataset("{REPO_NAME}", "ind_tet")  # Indonesian → Tetum')
print(f'   dataset = load_dataset("{REPO_NAME}", "ind_aaz")  # Indonesian → Amarasi')
print(f'   all_data = load_dataset("{REPO_NAME}")           # All {len(target_lang)} subsets')
