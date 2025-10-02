iso_codes=(englsv engojb t4t)

for code in "${iso_codes[@]}"; do
    echo "Aligning NT and OT for target language: $code"
    python scripts/create_aligned_nt_ot.py \
        --source_corpus_path ebible-corpus/eng/corpus/eng-${code}.txt \
        --target_corpus_path ebible-corpus/dhao-eng/nfa-nfa.txt \
        --vref_path ebible-corpus/vref.txt \
        --src_lang eng \
        --tgt_lang "$code" \
        --output_dir ebible-corpus/dhao-eng/${code} \
        --output_prefix aligned
done