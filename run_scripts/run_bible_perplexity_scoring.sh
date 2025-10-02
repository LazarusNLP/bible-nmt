#!/bin/bash
# Run the Bible perplexity scorer
python scripts/bible_perplexity_scorer.py \
    --english_corpus_dir "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/eng/corpus" \
    --target_corpus_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/nfa-nfa.txt" \
    --vref_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/vref.txt" \
    --giza_py_path "/data/projects/punim0478/setiawand/giza-py" \
    --output_dir "./results/perplexity_scoring" \
    --output_file "bible_perplexity_scores.csv" \
    --debug \
