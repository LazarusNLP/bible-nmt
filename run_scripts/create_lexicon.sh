
fast_align -i ebible-corpus/luang/aligned-eng-luang-all.txt -d -o -v -p ebible-corpus/luang/conditional_probs.txt

python scripts/create_lexicon_from_conditional_probs.py \
    ebible-corpus/luang/conditional_probs.txt \
    ebible-corpus/luang/ot_words_confidence_score.csv \
    --include-null \
    --exclude-punctuation