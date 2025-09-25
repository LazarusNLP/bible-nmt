
fast_align -i ebible-corpus/dhao/aligned-ind-nfa-ot.txt -d -o -v -p ebible-corpus/dhao/conditional_probs.txt

python scripts/create_lexicon_from_conditional_probs.py \
    ebible-corpus/dhao/conditional_probs.txt \
    ebible-corpus/dhao/ot_words_confidence_score.csv \
    --include-null \
    --exclude-punctuation