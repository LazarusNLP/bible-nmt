#!/bin/bash
export GEMINI_API_KEY="AIzaSyC0N8M7QkePL3pOlVYchXzecwkfEoYi_D0"
# vLLM Post-Editing using glossary/lexicon mode
# Demonstrates the new glossary-based few-shot prompting functionality

# Set working directory to project root
cd /data/projects/punim0478/setiawand/bible-nmt

# Change to post-editing directory for imports to work correctly
cd src/post-editing

# Generate timestamp for unique output directories
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT="../../results/run_lexicon_${TIMESTAMP}"

echo "Starting vLLM post-editing experiments with glossary/lexicon mode: ${TIMESTAMP}"
echo "Results will be saved to: ${BASE_OUTPUT}"

# python run_post_editing.py \
#   --model_type "vllm" \
#   --model_name "MISHANM/google-gemma-3-27b-it-fp8" \
#   --csv_path /data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao/aligned-ind-dhao-ot.csv \
#   --few_shot_corpus_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao/aligned-ind-dhao-nt.csv" \
#   --src "id" \
#   --tgt "nfa" \
#   --src_lang_name "Indonesian" \
#   --tgt_lang_name "Dhao" \
#   --output_dir "${BASE_OUTPUT}/gemma-3-27b-it-glossary-giza" \
#   --prompt "post_editing" \
#   --glossary_mode "smart" \
#   --glossary_path "/data/projects/punim0478/setiawand/bible-nmt/dictionary/dhao_ind_dictionary.csv" \
#   --few_shot_mode "both" \
#   --vectorizer bm25 \
#   --num_few_shot 5 \
#   --max_samples 10 \
#   --debug

python run_post_editing.py \
  --model_type "gemini" \
  --model_name "gemini-2.5-flash" \
  --csv_path /data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao/aligned-ind-dhao-ot.csv \
  --few_shot_corpus_path "/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao/aligned-ind-dhao-nt.csv" \
  --src "id" \
  --tgt "nfa" \
  --src_lang_name "Indonesian" \
  --tgt_lang_name "Dhao" \
  --output_dir "${BASE_OUTPUT}/gemma-3-27b-it-glossary-giza" \
  --prompt "post_editing" \
  --few_shot_mode "parallel" \
  --vectorizer bm25 \
  --num_few_shot 5 \
  --max_samples 3 \
  --debug \
  # --glossary_path "/data/projects/punim0478/setiawand/bible-nmt/dictionary/dhao_ind_dictionary.csv" \
  # --glossary_mode "smart" \


# Return to project root
cd ../..
