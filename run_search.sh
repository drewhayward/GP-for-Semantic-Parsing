#!/bin/bash --login

#echo "Testing 123"
./venv/bin/python scripts/wikitables/search_for_logical_forms.py data/WikiTableQuestions \
    ./data/WikiTableQuestions/data/random-split-1-train.examples \
    tmp/offline_search_output \
    --output-separate-files \
    --num-splits 12
