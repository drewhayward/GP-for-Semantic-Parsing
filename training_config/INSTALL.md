
# Install Steps

Make venv & Install requirements
```
python3 -m virtualenv .venv --python=python3
pip3 install -r requirements.text
```

## NLTK 404
For some reason NLTK might have an SSL error when downloading wordnet. You can
use the following snippet to manually install whatever corpus/package is failing.
It will open a GUI to select whatever packages you would like to download.

```python
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
```

## Data
Download the data zip: https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip
and put the extracted folder in `./data`

### Preprocess
```
PYTHONPATH=. python3 scripts/wikitables/search_for_logical_forms.py data/WikiTableQuestions \
    ./data/WikiTableQuestions/data/random-split-1-train.examples \
    tmp/offline_search_output \
    --output-separate-files
```

## Entrypoint

The basic command to start the model training

`allennlp train <CONFIG FILE> -s <OUTPUT DIR>`

So for our testing
`allennlp train ./training_config/wikitables_mml_parser.jsonnet -s ./tmp/debug`

You have to make sure the output directory is empty otherwise it throws an error.