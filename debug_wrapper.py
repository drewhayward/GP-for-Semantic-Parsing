import json
import shutil
import sys
from allennlp.commands import main
from overrides.overrides import overrides

overrides = json.dumps({
    'model.decoder_beam_search': {
        "type": 'evolutionary-search',
        "skip_failures": False,
        "num_generations": 20,
        "pop_size": 100,
        "pop_lambda": 75,
    }
})


# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "evaluate",
    "experiments/models/evosearch/",
    # "data/WikiTableQuestions/data/training-before300.examples",
    "data/WikiTableQuestions/data/random-split-1-dev.examples",
    "-o", overrides
]

main()