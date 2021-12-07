from argparse import ArgumentParser
import json
import shutil
import sys
from allennlp.commands import main


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--num_generations', type=int)
    parser.add_argument('--pop_size', type=int)
    parser.add_argument('--pop_lambda', type=int)
    parser.add_argument('--init_tree_depth', type=int)
    parser.add_argument('--mutation_ratio', type=int)
    parser.add_argument('--tournament_k', type=int)
    
    args = parser.parse_args()

    options = {"model.decoder_beam_search":{"type": 'evolutionary-search'}}
    for arg, value in vars(args).items():
        if value is not None:
            options["model.decoder_beam_search"][arg] = value

    overrides = json.dumps(options)
    
    # Assemble the command into sys.argv
    sys.argv = [
        "allennlp",  # command name, not used by main
        "evaluate",
        "experiments/models/evosearch/",
        "data/WikiTableQuestions/data/random-split-1-dev.examples",
        "--overrides", overrides
    ]
    
    main()