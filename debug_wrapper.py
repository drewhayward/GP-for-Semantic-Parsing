import json
import shutil
import sys
from allennlp.commands import main

#config_file = "training_configs/gqa_spatial_blitz.jsonnet"
config_file = "training_config/wikitables_mml_parser.jsonnet"

# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": -1}})

serialization_dir = "./experiments/debugger_train"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!

# shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "evaluate",
    "experiments/models/evosearch/",
    "data/WikiTableQuestions/data/random-split-1-dev.examples"
    # config_file,
    # "-s", serialization_dir,
    # "--include-package", "allennlp_semparse"
    # "-o", overrides,
]

main()