// The Wikitables data is available at https://ppasupat.github.io/WikiTableQuestions/
{
  "random_seed": 4536,
  "numpy_seed": 9834,
  "pytorch_seed": 953,
  "dataset_reader": {
    "type": "wikitables",
    "tables_directory": "./data/WikiTableQuestions/",
    "offline_logical_forms_directory": "./tmp/offline_search_output/",
    "max_offline_logical_forms": 60,
  },
  "validation_dataset_reader": {
    "type": "wikitables",
    "tables_directory": "./data/WikiTableQuestions/",
    "keep_if_no_logical_forms": true,
  },
  "vocabulary": {
    "min_count": {"tokens": 3},
    "tokens_to_add": {"tokens": ["-1"]}
  },
  // "train_data_path": "./data/WikiTableQuestions/data/random-split-1-train.examples",
  "train_data_path": "./data/WikiTableQuestions/data/random-split-1-dev.examples",
  "validation_data_path": "./data/WikiTableQuestions/data/random-split-1-dev.examples",
  "model": {
    "type": "wikitables_mml_parser",
    "question_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 200,
          "trainable": true
        }
      },
    },
    "action_embedding_dim": 100,
    "encoder": {
      "type": "lstm",
      "input_size": 400,
      "hidden_size": 100,
      "bidirectional": true,
      "num_layers": 1,
    },
    "entity_encoder": {
      "type": "boe",
      "embedding_dim": 200,
      "averaged": true
    },
    "decoder_beam_search": {
      "type": "evolutionary-search",
    },
    "max_decoding_steps": 16,
    "attention": {
      "type": "bilinear",
      "vector_dim": 200,
      "matrix_dim": 200
    },
    "dropout": 0.5
  },
  "data_loader": {
    "batch_size": 10,
  },
  "validation_data_loader": {
    "batch_size": 10
  },
  "trainer": {
    "num_epochs": 100,
    "patience": 10,
    "cuda_device": -1,
    "grad_norm": 5.0,
    "validation_metric": "+denotation_acc",
    "optimizer": {
      "type": "sgd",
      "lr": 0.1
    },
    "learning_rate_scheduler": {
      "type": "exponential",
      "gamma": 0.99
    },
    "callbacks": [
      {
        "type": "tensorboard"
      }
    ]
  }
}
