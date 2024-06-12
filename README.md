# Paige ML SDK (Alpha)

This package provides tools for training supervised machine learning models for computational pathology tasks using tile-level embeddings.

This repository accompanies the publication of the Paige Virchow foundation model suite. If you use this repository for academic research, please cite the following paper:
Vorontsov, Eugene et al. “Virchow: A Million-Slide Digital Pathology Foundation Model.” ArXiv abs/2309.07778 (2023).

This repository is intended strictly for non-commercial academic research use. For commercial collaborations, please contact Paige AI under the appropriate terms. For detailed terms, please refer to our LICENSE.

## Installation

`cd paige-ml-sdk; pip install -e .`

## Assumptions

### Dataset

Train, tune, and test datasets must be organized in separate csv files, each containing at least 2 columns:

1. One or more columns specifying the target values.
2. A column containing the names of the slides. This column will be used to lookup embedding files

For example, the table below specifies a small dataset with two slides, and two target columns:

| slidename | cancer | grade |
|--|--|--|
| slide_1.svs | 1 | 2 |
| slide_2.svs | 0 | -999 |

Missing label values may be encoded by the integer value of your choice. The above example uses -999.

### Embeddings

Embedding files must all be present under a single directory. For example:

```
embeddings_directory
|——slide_1.svs.pt
|——slide_2.svs.pt
```

Embedding files are expected to contain dictionaries with an `embeddings` key, whose value should be a tensor of shape `n_embeddings, embedding_size`. For example:

```python
>>> import torch
>>> t = torch.load('slide.pt')
>>> print(t['embeddings'].shape) # suppose the slide has 136 tiles
torch.Size([143, 2560])
```

## CLI Usage
### Data Args

Any of the arguments to `paige.ml_sdk.dataset_universe.datamodule.init_datamodule_from_dataset_filepaths` may be provided to the CLI. Let's take a look:

-  `--data.train_dataset_path`: A path to the train dataset csv
-  `--data.tune_dataset_path`: A path to the tune dataset csv
-  `--data.label_columns`: The names of the columns which will be used as labels
-  `--data.embeddings_filename_column`: The column containing the embedding filenames. This might be the column containing the slide names, if embedding files and slides have the same names (with different extensions).
-  `--data.train_embeddings_dir`: directory containing train embeddings

The following arguments are optional

-  `--data.test_dataset_path`: A path to the test dataset csv
-  `--data.tune_embeddings_dir`: directory containing tune embeddings. Defaults to train_embeddings_dir.
-  `--data.test_embeddings_dir`: directory containing test embeddings. Defaults to train_embeddings_dir
-  `--data.label_missing_value`: Missing label value. Defaults to -999
-  `--data.group_col`: The column to use if embeddings from multiple slides belong to a common group and are meant to be loaded together. Useful, for example, when training a model where labels are only accurate at a block, part, or case level.
-  `--data.num_workers`: Number of dataloader workers. Defaults to 0
-  `--data.batch_size`: Batch size. Defaults to 1
 
### Model Args

Users may specify any of the model classes defined in `paige.ml_sdk.model_universe.algos` Currently there are 3:

-  `BinClsAgata`: Trains an `Agata` model with one or more binary heads (labels)
-  `MultiClsAgata`: Trains an `Agata` model with one or more multiclass heads.
-  `BinClsPerceiver` Trains a `Perceiver` model with one or more binary heads.

The model is specified via the `--model` argument, e.g., `--model BinClsAgata`. The other arguments will depend on the arguments available to the model class's `__init__` method. For example, for `BinClsAgata`, the arguments are as follows:

-  `--model.label_names`: List of label names, to be used as model heads. Must match the values provided to `--data.label_columns`
-  `--model.in_features`: Dimensionality of the embeddings. Defaults to 2560
-  `--model.layer1_out_features`: Dimension of model's first linear layer. Defaults to 512
-  `--model.layer2_out_features`: Dimension of model's second linear layer. Defaults to 512
-  `--model.activation`: Activation function to use. Defaults to ReLU
-  `--model.scaled_attention`: Whether or not to use scaled attention. Defaults to False.
-  `--model.absolute_attention`: Whether or not to use absolute attention. Defaults to False
-  `--model.n_attention_queries`: Number of attention queries. Defaults to 1
-  `--model.padding_indicator`: Value that indicates padding positions in the embeddings mask. Defaults to 1.
-  `--model.missing_label_value`: Missing label value. Defaults to -999.

### Putting it all together

See `paige-ml-sdk/examples` for an example of how to train a simple aggregator on a dummy dataset.

#### From CLI args

A typical command might look like this:
```
python -m paige.ml_sdk --data.train_dataset_path train.csv --data.tune_dataset_path tune.csv --data.train_embeddings_dir ./embeddings --data.label_columns [cancer,precursor] --data.embeddings_filename_column slidename --model BinClsAgata --model.label_names [cancer,precursor]
``` 

#### From a config file

Specifying many cli args can be cumbersome. Experiments can also be configured from a yaml configuration file. When the command in the previous section is invoked, Lightning will create a yaml file in the current directory which offers an alternative way of configuring the experiment. The file will look something like this:

```bash
alice@hpc:~$  head  config.yaml
data:
train_dataset_path:  dataset.csv
tune_dataset_path:  dataset.csv
train_embeddings_dir:  .
label_columns:
-  precursor
-  cancer
embeddings_filename_column:  slide
...
```

The experiment can also be run from the config via the `--config` argument:
```bash
python  -m  paige.ml_sdk  --config  config.yaml
```

## Advanced Usage
### Customizing callbacks

By default, the CLI only applies a single callback which saves model checkpoints. Users wishing to configure this callback differently, or add additional callbacks, may consider the options offered here: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html#trainer-callbacks-and-arguments-with-class-type

### Changing Loggers

Use `--trainer.logger` and choose from any of the logger classes defined at https://lightning.ai/docs/pytorch/stable/extensions/logging.html

### Changing Optimizer or Learning Rate
Use `--optimizer` and choose from any of the optimizers defined at https://pytorch.org/docs/stable/optim.html
Similarly, use `--lr_scheduler` to adjust the learning rate.

### Changing Arbitrary Trainer Args
More generally, any arguments available to  Lightning's `Trainer`
class can be configured via `--trainer.xxx`. For all the configurable trainer args, see the CLI's help text:

```
alice@hpc:~$ python -m paige.ml_sdk

usage: __main__.py [-h] [-c CONFIG] [--print_config[=flags]] [--seed_everything SEED_EVERYTHING] [--trainer CONFIG]
[--trainer.accelerator.help CLASS_PATH_OR_NAME] [--trainer.accelerator ACCELERATOR]
[--trainer.strategy.help CLASS_PATH_OR_NAME] [--trainer.strategy STRATEGY] [--trainer.devices DEVICES]
[--trainer.num_nodes NUM_NODES] [--trainer.precision PRECISION] [--trainer.logger.help CLASS_PATH_OR_NAME]
[--trainer.logger LOGGER] [--trainer.callbacks.help CLASS_PATH_OR_NAME] [--trainer.callbacks CALLBACKS]
[--trainer.fast_dev_run FAST_DEV_RUN] [--trainer.max_epochs MAX_EPOCHS] [--trainer.min_epochs MIN_EPOCHS]
[--trainer.max_steps MAX_STEPS] [--trainer.min_steps MIN_STEPS] [--trainer.max_time MAX_TIME]
[--trainer.limit_train_batches LIMIT_TRAIN_BATCHES] [--trainer.limit_val_batches LIMIT_VAL_BATCHES]
[--trainer.limit_test_batches LIMIT_TEST_BATCHES] [--trainer.limit_predict_batches LIMIT_PREDICT_BATCHES]
[--trainer.overfit_batches OVERFIT_BATCHES] [--trainer.val_check_interval VAL_CHECK_INTERVAL]
[--trainer.check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--trainer.num_sanity_val_steps NUM_SANITY_VAL_STEPS]
[--trainer.log_every_n_steps LOG_EVERY_N_STEPS] [--trainer.enable_checkpointing {true,false,null}]
[--trainer.enable_progress_bar {true,false,null}] [--trainer.enable_model_summary {true,false,null}]
[--trainer.accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--trainer.gradient_clip_val GRADIENT_CLIP_VAL]
[--trainer.gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM] [--trainer.deterministic DETERMINISTIC]
[--trainer.benchmark {true,false,null}] [--trainer.inference_mode {true,false}]
[--trainer.use_distributed_sampler {true,false}] [--trainer.profiler.help CLASS_PATH_OR_NAME]
[--trainer.profiler PROFILER] [--trainer.detect_anomaly {true,false}] [--trainer.barebones {true,false}]
[--trainer.plugins.help CLASS_PATH_OR_NAME] [--trainer.plugins PLUGINS] [--trainer.sync_batchnorm {true,false}]
[--trainer.reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS]
[--trainer.default_root_dir DEFAULT_ROOT_DIR] [--model.help CLASS_PATH_OR_NAME]
--model CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE [--data CONFIG]
--data.train_dataset_path TRAIN_DATASET_PATH --data.tune_dataset_path TUNE_DATASET_PATH
--data.train_embeddings_dir TRAIN_EMBEDDINGS_DIR --data.label_columns [ITEM,...]
--data.embeddings_filename_column EMBEDDINGS_FILENAME_COLUMN [--data.test_dataset_path TEST_DATASET_PATH]
[--data.tune_embeddings_dir TUNE_EMBEDDINGS_DIR] [--data.test_embeddings_dir TEST_EMBEDDINGS_DIR]
[--data.label_missing_value LABEL_MISSING_VALUE] [--data.group_col GROUP_COL]
[--data.filename_extension FILENAME_EXTENSION] [--data.num_workers NUM_WORKERS] [--data.batch_size BATCH_SIZE]
[--data.mode {csv,parquet}] [--optimizer.help CLASS_PATH_OR_NAME]
[--optimizer CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE] [--lr_scheduler.help CLASS_PATH_OR_NAME]
[--lr_scheduler CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE]
```











