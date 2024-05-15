# Paige ML SDK

This package equips users with tools to train supervised machine learning models using foundation model embeddings

## Installation

`cd paige-ml-sdk; pip install -e .`

## Assumptions

### Dataset

Train, tune, and test datasets must be organized in separate csv files, each containing at least 2 columns:
1. One or more columns specifying the target values.
2. A column containing the names of the slides. This column will be used to lookup embedding files

For example, the table below specifies a small dataset with two slides, and two target columns:
|  slidename | cancer | grade |
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
>>> print(t['embeddings'].shape)  # suppose the slide has 136 tiles
torch.Size([143, 2560])
```
## CLI Usage

### Data Args

Any of the arguments to   `paige.ml_sdk.dataset_universe.datamodule.init_datamodule_from_dataset_filepaths` may be provided to the CLI. Let's take a look:

- `--data.train_dataset_path`: A path to the train dataset csv
- `--data.tune_dataset_path`: A path to the tune dataset csv
- `--data.label_columns`: The names of the columns which will be used as labels
- `--data.embeddings_filename_column`: The column containing the embedding filenames. This might be the column containing the slide names, if embedding files and slides have the same names (with different extensions).
- `--data.train_embeddings_dir`: directory containing train embeddings

The following arguments are optional
- `--data.test_dataset_path`: A path to the test dataset csv 
- `--data.tune_embeddings_dir`: directory containing tune embeddings. Defaults to train_embeddings_dir.
- `--data.test_embeddings_dir`: directory containing test embeddings. Defaults to train_embeddings_dir
- `--data.label_missing_value`: Missing label value. Defaults to -999
- `--data.group_col`: The column to use if embeddings from multiple slides belong to a common group and are meant to be loaded together. Useful, for example, when training a model where labels are only accurate at a block, part, or case level.
- `--data.num_workers`: Number of dataloader workers. Defaults to 0
- `--data.batch_size`: Batch size. Defaults to 1

### Model Args

Users may specify any of the model classes defined in `paige.ml_sdk.model_universe.algos` Currently there are 3:
- `BinClsAgata`: Trains an `Agata` model with one or more binary heads (labels)
- `MultiClsAgata`: Trains an `Agata` model with one or more multiclass heads.
- `BinClsPerceiver` Trains a `Perceiver` model with one or more binary heads.

The model is specified via the `--model` argument, e.g., `--model BinClsAgata`
The other arguments will depend on the arguments available to the model class's `__init__` method. For example, for `BinClsAgata`, the arguments are as follows:

- `--model.label_names`: List of label names, to be used as model heads. Must match the values provided to `--data.label_columns`
- `--model.in_features`: Dimensionality of the embeddings. Defaults to 2560
- `--model.layer1_out_features`: Dimension of model's first linear layer. Defaults to 512
- `--model.layer2_out_features`: Dimension of model's second linear layer. Defaults to  512
- `--model.activation`: Activation function to use. Defaults to ReLU
- `--model.scaled_attention`: Whether or not to use scaled attention. Defaults to False.
- `--model.absolute_attention`: Whether or not to use absolute attention. Defaults to False
- `--model.n_attention_queries`: Number of attention queries. Defaults to 1
- `--model.padding_indicator`:  Value that indicates padding positions in the embeddings mask. Defaults to 1.
- `--model.missing_label_value`: Missing label value. Defaults to -999.


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
alice@hpc:~$ head config.yaml
data:
	train_dataset_path: dataset.csv
	tune_dataset_path: dataset.csv
	train_embeddings_dir: .
	label_columns:
		- precursor
		- cancer
	embeddings_filename_column: slide
...
```
The experiment can also be run from the config via the `--config` argument:
```bash
python -m paige.ml_sdk --config config.yaml
```









