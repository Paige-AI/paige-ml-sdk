# Getting Started

This example trains a binary classifier in a supervised manner on a small synthetic dataset. To skip straight to running the example yourself, run [cmd.sh](https://github.com/Paige-AI/paige-ml-sdk/blob/main/examples/cmd.sh) or `python -m paige.ml_sdk --config config.yaml`.

## Experiment Setup

### Embeddings
The files `slide_1.svs.pt`, `slide_2.svs.pt`, `slide_3.svs.pt`, and `slide_4.svs.pt` contain simulated embeddings for four hypothetical slides.

```python
>>> import torch
>>> t = torch.load('slide_4.svs.pt')
>>> print(t['embeddings'].shape)
torch.Size([3, 5])  # 3 embeddings of length 5
```
Since this example was designed to run quickly, the dimensionality of the data is low. In reality embeddings will be larger.

### Dataset
This ground truth (referred to synonymously as 'labels' or 'targets') used for training supervision is specified in the `dataset.csv` file:

|slide|cancer|precursor|
|--|--|--|
|slide_1.svs|0|0|
|slide_2.svs|1|-999|
|slide_3.svs|0|0|
|slide_4.svs|1|1|

In this example, each slide has a name and two binary values: one indicating the presence or absence of precursor lesions, and the other indicating the same for cancer. 

#### Missing Labels
Missing labels are encoded by the integer -999, has evidenced by slide 2 in the dataset, which is missing a precursor label.

#### Dataset Splits
Train, tune, and test datasets should each go in separate csv files. Here, the same dataset.csv file is reused for each split for example purposes only.

## Training a Binary Classifier

We will train a multiheaded binary classifier with two prediction heads: one predicting whether or not a slide contains cancer, and the other doing the same for the precursor label. If the sdk is run with no arguments:

```bash
python -m paige.ml_sdk
```

an error is thrown with message `error: expected "subcommand" to be one of {fit,validate,test,predict}, but it was not provided`. Use the `fit` subcommand since to train a model:

```bash
python -m paige.ml_sdk fit
```

The sdk now displays `error: Parser key "data.label_columns": Expected a <class 'set'>. Got value: None`. It expected a `--data.label_columns` argument. To see documention on what that argument does, run `--help`:

```bash
python -m paige.ml_sdk fit --help
```

After scrolling down, the help text shows `--data.label_columns [ITEM,...]: Dataset csv columns to use as targets. Must be consistent across train, tune, and test dataset csvs. (required, type: Set[str])`. Let's use `cancer` and `precursor`:

```bash
python -m paige.ml_sdk fit --data.label_columns [cancer,precursor]
```

The sdk now throws a different error, because `--data.embeddings_filename_column` is missing. We can repeat this process of adding required arguments until the sdk is satisfied. Let's stop halfway through, once all the data arguments have been specified:

```bash
python -m paige.ml_sdk \
fit \
--data.label_columns [cancer,precursor] \
--data.embeddings_filename_column slide \
--data.train_dataset_path dataset.csv \
--data.tune_dataset_path dataset.csv \
--data.embeddings_dir .
```
Now comes the matter of selecting which model to use. Users may choose from any of the models defined in `src/ml_sdk/model_universe/algos.py`. Two popular choices are Perceiver and Agata. Let's use Agata, which stands for "Aggregator with Attention". It's a simple model that combines linear layers and attention to produce slide-level predictions. Agata can be used for multihead classification or regression tasks; different flavors of Agata are implemented by different classes in `algos.py`. Let's use `BinClsAgata`, which performs binary classifcation. The final command will look like this:

```bash
python -m paige.ml_sdk \
fit \
--data.label_columns [cancer,precursor] \
--data.embeddings_filename_column slide \
--data.tune_dataset_path dataset.csv \
--data.train_dataset_path dataset.csv \
--data.embeddings_dir . \
--model BinClsAgata \
--model.in_features 5 \
--model.layer1_out_features 3 \
--model.layer2_out_features 3
```
`in_features`, `layer1_out_features`, and `layer2_out_features` are all optional arguments which we're overriding to make the network smaller so that in trains faster in this example. You can run this example too- it should only take a few seconds.

### Logging Experiment Outputs

The sdk uses pytorch lightning, whose default logger writes experiment outputs to the `lightning_logs` folder. If you ran the previous command, you should now see that folder:

```bash
find lightning_logs/
```
```
lightning_logs/
lightning_logs/version_0
lightning_logs/version_0/hparams.yaml
lightning_logs/version_0/metrics.csv
lightning_logs/version_0/config.yaml
lightning_logs/version_0/checkpoints
lightning_logs/version_0/checkpoints/epoch=1-step=8.ckpt
```

###  Experiment Specification via YAML Config

Besides logging checkpoints and metrics, the experiment's configuration is also logged as a yaml file:

```bash
cat lightning_logs/version_0/config.yaml
```
This is useful for traceability and reproducibility. An experiment can be re-run from a config file using the `--config` option:

```bash
python -m paige.ml_sdk --config lightning_logs/version_0/config.yaml
```
For experiments where configuration from the command line would be cumbersome, users may opt for this config-based approach instead.


## Running Inference: Predict, Test, and Validate

The sdk has three other commands besides `fit`: predict, test, and validate. These three commands are conceptually similar to each other as they are all used to run inference on a chosen checkpoint. They all switch the model into eval mode, disable gradients, and make a single pass through the dataloader. Let's take a look at `predict`:

```bash
python -m paige.ml_sdk \
predict \
--data.label_columns [cancer,precursor] \
--data.embeddings_filename_column slide \
--data.test_dataset_path dataset.csv \
--data.embeddings_dir . \
--model BinClsAgata \
--model.in_features 5 \
--model.layer1_out_features 3 \
--model.layer2_out_features 3 \
--ckpt_path ./lightning_logs/version_0/checkpoints/epoch=1-step=8.ckpt
```
This command is similar to the `fit` command from above, only it specifies a test dataset (instead of train and tune), and has a `--ckpt_path` arg which is used to specify the model to run inference on. 
















