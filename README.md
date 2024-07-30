# Paige ML SDK (Alpha)

This package provides tools for training supervised machine learning models for computational pathology tasks using tile-level embeddings.

This repository accompanies the publication of the Paige Virchow foundation model suite. If you use this repository for academic research, please cite the following paper:
Vorontsov, E., Bozkurt, A., Casson, A. et al. "A foundation model for clinical-grade computational pathology and rare cancers detection." Nat Med (2024). https://doi.org/10.1038/s41591-024-03141-0

This repository is intended strictly for non-commercial academic research use. For commercial collaborations, please contact Paige AI under the appropriate terms. For detailed terms, please refer to our LICENSE.

## Installation
`cd paige-ml-sdk; pip install -e .`

## Getting Started
The SDK is equipped with a cli which can be used to train models and run inference. Run `python -m paige.ml_sdk --help` to get started, or refer to the [examples](https://github.com/Paige-AI/paige-ml-sdk/tree/main/examples) directory for a basic tutorial illustrating how to use the SDK and how to organize your data.

## Advanced Usage

The SDK and its CLI are powered by [pytorch lightning](https://lightning.ai/docs/pytorch/stable/) which has many customizeable features.

### Changing Loggers

By default, the sdk relies on Lightning's [CSV logger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.CSVLogger.html#csvlogger) which writes outputs to the `lightning_logs` folder. Use `--trainer.logger` and choose from any of Lightning's [built-in loggers](https://lightning.ai/docs/pytorch/stable/extensions/logging.html#supported-loggers) such as WandB or TensorBoard.

### Changing Optimizers and Adjusting Learning Rate
Use `--optimizer` and choose from any of the [torch.optim](https://pytorch.org/docs/stable/optim.html#algorithms) optimizers. Similarly, use `--lr_scheduler` to adjust the learning rate.

### Customizing Callbacks

Per pytorch lightning's [documentation](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html):

> Callbacks allow you to add arbitrary self-contained programs to your training. At specific points during the flow of execution (hooks), the Callback interface allows you to design programs that encapsulate a full set of functionality. It de-couples functionality that does not need to be in the lightning module and can be shared across projects.

> Lightning has a callback system to execute them when needed. Callbacks should capture NON-ESSENTIAL logic that is NOT required for your lightning module to run.

Callbacks can be used for many things, like saving model checkpoints, profiling exectution, or early stopping. By default, the CLI only applies a single callback which saves model checkpoints. Users wishing to configure this callback differently, or add additional callbacks may consider the options offered here: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html#trainer-callbacks-and-arguments-with-class-type

### Changing Arbitrary Trainer Args

Lightning's [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) class automates most of the complexity surrounding model training. It handles the training strategy (e.g., DDP, FSDP), the choice of hardware (cpu, gpu), training precision, the number of epochs to train for, and much more. Most common practice tricks in AI engineering can be configured via the Trainer class. All of the available trainer flags are documented in the cli's help text:

```bash
python -m paige.ml_sdk fit --help
```

for example, to train a model in 16-bit precision, set `--trainer.precision 16`.











