from typing import Dict, Set

from lightning.pytorch.cli import LightningCLI


class AggregatorCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Both the datamodule and the model require the same list of labels as input
        # But providing them as arguments twice is combersome and error prone. We use
        # `link_arguments` so that users only must specify `--data.label_columns`; the value will
        # get replicate automatically for --model.label_names.
        # See https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_expert.html#argument-linking
        parser.link_arguments("data.label_columns", "model.init_args.label_names")

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        """Defines the list of available subcommands and the arguments to skip.
        we override this method in order to add a `fit_and_test` command, which
        runs `fit` followed by `test` on the best checkpoint from `fit`.
        """
        _subcommands = LightningCLI.subcommands()
        _subcommands['fit_and_test'] = {'model', 'datamodule', 'ckpt_path'}
        return _subcommands
