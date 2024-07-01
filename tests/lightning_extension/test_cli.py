from paige.ml_sdk.lightning_extension.cli import AggregatorCLI


class TestAggregatorCLI:
    def test_subcommands_should_contain_fit_and_test(self) -> None:
        subcommands = AggregatorCLI.subcommands()
        assert subcommands.keys() == {'fit', 'validate', 'predict', 'test', 'fit_and_test'}
