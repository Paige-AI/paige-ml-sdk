import sys
from pathlib import Path

from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from paige.ml_sdk.__main__ import main  # noqa: E999


class TestCLI:
    def test_cli_should_run_with_BinClsAgata(
        self,
        capsys: CaptureFixture,
        monkeypatch: MonkeyPatch,
        f_path_to_dataset_csv: Path,
        f_path_to_embeddings_dir: Path,
    ) -> None:
        test_args = [
            '/home/aicompute/development/ai-core/lib/ml-sdk/src/paige/ml_sdk/__main__.py',
            '--model',
            'BinClsAgata',
            '--data.tune_dataset_path',
            str(f_path_to_dataset_csv),
            '--data.train_embeddings_dir',
            str(f_path_to_embeddings_dir),
            '--data.train_dataset_path',
            str(f_path_to_dataset_csv),
            '--data.embeddings_filename_column',
            'image_uri',
            '--model.label_names',
            '[cancer,precursor]',
            '--data.label_columns',
            '[cancer,precursor]',
            '--model.in_features',
            '5',
            '--model.layer1_out_features',
            '10',
            '--model.layer2_out_features',
            '10',
        ]
        monkeypatch.setattr(sys, 'argv', test_args)
        main()
        captured = capsys.readouterr()
        assert captured.err == ''

    def test_cli_should_run_with_MultiClsAgata(
        self,
        capsys: CaptureFixture,
        monkeypatch: MonkeyPatch,
        f_path_to_dataset_csv: Path,
        f_path_to_embeddings_dir: Path,
    ) -> None:
        test_args = [
            '/home/aicompute/development/ai-core/lib/ml-sdk/src/paige/ml_sdk/__main__.py',
            '--model',
            'MultiClsAgata',
            '--data.tune_dataset_path',
            str(f_path_to_dataset_csv),
            '--data.train_embeddings_dir',
            str(f_path_to_embeddings_dir),
            '--data.train_dataset_path',
            str(f_path_to_dataset_csv),
            '--data.embeddings_filename_column',
            'image_uri',
            '--model.label_names',
            '[precursor,grade]',
            '--model.n_classes',
            '[2,3]',
            '--data.label_columns',
            '[precursor,grade]',
            '--model.in_features',
            '5',
            '--model.layer1_out_features',
            '10',
            '--model.layer2_out_features',
            '10',
        ]
        monkeypatch.setattr(sys, 'argv', test_args)
        main()
        captured = capsys.readouterr()
        assert captured.err == ''
