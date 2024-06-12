import sys
import pytest
from pathlib import Path
from typing import List

from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from paige.ml_sdk.__main__ import main  # noqa: E999


@pytest.fixture
def bin_cls_agata_args() -> List[str]:
    return [
        '--model',
        'BinClsAgata',
        '--model.label_names',
        '[cancer,precursor]',
        '--model.in_features',
        '5',
        '--model.layer1_out_features',
        '10',
        '--model.layer2_out_features',
        '10',
    ]


@pytest.fixture
def data_args(f_path_to_embeddings_dir: Path) -> List[str]:
    return [
        '--data.embeddings_dir',
        str(f_path_to_embeddings_dir),
        '--data.embeddings_filename_column',
        'image_uri',
        '--data.label_columns',
        '[cancer,precursor]',
    ]


class TestCLI:
    def test_cli_should_run_fit_with_BinClsAgata(
        self,
        capsys: CaptureFixture,
        monkeypatch: MonkeyPatch,
        f_path_to_dataset_csv: Path,
        data_args: List[str],
        bin_cls_agata_args: List[str],
    ) -> None:
        test_args = [
            '/home/aicompute/development/ai-core/lib/ml-sdk/src/paige/ml_sdk/__main__.py',
            'fit',
            '--data.train_dataset_path',
            str(f_path_to_dataset_csv),
            '--data.tune_dataset_path',
            str(f_path_to_dataset_csv),
        ]
        test_args += data_args + bin_cls_agata_args

        monkeypatch.setattr(sys, 'argv', test_args)
        main()
        captured = capsys.readouterr()
        assert captured.err == ''

    def test_cli_should_run_validate_with_BinClsAgata(
        self,
        capsys: CaptureFixture,
        monkeypatch: MonkeyPatch,
        f_path_to_dataset_csv: Path,
        data_args: List[str],
        bin_cls_agata_args: List[str],
    ) -> None:
        test_args = [
            '/home/aicompute/development/ai-core/lib/ml-sdk/src/paige/ml_sdk/__main__.py',
            'validate',
            '--data.tune_dataset_path',
            str(f_path_to_dataset_csv),
        ]
        test_args += data_args + bin_cls_agata_args

        monkeypatch.setattr(sys, 'argv', test_args)
        main()
        captured = capsys.readouterr()
        assert captured.err == ''

    def test_cli_should_run_test_with_BinClsAgata(
        self,
        capsys: CaptureFixture,
        monkeypatch: MonkeyPatch,
        f_path_to_dataset_csv: Path,
        data_args: List[str],
        bin_cls_agata_args: List[str],
    ) -> None:
        test_args = [
            '/home/aicompute/development/ai-core/lib/ml-sdk/src/paige/ml_sdk/__main__.py',
            'test',
            '--data.test_dataset_path',
            str(f_path_to_dataset_csv),
            '--ckpt_path',
            '~/projects/paige-ml-sdk/tests/lightning_logs/version_42/checkpoints/epoch=1-step=6.ckpt',
        ]
        test_args += data_args + bin_cls_agata_args
        monkeypatch.setattr(sys, 'argv', test_args)
        main()
        captured = capsys.readouterr()
        assert captured.err == ''

    def test_cli_should_run_predict_with_BinClsAgata(
        self,
        capsys: CaptureFixture,
        monkeypatch: MonkeyPatch,
        f_path_to_dataset_csv: Path,
        data_args: List[str],
        bin_cls_agata_args: List[str],
    ) -> None:
        test_args = [
            '/home/aicompute/development/ai-core/lib/ml-sdk/src/paige/ml_sdk/__main__.py',
            'predict',
            '--data.test_dataset_path',
            str(f_path_to_dataset_csv),
            '--ckpt_path',
            '~/projects/paige-ml-sdk/tests/lightning_logs/version_42/checkpoints/epoch=1-step=6.ckpt',
        ]
        test_args += data_args + bin_cls_agata_args
        monkeypatch.setattr(sys, 'argv', test_args)
        main()
        captured = capsys.readouterr()
        assert captured.err == ''

    def test_cli_should_run_test_after_fit_with_best_checkpoint(
        self,
        capsys: CaptureFixture,
        monkeypatch: MonkeyPatch,
        f_path_to_dataset_csv: Path,
        data_args: List[str],
        bin_cls_agata_args: List[str],
    ) -> None:
        test_args = [
            '/home/aicompute/development/ai-core/lib/ml-sdk/src/paige/ml_sdk/__main__.py',
            'fit',
            '--data.train_dataset_path',
            str(f_path_to_dataset_csv),
            '--data.tune_dataset_path',
            str(f_path_to_dataset_csv),
        ]
        test_args += data_args + bin_cls_agata_args

        monkeypatch.setattr(sys, 'argv', test_args)
        main()
        captured = capsys.readouterr()
        assert captured.err == ''

        test_args = [
            '/home/aicompute/development/ai-core/lib/ml-sdk/src/paige/ml_sdk/__main__.py',
            'test',
            '--data.test_dataset_path',
            str(f_path_to_dataset_csv),
            '--ckpt_path',
            'best',
        ]
        test_args += data_args

    def test_cli_should_raise_RuntimeError_if_dataset_doesnt_match_command(
        self,
        monkeypatch: MonkeyPatch,
        f_path_to_dataset_csv: Path,
        data_args: List[str],
        bin_cls_agata_args: List[str],
    ) -> None:
        test_args = [
            '/home/aicompute/development/ai-core/lib/ml-sdk/src/paige/ml_sdk/__main__.py',
            'validate',
            '--data.train_dataset_path',
            str(f_path_to_dataset_csv),
        ]
        test_args += data_args + bin_cls_agata_args
        monkeypatch.setattr(sys, 'argv', test_args)
        with pytest.raises(RuntimeError):
            main()

    def test_cli_should_raise_RuntimeError_if_embeddings_dir_mismatch(
        self,
        monkeypatch: MonkeyPatch,
        f_path_to_dataset_csv: Path,
        f_path_to_embeddings_dir: Path,
        data_args: List[str],
        bin_cls_agata_args: List[str],
    ) -> None:
        test_args = [
            '/home/aicompute/development/ai-core/lib/ml-sdk/src/paige/ml_sdk/__main__.py',
            'validate',
            '--data.tune_dataset_path',
            str(f_path_to_dataset_csv),
            '--data.tune_embeddings_dir',
            str(f_path_to_embeddings_dir),
        ]
        test_args += data_args + bin_cls_agata_args
        monkeypatch.setattr(sys, 'argv', test_args)
        with pytest.raises(RuntimeError):
            main()

    def test_cli_should_raise_RuntimeError_if_no_embeddings_dir_provided(
        self,
        monkeypatch: MonkeyPatch,
        f_path_to_dataset_csv: Path,
        bin_cls_agata_args: List[str],
    ) -> None:
        test_args = [
            '/home/aicompute/development/ai-core/lib/ml-sdk/src/paige/ml_sdk/__main__.py',
            'validate',
            '--data.tune_dataset_path',
            str(f_path_to_dataset_csv),
            '--data.embeddings_filename_column',
            'image_uri',
            '--data.label_columns',
            '[cancer,precursor]',
        ]
        test_args += bin_cls_agata_args
        monkeypatch.setattr(sys, 'argv', test_args)
        with pytest.raises(RuntimeError):
            main()

    def test_cli_should_raise_RuntimeError_if_no_dataset_provided(
        self,
        monkeypatch: MonkeyPatch,
        f_path_to_dataset_csv: Path,
        data_args: List[str],
        bin_cls_agata_args: List[str],
    ) -> None:
        test_args = [
            '/home/aicompute/development/ai-core/lib/ml-sdk/src/paige/ml_sdk/__main__.py',
            'validate',
        ]
        test_args += data_args + bin_cls_agata_args
        monkeypatch.setattr(sys, 'argv', test_args)
        with pytest.raises(RuntimeError):
            main()

    def test_cli_should_run_fit_with_MultiClsAgata(
        self,
        capsys: CaptureFixture,
        monkeypatch: MonkeyPatch,
        f_path_to_dataset_csv: Path,
        f_path_to_embeddings_dir: Path,
    ) -> None:
        test_args = [
            '/home/aicompute/development/ai-core/lib/ml-sdk/src/paige/ml_sdk/__main__.py',
            'fit',
            '--model',
            'MultiClsAgata',
            '--data.tune_dataset_path',
            str(f_path_to_dataset_csv),
            '--data.embeddings_dir',
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
