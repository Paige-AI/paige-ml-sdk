from pathlib import Path

import pandas as pd
import pytest
import torch

SLIDENAMES = ['slide_0.svs', 'slide_1.svs', 'slide_2.svs']


@pytest.fixture
def f_path_to_dataset_csv(tmpdir: Path) -> Path:
    df = pd.DataFrame(
        {
            'group': ['group_1', 'group_2', 'group_1'],
            'image_uri': SLIDENAMES,
            'cancer': [0, 1, 0],
            'precursor': [0, -999, 0],
            'grade': [2, 0, 2],
        }
    )
    out = Path(tmpdir / 'dataset.csv')
    df.to_csv(out)

    return out


@pytest.fixture
def f_path_to_embeddings_dir(tmpdir: Path) -> Path:
    for s in SLIDENAMES:
        embedding = torch.randn(size=(1, 5))
        out = Path(tmpdir / f'{s}.pt')
        torch.save({'embeddings': embedding}, out)
    return tmpdir
