from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image

from critter_capture.config import DataConfig
from critter_capture.data import build_dataloaders, prepare_datasets


def _make_image(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    return Image.fromarray(array)


def test_prepare_datasets_and_dataloaders(tmp_path) -> None:
    image_paths = []
    for i in range(6):
        img = _make_image(i)
        path = tmp_path / f"img_{i}.jpg"
        img.save(path)
        image_paths.append(path)

    rows = [
        {"uuid": "u0", "image_url": str(image_paths[0]), "taxon_id": 101, "common_name": "Label A"},
        {"uuid": "u0", "image_url": str(image_paths[0]), "taxon_id": 102, "common_name": "Label B"},
        {"uuid": "u1", "image_url": str(image_paths[1]), "taxon_id": 101, "common_name": "Label A"},
        {"uuid": "u2", "image_url": str(image_paths[2]), "taxon_id": 103, "common_name": "Label C"},
        {"uuid": "u3", "image_url": str(image_paths[3]), "taxon_id": 102, "common_name": "Label B"},
        {"uuid": "u4", "image_url": str(image_paths[4]), "taxon_id": 104, "common_name": "Label D"},
        {"uuid": "u5", "image_url": str(image_paths[5]), "taxon_id": 101, "common_name": "Label A"},
    ]

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "observations.csv"
    df.to_csv(csv_path, index=False)

    cfg = DataConfig(
        csv_path=csv_path,
        uuid_column="uuid",
        image_url_column="image_url",
        label_column="taxon_id",
        label_names_column="common_name",
        validation_size=0.2,
        test_size=0.2,
        num_workers=0,
        image_size=64,
        normalize_mean=[0.5, 0.5, 0.5],
        normalize_std=[0.2, 0.2, 0.2],
        augmentations=False,
        image_cache_dir=tmp_path / "cache",
    )

    bundle = prepare_datasets(cfg, seed=123)

    assert len(bundle.train) > 0
    assert len(bundle.validation) > 0
    assert len(bundle.test) > 0
    assert len(bundle.label_names) == 4
    assert list(bundle.label_ids) == [101, 102, 103, 104]

    loaders = build_dataloaders(bundle, batch_size=2, num_workers=0)
    batch = next(iter(loaders["train"]))
    assert batch["image"].shape[0] == 2
    assert batch["image"].shape[1:] == (3, 64, 64)
    assert batch["target"].shape == (2,)
    assert batch["target"].dtype == torch.long


def test_prepare_datasets_uses_fallback_column(tmp_path) -> None:
    img = _make_image(42)
    image_path = tmp_path / "img_fallback.jpg"
    img.save(image_path)

    rows = [
        {
            "uuid": f"fallback_{i}",
            "image_url": np.nan,
            "url": str(image_path),
            "taxon_id": 123,
            "common_name": "Fallback Label",
        }
        for i in range(5)
    ]
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "observations_fallback.csv"
    df.to_csv(csv_path, index=False)

    cfg = DataConfig(
        csv_path=csv_path,
        uuid_column="uuid",
        image_url_column="image_url",
        image_url_fallback_column="url",
        label_column="taxon_id",
        label_names_column="common_name",
        validation_size=0.2,
        test_size=0.2,
        num_workers=0,
        image_size=64,
        augmentations=False,
        image_cache_dir=tmp_path / "cache_fallback",
    )

    bundle = prepare_datasets(cfg, seed=7)
    total = len(bundle.train) + len(bundle.validation) + len(bundle.test)
    assert total == len(rows)
    assert any(dataset._records for dataset in (bundle.train, bundle.validation, bundle.test))
