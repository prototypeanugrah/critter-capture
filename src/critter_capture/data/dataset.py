"""Dataset utilities for the observations CSV dataset."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from critter_capture.config import DataConfig
from critter_capture.data.splits import stratified_train_val_test_split
from critter_capture.data.transforms import build_transforms
from critter_capture.utils import ensure_dir

LOGGER = logging.getLogger(__name__)


@dataclass
class ObservationRecord:
    uuid: str
    image_url: str
    label_index: int


class ObservationsDataset(TorchDataset):
    """PyTorch dataset for observations backed by cached image files."""

    def __init__(
        self,
        records: List[ObservationRecord],
        label_names: Sequence[str],
        transform,
        cache_dir: Path,
    ) -> None:
        self._records = records
        self._label_names = list(label_names)
        self._transform = transform
        self._cache_dir = ensure_dir(cache_dir)
        self._num_classes = len(label_names)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self._records[index]
        image = self._load_image(record)
        tensor = self._transform(image)
        target = torch.tensor(record.label_index, dtype=torch.long)

        return {
            "image": tensor,
            "target": target,
            "uuid": record.uuid,
        }

    @property
    def label_names(self) -> Sequence[str]:  # pragma: no cover - simple
        return self._label_names

    def _load_image(self, record: ObservationRecord) -> Image.Image:
        cache_path = self._cache_dir / f"{record.uuid}.jpg"
        if not cache_path.exists():
            self._download_image(record.image_url, cache_path)

        try:
            # Open the image file directly instead of using a file handle
            image = Image.open(cache_path).convert("RGB")
            return image
        except Exception as e:
            LOGGER.warning(
                "Failed to load image %s (UUID: %s): %s. Attempting to re-download...",
                cache_path,
                record.uuid,
                str(e),
            )
            # Try to re-download the image in case it was corrupted
            try:
                self._download_image(record.image_url, cache_path)
                image = Image.open(cache_path).convert("RGB")
                return image
            except Exception as e2:
                LOGGER.error(
                    "Failed to load image %s (UUID: %s) even after re-download: %s",
                    cache_path,
                    record.uuid,
                    str(e2),
                )
                # Return a placeholder image to prevent the training from crashing
                # This is a 1x1 white pixel image
                return Image.new("RGB", (1, 1), color="white")

    @staticmethod
    def _download_image(url: str, destination: Path) -> None:
        ensure_dir(destination.parent)
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            LOGGER.debug("Downloading image from %s", url)
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            destination.write_bytes(response.content)
        else:
            source = Path(url)
            if not source.exists():  # pragma: no cover - defensive
                raise FileNotFoundError(f"Image source not found for {url}")
            shutil.copy(source, destination)


@dataclass
class DatasetBundle:
    train: ObservationsDataset
    validation: ObservationsDataset
    test: ObservationsDataset
    label_names: Sequence[str]
    label_ids: Sequence[int]


def _load_records(
    cfg: DataConfig,
) -> tuple[List[ObservationRecord], List[str], List[int]]:
    df = pd.read_csv(cfg.csv_path)
    required_columns = {
        cfg.uuid_column,
        cfg.image_url_column,
        cfg.label_column,
        cfg.label_names_column,
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df = df.dropna(subset=list(required_columns))

    if cfg.sample_limit:
        df = df.head(cfg.sample_limit)

    df[cfg.label_column] = df[cfg.label_column].astype(int)

    label_id_to_name = (
        df[[cfg.label_column, cfg.label_names_column]]
        .drop_duplicates(subset=[cfg.label_column])
        .set_index(cfg.label_column)[cfg.label_names_column]
        .to_dict()
    )
    label_ids_sorted = sorted(label_id_to_name.keys())
    label_names = [label_id_to_name[label_id] for label_id in label_ids_sorted]
    label_to_index = {label_id: idx for idx, label_id in enumerate(label_ids_sorted)}

    grouped = df.groupby(cfg.uuid_column, sort=False)
    records: List[ObservationRecord] = []
    for uuid, group in grouped:
        image_url = str(group.iloc[0][cfg.image_url_column])
        label_ids_unique = sorted(
            {int(label) for label in group[cfg.label_column].tolist()}
        )
        if len(label_ids_unique) != 1:
            LOGGER.warning(
                "Observation %s has %d labels; selecting the first for multi-class setup.",
                uuid,
                len(label_ids_unique),
            )
        label_id = label_ids_unique[0]
        label_index = label_to_index[label_id]
        records.append(
            ObservationRecord(
                uuid=uuid,
                image_url=image_url,
                label_index=label_index,
            )
        )

    LOGGER.info(
        "Loaded %d unique observations with %d labels", len(records), len(label_names)
    )
    return records, label_names, label_ids_sorted


def prepare_datasets(cfg: DataConfig, seed: int) -> DatasetBundle:
    records, label_names, label_ids = _load_records(cfg)
    num_classes = len(label_names)
    if num_classes == 0:
        raise ValueError("No labels discovered in dataset.")

    label_array = np.array(
        [record.label_index for record in records],
        dtype=np.int32,
    )

    train_idx, val_idx, test_idx = stratified_train_val_test_split(
        label_array,
        validation_size=cfg.validation_size,
        test_size=cfg.test_size,
        seed=seed,
    )

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]

    train_transform, eval_transform = build_transforms(
        cfg.image_size, cfg.augmentations
    )

    torch_train = ObservationsDataset(
        train_records,
        label_names,
        train_transform,
        cfg.image_cache_dir,
    )
    torch_val = ObservationsDataset(
        val_records,
        label_names,
        eval_transform,
        cfg.image_cache_dir,
    )
    torch_test = ObservationsDataset(
        test_records,
        label_names,
        eval_transform,
        cfg.image_cache_dir,
    )

    return DatasetBundle(
        train=torch_train,
        validation=torch_val,
        test=torch_test,
        label_names=label_names,
        label_ids=label_ids,
    )


def build_dataloaders(
    bundle: DatasetBundle,
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    return {
        "train": DataLoader(
            bundle.train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "validation": DataLoader(
            bundle.validation,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            bundle.test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }


__all__ = [
    "ObservationRecord",
    "ObservationsDataset",
    "DatasetBundle",
    "prepare_datasets",
    "build_dataloaders",
]
