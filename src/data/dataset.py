"""Dataset utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.config import DataConfig
from src.data.dataloader import build_transforms

LOGGER = logging.getLogger(__name__)


@dataclass
class ImageRecord:
    """Image record.

    Args:
        uuid (str): The UUID of the image.
        image_url (str): The URL of the image.
        label_index (int): The index of the label.
    """

    uuid: str
    image_url: str
    label_index: int
    image_path: str


@dataclass
class AnimalDataset(Dataset):
    """Animal dataset.

    Args:
        records (List[ImageRecord]): A list of image records.
        label_names (List[str]): A list of label names.
        transform (Optional[transforms.Compose]): A transform to apply to the image.
        cache_dir (Optional[Path]): A directory to cache the images.
    """

    _MAX_DOWNLOAD_RETRIES = 1
    _RETRY_BACKOFF_SECONDS = 2.0

    def __init__(
        self,
        records: List[ImageRecord],
        label_names: List[str],
        transform: Optional[transforms.Compose] = None,
        cache_dir: Optional[Path] = None,
    ):
        self._transform = transform
        self._records = records
        self._label_names = label_names
        self._cache_dir = cache_dir
        self._num_classes = len(label_names)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        record = self._records[idx]
        image = self._load_image(record)
        image = self._transform(image)
        target = torch.tensor(record.label_index, dtype=torch.long)
        return image, target

    def _load_image(self, record: ImageRecord) -> Image.Image:
        image_path = Path(record.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image path {image_path} does not exist.")
        return Image.open(image_path).convert("RGB")


@dataclass
class DatasetBundle:
    train: AnimalDataset
    validation: AnimalDataset
    test: AnimalDataset
    label_names: List[str]
    label_ids: List[int]


def clean_and_prepare_data(
    data_path: Path, config: DataConfig
) -> Tuple[List[ImageRecord], List[str], List[int]]:
    """
    Clean and prepare the data for the model. This function will:
    - Remove duplicate image URLs.
    - Remove entries without image URLs.
    - Remove labels with less than config.keep_min_samples_per_label counts.
    - Keep labels with at least config.keep_min_samples_per_label counts.
    - Sort the labels by label IDs.

    Args:
        data (pd.DataFrame): The data to clean and prepare.
        config (DataConfig): The configuration.

    Returns:
        Tuple[List[ImageRecord], List[str], List[int]]: A tuple containing the records, label names, and label IDs sorted.
            - records (List[ImageRecord]): A list of image records.
            - label_names (List[str]): A list of label names sorted by label IDs.
            - label_ids_sorted (List[int]): A list of label IDs sorted by label names.
    """
    data = pd.read_csv(data_path)
    LOGGER.info(
        "Shape of the dataset: %s",
        data.shape,
    )
    LOGGER.info(
        "Number of unique %s: %s",
        config.label_names_column,
        data[config.label_names_column].nunique(),
    )
    LOGGER.info(
        "Number of missing values in %s: %s",
        config.image_url_column,
        data[config.image_url_column].isna().sum(),
    )
    data = data[data[config.image_url_column].notna()]
    LOGGER.info(
        "Shape of the dataset after removing entries without image URLs: %s",
        data.shape,
    )

    LOGGER.info(
        "Shape of the dataset before removing duplicate image URLs: %s",
        data.shape,
    )
    data = data.drop_duplicates(subset=[config.image_url_column], keep="first")
    LOGGER.info(
        "Shape of the dataset after removing duplicate image URLs: %s",
        data.shape,
    )

    # get the common_name which have only 1 count
    vc = data[config.label_names_column].value_counts()

    less_count_df = data[
        data[config.label_names_column].map(vc) < config.keep_min_samples_per_label
    ]
    LOGGER.info(
        "Shape of the dataset containing %s with less than %s counts: %s",
        config.label_names_column,
        config.keep_min_samples_per_label,
        less_count_df.shape,
    )
    LOGGER.info(
        "Number of unique %s with less than %s counts: %s",
        config.label_names_column,
        config.keep_min_samples_per_label,
        less_count_df[config.label_names_column].nunique(),
    )

    data = data[
        data[config.label_names_column].map(vc) >= config.keep_min_samples_per_label
    ]
    LOGGER.info(
        "Shape of the dataset containing %s with at least %s counts: %s",
        config.label_names_column,
        config.keep_min_samples_per_label,
        data.shape,
    )
    LOGGER.info(
        "Number of unique %s with at least %s counts: %s",
        config.label_names_column,
        config.keep_min_samples_per_label,
        data[config.label_names_column].nunique(),
    )

    data[config.label_column] = data[config.label_column].astype(int)

    label_id_to_name = (
        data[[config.label_column, config.label_names_column]]
        .drop_duplicates(subset=[config.label_column])
        .set_index(config.label_column)[config.label_names_column]
        .to_dict()
    )
    label_ids_sorted = sorted(label_id_to_name.keys())
    label_names = [label_id_to_name[label_id] for label_id in label_ids_sorted]
    label_to_index = {label_id: idx for idx, label_id in enumerate(label_ids_sorted)}

    grouped = data.groupby(config.uuid_column, sort=False)
    records: List[ImageRecord] = []
    missing_image_urls = []

    for uuid, group in grouped:
        image_url = group[config.image_url_column].iloc[0]
        image_path = group[config.image_path_column].iloc[0]

        if not image_url or not image_path:
            LOGGER.warning("Image URL is missing for UUID: %s", uuid)
            continue

        label_ids_unique = sorted(
            {int(label_id) for label_id in group[config.label_column].unique()}
        )

        label_id = label_ids_unique[0]
        label_index = label_to_index[label_id]
        records.append(
            ImageRecord(
                uuid=uuid,
                image_url=image_url,
                label_index=label_index,
                image_path=image_path,
            )
        )
    LOGGER.info(
        "Number of records with missing image URLs: %s",
        len(missing_image_urls),
    )
    return records, label_names, label_ids_sorted


def create_stratified_splits(
    labels: np.ndarray,
    validation_size: float,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create stratified train/validation/test splits for multi-class data."""

    if validation_size + test_size >= 0.9:
        raise ValueError(
            "Combined validation and test size should leave reasonable train data."
        )

    indices = np.arange(len(labels))
    stratify_labels = labels if np.min(np.bincount(labels)) >= 2 else None
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=validation_size + test_size,
        random_state=seed,
        stratify=stratify_labels,
    )

    holdout_labels = labels[holdout_idx]
    stratify_holdout = (
        holdout_labels if np.min(np.bincount(holdout_labels)) >= 2 else None
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=test_size / (validation_size + test_size),
        random_state=seed,
        stratify=stratify_holdout,
    )

    return train_idx, val_idx, test_idx


def prepare_data_for_training(
    config: DataConfig,
) -> DatasetBundle:
    """
    Prepare the data for training. This function will:
    - Convert the records to a PyTorch dataset.
    - Return the dataset, label names, and label IDs sorted.
    """
    data_path = Path(config.csv_path)
    records, label_names, label_ids = clean_and_prepare_data(
        data_path,
        config,
    )

    num_classes = len(label_names)
    if num_classes == 0:
        raise ValueError("No classes found in the dataset")

    label_array = np.array([record.label_index for record in records])
    train_idx, val_idx, test_idx = create_stratified_splits(
        label_array,
        config.validation_size,
        config.test_size,
        config.seed,
    )

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]

    train_transform, eval_transform = build_transforms(config.image_size)

    train_dataset = AnimalDataset(
        train_records,
        label_names,
        train_transform,
        config.image_cache_dir,
    )
    val_dataset = AnimalDataset(
        val_records,
        label_names,
        eval_transform,
        config.image_cache_dir,
    )
    test_dataset = AnimalDataset(
        test_records,
        label_names,
        eval_transform,
        config.image_cache_dir,
    )

    return DatasetBundle(
        train=train_dataset,
        validation=val_dataset,
        test=test_dataset,
        label_names=label_names,
        label_ids=label_ids,
    )


def build_data_loaders(
    bundle: DatasetBundle,
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    """Build data loaders for the training, validation, and test datasets."""
    train_loader = DataLoader(
        bundle.train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        bundle.validation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        bundle.test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader,
    }


__all__ = [
    "ImageRecord",
    "AnimalDataset",
    "DatasetBundle",
    "clean_and_prepare_data",
    "prepare_data_for_training",
    "build_data_loaders",
]
