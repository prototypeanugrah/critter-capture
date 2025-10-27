"""Dataset utilities for the observations CSV dataset."""

from __future__ import annotations

import logging
import re
import shutil
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from requests import RequestException
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from critter_capture.config import DataConfig
from critter_capture.data.splits import stratified_train_val_test_split
from critter_capture.data.transforms import build_transforms
from critter_capture.utils import ensure_dir

LOGGER = logging.getLogger(__name__)
_INAT_OBSERVATION_PATTERN = re.compile(r"/observations/(\\d+)")
_INAT_API_TEMPLATE = "https://api.inaturalist.org/v1/observations/{obs_id}"


@dataclass
class ObservationRecord:
    uuid: str
    image_url: str
    label_index: int


class ObservationsDataset(TorchDataset):
    """PyTorch dataset for observations backed by cached image files."""

    _MAX_DOWNLOAD_RETRIES = 3
    _RETRY_BACKOFF_SECONDS = 2.0

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
            if not self._download_with_retries(
                record.image_url, cache_path, record.uuid
            ):
                return self._placeholder_image(cache_path)

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
            cache_path.unlink(missing_ok=True)
            if self._download_with_retries(record.image_url, cache_path, record.uuid):
                try:
                    image = Image.open(cache_path).convert("RGB")
                    return image
                except Exception as e2:
                    LOGGER.error(
                        "Failed to load image %s (UUID: %s) even after re-download: %s",
                        cache_path,
                        record.uuid,
                        str(e2),
                    )
            return self._placeholder_image(cache_path)

    def _download_with_retries(self, url: str, destination: Path, uuid: str) -> bool:
        for attempt in range(1, self._MAX_DOWNLOAD_RETRIES + 1):
            try:
                self._download_image(url, destination)
                return True
            except RequestException as exc:
                LOGGER.warning(
                    "HTTP error downloading image for UUID %s (attempt %d/%d): %s",
                    uuid,
                    attempt,
                    self._MAX_DOWNLOAD_RETRIES,
                    exc,
                )
                destination.unlink(missing_ok=True)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning(
                    "Unexpected error downloading image for UUID %s (attempt %d/%d): %s",
                    uuid,
                    attempt,
                    self._MAX_DOWNLOAD_RETRIES,
                    exc,
                )
                destination.unlink(missing_ok=True)

            if attempt < self._MAX_DOWNLOAD_RETRIES:
                time.sleep(self._RETRY_BACKOFF_SECONDS * attempt)

        LOGGER.error(
            "Giving up on downloading image for UUID %s from %s after %d attempts.",
            uuid,
            url,
            self._MAX_DOWNLOAD_RETRIES,
        )
        return False

    @staticmethod
    def _download_image(url: str, destination: Path) -> None:
        ensure_dir(destination.parent)
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            resolved_url = _maybe_resolve_inat_photo_url(parsed)
            if resolved_url:
                url = resolved_url
                parsed = urlparse(url)
            LOGGER.debug("Downloading image from %s", url)
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type.lower():
                raise RequestException(
                    f"Unexpected content type '{content_type}' while downloading {url}"
                )
            destination.write_bytes(response.content)
        else:
            source = Path(url)
            if not source.exists():  # pragma: no cover - defensive
                raise FileNotFoundError(f"Image source not found for {url}")
            shutil.copy(source, destination)

    @staticmethod
    def _placeholder_image(destination: Path) -> Image.Image:
        placeholder = Image.new("RGB", (1, 1), color="white")
        try:
            placeholder.save(destination, format="JPEG")
        except Exception:  # pragma: no cover - best effort
            pass
        return placeholder


def _extract_image_url(row: pd.Series, cfg: DataConfig) -> str | None:
    def _clean(value) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return text

    primary = _clean(row.get(cfg.image_url_column))
    if primary:
        return primary

    fallback_col = getattr(cfg, "image_url_fallback_column", None)
    if fallback_col:
        fallback = _clean(row.get(fallback_col))
        if fallback:
            return fallback

    return None


def _maybe_resolve_inat_photo_url(parsed_url) -> str | None:
    if "inaturalist.org" not in parsed_url.netloc:
        return None

    match = _INAT_OBSERVATION_PATTERN.search(parsed_url.path)
    if not match:
        return None

    obs_id = match.group(1)
    return _resolve_inat_photo_url(obs_id)


@lru_cache(maxsize=8192)
def _resolve_inat_photo_url(obs_id: str) -> str | None:
    api_url = _INAT_API_TEMPLATE.format(obs_id=obs_id)
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results") or []
    if not results:
        return None

    photos = results[0].get("photos") or []
    if not photos:
        return None

    preferred_keys = (
        "original_url",
        "large_url",
        "medium_url",
        "url",
        "small_url",
    )
    for key in preferred_keys:
        candidate = photos[0].get(key)
        if candidate:
            return candidate.replace("square", "large")
    return None


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
    if cfg.image_url_fallback_column:
        required_columns.add(cfg.image_url_fallback_column)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    essential_columns = {
        cfg.uuid_column,
        cfg.label_column,
        cfg.label_names_column,
    }
    df = df.dropna(subset=list(essential_columns))

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
        image_url = _extract_image_url(group.iloc[0], cfg)
        if not image_url:
            LOGGER.warning("Skipping observation %s due to missing image URL.", uuid)
            continue
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


def compute_class_weights(dataset: ObservationsDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced classification."""

    counts = torch.zeros(dataset._num_classes, dtype=torch.float32)
    for record in dataset._records:
        counts[record.label_index] += 1.0

    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (counts * counts.numel())
    weights = weights / weights.mean()
    return weights


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
    "compute_class_weights",
    "build_dataloaders",
]
