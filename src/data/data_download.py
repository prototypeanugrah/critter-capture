import argparse
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download images from URLs.")
    parser.add_argument("--data_path", type=Path, default=Path("data/data.csv"))
    parser.add_argument("--save_dir", type=Path, default=Path("data/raw/images"))
    parser.add_argument("--uuid_column", type=str, default="uuid")
    parser.add_argument("--image_url_column", type=str, default="image_url")
    parser.add_argument("--plot_save_dir", type=Path, default=Path("data/raw/plots"))
    parser.add_argument("--max_workers", type=int, default=10)
    return parser


def download_image(
    uuid,
    image_url,
    save_dir,
    session=None,
):
    """
    Download a single image from URL and save it.

    Args:
        uuid: Unique identifier for the image
        image_url: URL of the image to download
        save_dir: Directory to save the image
        session: requests.Session object for connection pooling (optional)

    Returns:
        Tuple of (uuid, save_path) on success, (uuid, None) on failure
    """
    save_path = os.path.join(save_dir, f"{uuid}.jpg")

    # Skip if already downloaded
    if os.path.exists(save_path):
        return (uuid, save_path)

    try:
        # Use session if provided for connection pooling, otherwise create new request
        if session:
            response = session.get(
                image_url,
                stream=True,
                timeout=30,
            )
        else:
            response = requests.get(
                image_url,
                stream=True,
                timeout=30,
            )

        if response.status_code == 200:
            # Read image content
            img = Image.open(io.BytesIO(response.content))
            # Save image
            img.save(save_path)
            return (uuid, save_path)
        else:
            return (uuid, None)
    except Exception:
        # Silently fail - we'll track errors separately
        return (uuid, None)


def download_images_parallel(
    data,
    uuid_column,
    image_url_column,
    save_dir,
    max_workers=20,
):
    """
    Download images in parallel using ThreadPoolExecutor.

    Args:
        data: DataFrame with image URLs
        uuid_column: Column name containing UUIDs
        image_url_column: Column name containing image URLs
        save_dir: Directory to save images
        max_workers: Number of concurrent downloads (adjust based on your connection)

    Returns:
        Dictionary mapping UUID to image path
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create a session for connection pooling
    session = requests.Session()
    # Use adapter with connection pooling
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers,
        max_retries=3,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Filter out rows with no image URL
    download_rows = data[data[image_url_column].notna()].copy()

    # Create partial function with session and save_dir
    download_func = partial(
        download_image,
        save_dir=save_dir,
        session=session,
    )

    # Prepare download tasks
    tasks = [
        (row[uuid_column], row[image_url_column]) for _, row in download_rows.iterrows()
    ]

    # Track results
    results = {}
    failed = []

    # Use ThreadPoolExecutor for concurrent downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_uuid = {
            executor.submit(download_func, uuid, url): uuid for uuid, url in tasks
        }

        # Process completed downloads with progress bar
        with tqdm(
            total=len(tasks),
            desc="Downloading images",
        ) as pbar:
            for future in as_completed(future_to_uuid):
                uuid = future_to_uuid[future]
                try:
                    result_uuid, save_path = future.result()
                    results[result_uuid] = save_path
                    if save_path is None:
                        failed.append(result_uuid)
                except Exception:
                    failed.append(uuid)
                    results[uuid] = None
                finally:
                    pbar.update(1)

    session.close()

    LOGGER.info("\nDownload completed:")
    LOGGER.info(
        "  Success: %s",
        len([v for v in results.values() if v is not None]),
    )
    LOGGER.info("  Failed: %s", len(failed))

    return results, failed


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    data = pd.read_csv(args.data_path)
    # Download images in parallel
    save_dir = args.save_dir

    max_workers = args.max_workers

    LOGGER.info("Starting parallel download with %s workers...", max_workers)
    image_paths, failed_uuids = download_images_parallel(
        data,
        args.uuid_column,
        args.image_url_column,
        save_dir,
        max_workers=max_workers,
    )

    # Add image paths to dataframe
    data["image_path"] = data[args.uuid_column].map(image_paths)
    LOGGER.info(
        "Shape of data before removing rows with no image paths: %s",
        data.shape,
    )
    data = data[data["image_path"].notna()]
    LOGGER.info(
        "Shape of data after removing rows with no image paths: %s",
        data.shape,
    )

    data["image_path"] = data["image_path"].str.replace(
        "^../",
        "",
        regex=True,
    )

    data.to_csv(
        args.data_path.parent / "data_existing_image_paths.csv",
        index=False,
    )
