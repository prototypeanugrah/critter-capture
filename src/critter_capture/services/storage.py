"""Storage helper utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import boto3

from critter_capture.utils.env import require_env

LOGGER = logging.getLogger(__name__)


def upload_feedback(
    bucket: str,
    region: Optional[str],
    key_prefix: str,
    image_path: Path,
    metadata: Dict[str, str],
) -> None:
    """Upload feedback image and metadata to S3."""

    require_env(["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])
    session = boto3.session.Session(region_name=region)
    s3 = session.client("s3")

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    key_base = f"{key_prefix}/{timestamp}_{image_path.name}"
    LOGGER.info("Uploading feedback to s3://%s/%s", bucket, key_base)

    with image_path.open("rb") as handle:
        s3.upload_fileobj(handle, bucket, key_base)

    meta_key = key_base + ".json"
    s3.put_object(
        Bucket=bucket,
        Key=meta_key,
        Body=json.dumps(metadata, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


__all__ = ["upload_feedback"]
