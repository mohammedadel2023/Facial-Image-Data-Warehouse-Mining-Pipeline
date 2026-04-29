"""
etl/ingest.py
─────────────────────────────────────────────────────────────────────────────
Phase 2, Step 3 – Data Ingestion

Downloads FER2013 and CelebA datasets from Kaggle using `kagglehub`.
kagglehub handles authentication automatically via the KAGGLE_USERNAME and
KAGGLE_KEY environment variables (or ~/.kaggle/kaggle.json).

Returns the local paths so downstream preprocessing modules know where
to find the raw images.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import kagglehub  # pip install kagglehub

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset identifiers on Kaggle
# ---------------------------------------------------------------------------
_FER2013_HANDLE = "msambare/fer2013"          # kaggle dataset slug
_CELEBA_HANDLE  = "jessicali9530/celeba-dataset"  # kaggle dataset slug


def download_fer2013(target_dir: Path | str | None = None) -> Path:
    """
    Download the FER2013 dataset via kagglehub.

    Parameters
    ----------
    target_dir : optional path
        If provided, the dataset files are moved/symlinked here.
        If None, kagglehub uses its own default cache directory
        (~/.cache/kagglehub on Linux/macOS, %USERPROFILE%/.cache/kagglehub on Windows).

    Returns
    -------
    Path
        Local root directory of the downloaded FER2013 dataset.
    """
    logger.info("Downloading FER2013 dataset …")
    try:
        local_path = kagglehub.dataset_download(_FER2013_HANDLE)
    except Exception as exc:
        logger.error("Failed to download FER2013: %s", exc)
        raise

    result = Path(local_path)
    logger.info("FER2013 downloaded to: %s", result)
    return result


def download_celeba(target_dir: Path | str | None = None) -> Path:
    """
    Download the CelebA dataset via kagglehub.

    Parameters
    ----------
    target_dir : optional path
        See `download_fer2013` docstring.

    Returns
    -------
    Path
        Local root directory of the downloaded CelebA dataset.
    """
    logger.info("Downloading CelebA dataset …")
    try:
        local_path = kagglehub.dataset_download(_CELEBA_HANDLE)
    except Exception as exc:
        logger.error("Failed to download CelebA: %s", exc)
        raise

    result = Path(local_path)
    logger.info("CelebA downloaded to: %s", result)
    return result


def ingest_all(data_root: Path | str | None = None) -> dict[str, Path]:
    """
    Download both datasets and return a mapping of dataset name → local path.

    Parameters
    ----------
    data_root : optional root directory
        Sub-folders 'fer2013' and 'celeba' will be created inside this root.
        Defaults to the DATA_ROOT environment variable, or './data'.

    Returns
    -------
    dict
        {
            "fer2013": Path("/path/to/fer2013/"),
            "celeba":  Path("/path/to/celeba/"),
        }
    """
    if data_root is None:
        data_root = os.environ.get("DATA_ROOT", "./data")

    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    return {
        "fer2013": download_fer2013(target_dir=data_root / "fer2013"),
        "celeba":  download_celeba(target_dir=data_root / "celeba"),
    }


# ---------------------------------------------------------------------------
# Quick sanity-check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    paths = ingest_all()
    for name, p in paths.items():
        print(f"{name:10s} → {p}")
