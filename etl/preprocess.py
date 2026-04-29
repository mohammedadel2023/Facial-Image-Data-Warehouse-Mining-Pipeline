"""
etl/preprocess.py
─────────────────────────────────────────────────────────────────────────────
Phase 2, Step 4 – Image Preprocessing

Provides OpenCV-based preprocessing utilities that normalise both datasets
into an identical tensor shape so the CNN feature extractor receives a
consistent input format.

Pipeline per image
──────────────────
1. Load image from disk (OpenCV BGR by default).
2. Resize to TARGET_SIZE × TARGET_SIZE (224×224).
3. FER2013 images are already grayscale (48×48).  After resizing, the single
   channel is duplicated into 3 identical channels to simulate RGB — this
   prevents the ResNet18 from receiving colour information that FER lacks,
   eliminating colour bias between datasets.
4. CelebA images are colour.  They are converted to grayscale FIRST, then the
   single channel is duplicated into 3 channels by the same method, so both
   datasets reach the CNN as "pseudo-RGB" grayscale images.
5. The result is converted to a float32 NumPy array normalised to [0, 1].

Why duplicate instead of using cv2.COLORMAP?
    Duplication (np.stack or cv2.merge) ensures the three channels are
    numerically identical, meaning the CNN sees no hue information — only
    luminance.  This is intentional to remove colour as a confounding
    variable when comparing FER and CelebA feature spaces.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_SIZE: int = 224          # Both spatial dimensions (pixels)
SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ── Core preprocessing function ──────────────────────────────────────────────

def preprocess_image(image_path: str | Path) -> np.ndarray:
    """
    Load and preprocess a single image into a normalised 3-channel tensor.

    Steps
    -----
    1. Read with OpenCV (handles colour or grayscale automatically).
    2. Resize to (TARGET_SIZE, TARGET_SIZE).
    3. Convert to grayscale (eliminating colour bias).
    4. Duplicate the single grayscale channel to produce a (H, W, 3) array.
    5. Normalise pixel values to float32 in [0.0, 1.0].

    Parameters
    ----------
    image_path : path to any JPEG, PNG, BMP, or TIFF image.

    Returns
    -------
    np.ndarray
        Shape: (TARGET_SIZE, TARGET_SIZE, 3), dtype: float32, range: [0, 1].

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If OpenCV cannot decode the file.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Step 1 – Load (OpenCV reads BGR; for grayscale images it returns (H, W))
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")

    # Step 2 – Resize to 224×224 using bilinear interpolation
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)

    # Step 3 – Convert to grayscale (works for both colour and already-grey images)
    if img.ndim == 3 and img.shape[2] == 3:
        # Colour BGR → grayscale  (standard luminosity formula)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        # RGBA → grayscale (drop alpha)
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        # Already a single-channel array (shape: H×W)
        gray = img

    # Ensure gray is 2-D at this point
    if gray.ndim != 2:
        raise ValueError(f"Unexpected array shape after grayscale conversion: {gray.shape}")

    # Step 4 – Mathematically duplicate the single channel into 3 channels.
    #   np.stack creates shape (3, H, W); cv2.merge expects a list of 2-D arrays
    #   and returns (H, W, 3) – the format PyTorch transforms expect before
    #   PIL conversion, and also what cv2.imshow expects.
    pseudo_rgb = cv2.merge([gray, gray, gray])     # shape: (H, W, 3)

    # Step 5 – Normalise to float32 [0, 1]
    normalised = pseudo_rgb.astype(np.float32) / 255.0

    return normalised   # (224, 224, 3) float32


# ── Batch helpers ─────────────────────────────────────────────────────────────

def iter_image_paths(directory: str | Path) -> Iterator[Path]:
    """
    Recursively yield all image file paths inside *directory*.

    Parameters
    ----------
    directory : root folder to walk.

    Yields
    ------
    Path
        Absolute path to each image file found.
    """
    directory = Path(directory)
    for path in directory.rglob("*"):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def preprocess_batch(
    directory: str | Path,
    *,
    max_images: int | None = None,
) -> Iterator[tuple[Path, np.ndarray]]:
    """
    Lazily preprocess all images in a directory tree.

    Memory-efficient: images are processed one at a time and yielded
    as (path, array) tuples, so the caller can process then discard each
    before loading the next.

    Parameters
    ----------
    directory    : root folder containing raw images.
    max_images   : optional hard cap for debugging / dry-runs.

    Yields
    ------
    (Path, np.ndarray)
        The file path and the preprocessed (224, 224, 3) float32 array.
    """
    count = 0
    for img_path in iter_image_paths(directory):
        if max_images is not None and count >= max_images:
            break
        try:
            tensor = preprocess_image(img_path)
            yield img_path, tensor
            count += 1
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping %s — %s", img_path.name, exc)


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m etl.preprocess <image_path_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_file():
        arr = preprocess_image(target)
        print(f"Shape: {arr.shape}  dtype: {arr.dtype}  min: {arr.min():.4f}  max: {arr.max():.4f}")
    else:
        for p, arr in preprocess_batch(target, max_images=5):
            print(f"{p.name:40s}  shape={arr.shape}  dtype={arr.dtype}")
