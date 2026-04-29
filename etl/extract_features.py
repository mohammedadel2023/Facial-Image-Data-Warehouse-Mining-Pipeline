"""
etl/extract_features.py
─────────────────────────────────────────────────────────────────────────────
Phase 2, Step 5 – Feature Extraction

Loads a pre-trained ResNet18 from torchvision and surgically removes its
final fully-connected classification layer.  The resulting "headless" model
outputs a 512-dimensional feature vector (the avgpool layer output) for
every image instead of class probabilities.

Why ResNet18?
    – Lightweight enough to run on CPU during development.
    – The avgpool output is a compact 512-d embedding well-suited for
      downstream clustering (K-Means) and association rule mining (Apriori).
    – Pre-trained on ImageNet, so its convolutional filters already capture
      general visual structures (edges, textures, shapes) relevant to faces.

Input contract (matches etl/preprocess.py output):
    NumPy array, shape (224, 224, 3), dtype float32, range [0, 1].

Output:
    NumPy array, shape (512,), dtype float32 — the raw feature vector to
    store in Fact_Image_Analysis.Feature_Vector.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# ── ImageNet normalisation statistics ─────────────────────────────────────────
# Even though our images are grayscale-duplicated, using the standard ImageNet
# mean/std is correct: each of the 3 identical channels will be normalised
# independently, but since they are all equal the effect is consistent.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# ── ResNet18 output dimension after head removal ──────────────────────────────
FEATURE_DIM: int = 512  # Must match VECTOR(512) in init.sql


def _build_transform() -> transforms.Compose:
    """
    Build the torchvision transform pipeline.

    The preprocessing module already handles resize and [0,1] normalisation,
    so here we only need to:
      1. Convert the (H, W, C) NumPy float32 array to a PyTorch Tensor.
      2. Apply ImageNet channel normalisation.
    """
    return transforms.Compose([
        transforms.ToTensor(),          # (H, W, C) float32 [0,1] → (C, H, W) float32 [0,1]
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class FeatureExtractor:
    """
    Headless ResNet18 wrapper for extracting 512-d feature vectors.

    Usage
    -----
    >>> extractor = FeatureExtractor()
    >>> vec = extractor.extract(preprocessed_numpy_image)
    >>> print(vec.shape)  # (512,)
    """

    def __init__(self, device: str | None = None) -> None:
        """
        Initialise and load the headless ResNet18.

        Parameters
        ----------
        device : 'cuda', 'cpu', or None (auto-detect).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        logger.info("Initialising FeatureExtractor on device: %s", self.device)

        # ── Load pre-trained ResNet18 ──────────────────────────────────────────
        # weights=DEFAULT uses the latest recommended ImageNet-1k weights.
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # ── Remove the classification head ────────────────────────────────────
        # ResNet18 architecture (last three layers):
        #   …
        #   avgpool : AdaptiveAvgPool2d(output_size=(1, 1))  → (B, 512, 1, 1)
        #   flatten : (added in forward())                   → (B, 512)
        #   fc      : Linear(512, 1000)                      → (B, 1000)  ← REMOVE THIS
        #
        # We replace `fc` with an Identity layer so the forward pass stops
        # at the 512-d flattened avgpool output.
        backbone.fc = nn.Identity()

        self.model = backbone.to(self.device).eval()  # eval() disables dropout/BN training mode
        self.transform = _build_transform()

        logger.info("FeatureExtractor ready. Output dimension: %d", FEATURE_DIM)

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract a feature vector from a single preprocessed image.

        Parameters
        ----------
        image : np.ndarray
            Shape (224, 224, 3), dtype float32, range [0, 1].
            This is the direct output of etl.preprocess.preprocess_image().

        Returns
        -------
        np.ndarray
            Shape (512,), dtype float32.  Ready to insert into PostgreSQL
            as a pgvector VECTOR(512) value.
        """
        # Apply torchvision normalisation and add batch dimension → (1, 3, 224, 224)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Forward pass through headless ResNet18 → (1, 512)
        features = self.model(tensor)

        # Squeeze batch dim, move to CPU, convert to NumPy
        vector: np.ndarray = features.squeeze(0).cpu().numpy()  # shape: (512,)
        return vector

    @torch.no_grad()
    def extract_batch(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """
        Extract feature vectors from a batch of preprocessed images.

        Parameters
        ----------
        images : sequence of np.ndarray
            Each array has shape (224, 224, 3), dtype float32, range [0, 1].

        Returns
        -------
        np.ndarray
            Shape (N, 512), dtype float32.
        """
        tensors = torch.stack(
            [self.transform(img) for img in images]
        ).to(self.device)  # (N, 3, 224, 224)

        features = self.model(tensors)            # (N, 512)
        return features.cpu().numpy()


# ── Module-level singleton (lazy init) ───────────────────────────────────────
_extractor: FeatureExtractor | None = None


def get_extractor(device: str | None = None) -> FeatureExtractor:
    """
    Return the module-level FeatureExtractor singleton, creating it on
    first call.  Subsequent calls return the cached instance, avoiding
    repeated model loading overhead.
    """
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor(device=device)
    return _extractor


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    from etl.preprocess import preprocess_image

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m etl.extract_features <image_path>")
        sys.exit(1)

    img_arr = preprocess_image(sys.argv[1])
    extractor = FeatureExtractor()
    vec = extractor.extract(img_arr)
    print(f"Feature vector shape : {vec.shape}")
    print(f"dtype                : {vec.dtype}")
    print(f"First 8 values       : {vec[:8]}")
