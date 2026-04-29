"""
pipeline.py
─────────────────────────────────────────────────────────────────────────────
Main ETL Orchestrator

Ties together all ETL phases in the correct execution order:
  1. Ingest  – Download FER2013 and CelebA via kagglehub
  2. Process – Preprocess each image (resize → grayscale → 3-channel duplicate)
  3. Extract – Run headless ResNet18 to get 512-d feature vectors
  4. Validate – Pass each record through Pydantic schemas
  5. Load    – UPSERT dimensions then INSERT fact rows into PostgreSQL

Run this script to execute the full pipeline end-to-end:
    python pipeline.py

Optional flags (see argparse section at the bottom):
    --dry-run    : run all steps but skip DB insertion (for testing)
    --max-images : limit images per dataset (useful during development)
    --data-root  : override the data directory
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

# ── ETL module imports ────────────────────────────────────────────────────────
from etl.ingest import ingest_all
from etl.preprocess import preprocess_batch
from etl.extract_features import FeatureExtractor
from etl.schemas import (
    DimSourceSchema,
    DimEmotionSchema,
    DimFacialAttributesSchema,
    FactImageAnalysisSchema,
)
from etl.load import (
    managed_connection,
    upsert_source,
    upsert_emotion,
    upsert_facial_attributes,
    insert_fact,
)

# ── Load .env before anything else ───────────────────────────────────────────
load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_emotion_from_path(image_path: Path) -> str | None:
    """
    FER2013 organises images into sub-folders named after the emotion label
    (e.g.  data/fer2013/train/happy/im1234.png).

    Walk up the parent chain and return the first folder name that matches
    a known FER2013 emotion label.  Returns None if no match is found.
    """
    known_emotions = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"}
    for parent in image_path.parents:
        if parent.name.lower() in known_emotions:
            return parent.name.lower()
    return None


def _resolve_celeba_image_dir(celeba_root: Path) -> Path:
    """
    Return the directory that actually contains the CelebA JPEG images.

    The Kaggle `jessicali9530/celeba-dataset` download produces a
    double-nested layout:

        <celeba_root>/
            list_attr_celeba.csv
            img_align_celeba/
                img_align_celeba/
                    000001.jpg
                    000002.jpg
                    …

    This function resolves and validates that path, raising a clear
    error if the expected structure is not found.
    """
    img_dir = celeba_root / "img_align_celeba" / "img_align_celeba"
    if img_dir.is_dir():
        return img_dir

    # Fallback: single-level nesting (some mirror downloads)
    fallback = celeba_root / "img_align_celeba"
    if fallback.is_dir():
        logger.warning(
            "Expected double-nested img_align_celeba/img_align_celeba/ not found; "
            "falling back to single-level: %s",
            fallback,
        )
        return fallback

    raise FileNotFoundError(
        f"CelebA image directory not found. Expected: {img_dir}\n"
        "Make sure the dataset was downloaded correctly via kagglehub."
    )


def _parse_celeba_annotations(celeba_root: Path) -> dict[str, dict[str, int]]:
    """
    Parse the CelebA attribute CSV into a filename-keyed lookup dict.

    The Kaggle `jessicali9530/celeba-dataset` stores attributes in:
        <celeba_root>/list_attr_celeba.csv

    Confirmed CSV format (from live dataset inspection):
        First column  : image_id      (e.g. '000001.jpg')
        Columns 2–41  : 40 binary attributes, each an int:  1 = present, -1 = absent

    Confirmed column headers (Title_Case, as they appear in the CSV):
        5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald,
        Bangs, Big_Lips, Big_Nose, Black_Hair, Blond_Hair, Blurry, Brown_Hair,
        Bushy_Eyebrows, Chubby, Double_Chin, Eyeglasses, Goatee, Gray_Hair,
        Heavy_Makeup, High_Cheekbones, Male, Mouth_Slightly_Open, Mustache,
        No_Beard, Oval_Face, Pale_Skin, Pointy_Nose, Receding_Hairline,
        Rosy_Cheeks, Sideburns, Smiling, Straight_Hair, Wavy_Hair,
        Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace,
        Wearing_Necktie, Young

    Returns
    -------
    dict[str, dict[str, int]]
        filename -> {snake_case_schema_field_name: 1 or -1}
        e.g. {'000001.jpg': {'five_o_clock_shadow': -1, 'arched_eyebrows': 1, ...}}
    """
    import pandas as pd  # local import — only needed here

    # ── Explicit column rename map ────────────────────────────────────────────
    # Built directly from the real CSV headers confirmed via dataset inspection.
    # Using an explicit dict (not a programmatic transform) ensures correctness
    # even if Kaggle ever updates the CSV header formatting.
    _CSV_TO_SCHEMA: dict[str, str] = {
        "5_o_Clock_Shadow":    "five_o_clock_shadow",   # '5' -> 'five' (only special case)
        "Arched_Eyebrows":     "arched_eyebrows",
        "Attractive":          "attractive",
        "Bags_Under_Eyes":     "bags_under_eyes",
        "Bald":                "bald",
        "Bangs":               "bangs",
        "Big_Lips":            "big_lips",
        "Big_Nose":            "big_nose",
        "Black_Hair":          "black_hair",
        "Blond_Hair":          "blond_hair",
        "Blurry":              "blurry",
        "Brown_Hair":          "brown_hair",
        "Bushy_Eyebrows":      "bushy_eyebrows",
        "Chubby":              "chubby",
        "Double_Chin":         "double_chin",
        "Eyeglasses":          "eyeglasses",
        "Goatee":              "goatee",
        "Gray_Hair":           "gray_hair",
        "Heavy_Makeup":        "heavy_makeup",
        "High_Cheekbones":     "high_cheekbones",
        "Male":                "male",
        "Mouth_Slightly_Open": "mouth_slightly_open",
        "Mustache":            "mustache",
        "Narrow_Eyes":         "narrow_eyes",
        "No_Beard":            "no_beard",
        "Oval_Face":           "oval_face",
        "Pale_Skin":           "pale_skin",
        "Pointy_Nose":         "pointy_nose",
        "Receding_Hairline":   "receding_hairline",
        "Rosy_Cheeks":         "rosy_cheeks",
        "Sideburns":           "sideburns",
        "Smiling":             "smiling",
        "Straight_Hair":       "straight_hair",
        "Wavy_Hair":           "wavy_hair",
        "Wearing_Earrings":    "wearing_earrings",
        "Wearing_Hat":         "wearing_hat",
        "Wearing_Lipstick":    "wearing_lipstick",
        "Wearing_Necklace":    "wearing_necklace",
        "Wearing_Necktie":     "wearing_necktie",
        "Young":               "young",
    }  # 40 attributes total — matches Dim_Facial_Attributes columns exactly

    # ── Locate the CSV file ───────────────────────────────────────────────────
    # kagglehub downloads to:
    #   ~/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2/
    # so list_attr_celeba.csv sits directly in celeba_root.
    candidates = [
        celeba_root / "list_attr_celeba.csv",
        celeba_root / "anno" / "list_attr_celeba.csv",
        celeba_root / "celeba" / "list_attr_celeba.csv",
    ]
    attr_file: Path | None = None
    for c in candidates:
        if c.exists():
            attr_file = c
            break

    if attr_file is None:
        logger.warning(
            "list_attr_celeba.csv not found under %s. "
            "Facial attribute rows will be empty for all CelebA images.",
            celeba_root,
        )
        return {}

    logger.info("Reading CelebA attributes from: %s", attr_file)

    # ── Read and rename ───────────────────────────────────────────────────────
    df = pd.read_csv(attr_file)

    # Normalise the first column to a stable name regardless of what Kaggle calls it
    df = df.rename(columns={df.columns[0]: "image_id"})

    # Apply the explicit map — only rename columns that exist; warn about any missing
    rename_map = {
        csv_col: schema_col
        for csv_col, schema_col in _CSV_TO_SCHEMA.items()
        if csv_col in df.columns
    }
    missing = set(_CSV_TO_SCHEMA.keys()) - set(df.columns)
    if missing:
        logger.warning(
            "%d expected CelebA attribute column(s) not found in CSV: %s",
            len(missing), sorted(missing),
        )
    df = df.rename(columns=rename_map)

    # ── Build lookup dict (vectorised — avoids slow iterrows on 202k rows) ───
    schema_cols = list(rename_map.values())
    df = df.set_index("image_id")[schema_cols].astype(int)
    annotations: dict[str, dict[str, int]] = df.to_dict(orient="index")

    logger.info(
        "Parsed %d CelebA attribute annotations (%d attributes each).",
        len(annotations), len(schema_cols),
    )
    return annotations


# ─────────────────────────────────────────────────────────────────────────────
# Batch iterator helper
# ─────────────────────────────────────────────────────────────────────────────

from itertools import islice

def _batch_iter(iterable, batch_size: int):
    """
    Lazily yield successive chunks of `batch_size` items from any iterable.

    Works with generators (unlike list slicing), so it never materialises the
    entire dataset into memory — critical for 200k-image datasets.

    Example
    -------
    >>> list(_batch_iter(range(10), 3))
    [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]
    """
    it = iter(iterable)
    while chunk := list(islice(it, batch_size)):
        yield chunk


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset ETL functions
# ─────────────────────────────────────────────────────────────────────────────

def run_fer2013_etl(
    fer_root: Path,
    source_id: int,
    emotion_id_map: dict[str, int],
    extractor: FeatureExtractor,
    conn,
    *,
    max_images: int | None,
    dry_run: bool,
    batch_size: int = 64,
) -> int:
    """
    Process all FER2013 images and load them into the fact table.

    Performance strategy
    --------------------
    Images are processed in mini-batches (default 64) so the CNN forward
    pass runs on a full tensor batch instead of one image at a time.
    Database commits happen once per batch (not once per image), reducing
    PostgreSQL round-trips by ~batch_size fold.
    """
    inserted   = 0
    skipped    = 0
    image_gen  = preprocess_batch(fer_root, max_images=max_images)

    for batch in _batch_iter(image_gen, batch_size):
        paths   = [img_path for img_path, _ in batch]
        tensors = [tensor   for _, tensor    in batch]

        # ── Batched CNN inference ─────────────────────────────────────────────
        # extract_batch() stacks all tensors into a single (B, 3, 224, 224)
        # tensor and runs one forward pass, which is far more efficient than
        # B sequential single-image forward passes on CPU or GPU.
        try:
            vectors = extractor.extract_batch(tensors)  # shape: (B, 512)
        except Exception as exc:
            logger.warning("Feature extraction failed for batch starting at %s: %s",
                           paths[0].name, exc)
            skipped += len(paths)
            continue

        # ── Validate and accumulate fact records ──────────────────────────────
        facts: list[FactImageAnalysisSchema] = []
        for img_path, vector in zip(paths, vectors):
            emotion_label = _detect_emotion_from_path(img_path)
            emotion_id    = emotion_id_map.get(emotion_label) if emotion_label else None
            try:
                facts.append(FactImageAnalysisSchema(
                    Source_ID      = source_id,
                    Emotion_ID     = emotion_id,
                    Attribute_ID   = None,
                    Cluster_ID     = None,
                    File_Name      = img_path.name,
                    Feature_Vector = vector.tolist(),
                ))
            except ValidationError as exc:
                logger.warning("Validation failed for %s: %s", img_path.name, exc)
                skipped += 1

        # ── Single batch commit ───────────────────────────────────────────────
        if not dry_run and facts:
            for fact in facts:
                insert_fact(conn, fact)
            conn.commit()   # one commit per batch, not one per image

        inserted += len(facts)
        logger.debug("FER2013 batch done — cumulative inserted: %d", inserted)

    logger.info("FER2013 ETL complete — inserted: %d, skipped: %d", inserted, skipped)
    return inserted


def run_celeba_etl(
    celeba_root: Path,
    source_id: int,
    annotations: dict[str, dict[str, int]],
    extractor: FeatureExtractor,
    conn,
    *,
    max_images: int | None,
    dry_run: bool,
    batch_size: int = 64,
) -> int:
    """
    Process all CelebA images and load them into the fact table.

    Images are read from the double-nested directory:
        <celeba_root>/img_align_celeba/img_align_celeba/

    Performance strategy
    --------------------
    Same batched approach as FER2013: CNN inference runs on a full batch
    (default 64 images) per forward pass. Attribute dimension rows are
    inserted individually (they have no natural unique key for UPSERT)
    but fact rows and their attribute inserts are committed once per batch.
    """
    inserted   = 0
    skipped    = 0

    # Resolve the actual image directory (handles double-nesting automatically)
    img_dir = _resolve_celeba_image_dir(celeba_root)
    logger.info("CelebA images resolved to: %s", img_dir)

    image_gen = preprocess_batch(img_dir, max_images=max_images)

    for batch in _batch_iter(image_gen, batch_size):
        paths   = [img_path for img_path, _ in batch]
        tensors = [tensor   for _, tensor    in batch]

        # ── Batched CNN inference ─────────────────────────────────────────────
        try:
            vectors = extractor.extract_batch(tensors)  # shape: (B, 512)
        except Exception as exc:
            logger.warning("Feature extraction failed for batch starting at %s: %s",
                           paths[0].name, exc)
            skipped += len(paths)
            continue

        # ── Per-image: insert attribute row, build fact record ────────────────
        facts: list[FactImageAnalysisSchema] = []
        for img_path, vector in zip(paths, vectors):
            attribute_id: int | None = None

            if img_path.name in annotations:
                raw_attrs = annotations[img_path.name]
                try:
                    attrs_schema = DimFacialAttributesSchema.from_celeba_row(raw_attrs)
                except ValidationError as exc:
                    logger.warning("Attribute validation failed for %s: %s",
                                   img_path.name, exc)
                    attrs_schema = None

                # Attribute inserts are cheap individual rows; commit happens
                # at end of batch so they're in the same transaction as facts.
                if attrs_schema is not None and not dry_run:
                    attribute_id = upsert_facial_attributes(conn, attrs_schema)

            try:
                facts.append(FactImageAnalysisSchema(
                    Source_ID      = source_id,
                    Emotion_ID     = None,
                    Attribute_ID   = attribute_id,
                    Cluster_ID     = None,
                    File_Name      = img_path.name,
                    Feature_Vector = vector.tolist(),
                ))
            except ValidationError as exc:
                logger.warning("Validation failed for %s: %s", img_path.name, exc)
                skipped += 1

        # ── Single batch commit (attributes + facts together) ─────────────────
        if not dry_run and facts:
            for fact in facts:
                insert_fact(conn, fact)
            conn.commit()   # one commit per batch, not one per image

        inserted += len(facts)
        logger.debug("CelebA batch done — cumulative inserted: %d", inserted)

    logger.info("CelebA ETL complete — inserted: %d, skipped: %d", inserted, skipped)
    return inserted


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    data_root   = Path(args.data_root)
    max_images  = args.max_images
    dry_run     = args.dry_run

    if dry_run:
        logger.info("=== DRY RUN MODE — no data will be written to the database ===")

    # ── Phase 1: Ingest ───────────────────────────────────────────────────────
    logger.info("=== Phase 1: Data Ingestion ===")
    dataset_paths = ingest_all(data_root=data_root)
    fer_root    = dataset_paths["fer2013"]
    celeba_root = dataset_paths["celeba"]

    # ── Phase 2: Feature Extractor Init ──────────────────────────────────────
    logger.info("=== Phase 2: Loading Feature Extractor ===")
    extractor = FeatureExtractor()

    # ── Phase 3: CelebA annotations ──────────────────────────────────────────
    logger.info("=== Phase 3: Parsing CelebA Annotations ===")
    celeba_annotations = _parse_celeba_annotations(celeba_root)

    # ── Phase 4: ETL with DB connection ──────────────────────────────────────
    logger.info("=== Phase 4: ETL → Database Load ===")

    with managed_connection() as conn:
        # Upsert source dimension rows and cache their IDs
        fer_source_id = upsert_source(
            conn,
            DimSourceSchema(
                Source_Name="FER2013",
                Description="Facial Expression Recognition 2013 dataset (Kaggle).",
            ),
        )
        conn.commit()

        celeba_source_id = upsert_source(
            conn,
            DimSourceSchema(
                Source_Name="CelebA",
                Description="Large-scale CelebFaces Attributes dataset (Kaggle).",
            ),
        )
        conn.commit()

        logger.info("Source IDs — FER2013: %d, CelebA: %d", fer_source_id, celeba_source_id)

        # Build an emotion label → ID map from the DB (they were seeded in init.sql)
        with conn.cursor() as cur:
            cur.execute("SELECT Emotion_Label, Emotion_ID FROM Dim_Emotion;")
            emotion_id_map: dict[str, int] = {row[0]: row[1] for row in cur.fetchall()}
        logger.info("Loaded %d emotion dimension entries.", len(emotion_id_map))

        # Run per-dataset ETL
        fer_count = run_fer2013_etl(
            fer_root, fer_source_id, emotion_id_map, extractor, conn,
            max_images=max_images, dry_run=dry_run, batch_size=args.batch_size,
        )
        celeba_count = run_celeba_etl(
            celeba_root, celeba_source_id, celeba_annotations, extractor, conn,
            max_images=max_images, dry_run=dry_run, batch_size=args.batch_size,
        )

    logger.info(
        "=== Pipeline complete — FER2013: %d rows, CelebA: %d rows ===",
        fer_count, celeba_count,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Warehouse ETL Pipeline — Ingest, Preprocess, Extract, Load."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", "./data"),
        help="Root directory for downloaded datasets (default: ./data or $DATA_ROOT).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit images processed per dataset. Omit for full run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all steps except DB insertion. Useful for testing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Images per CNN forward pass (default: 64). Increase on GPU, decrease if OOM.",
    )
    parsed = parser.parse_args()
    main(parsed)
