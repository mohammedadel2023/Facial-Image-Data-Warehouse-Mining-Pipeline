"""
data_mining.py
─────────────────────────────────────────────────────────────────────────────
Phase 3, Step 8 – Machine Learning Stub

This module is the SKELETON for your custom data mining logic.
The database plumbing is fully implemented so you can focus entirely on
writing the mathematics for K-Means and Apriori.

Execution flow (once you fill in the TODOs):
  1. fetch_feature_vectors()  → loads Image_ID + Feature_Vector from DB
  2. [YOUR K-Means logic]     → assigns each image to a cluster
  3. [YOUR Apriori logic]     → mines association rules from CelebA attributes
  4. save_cluster_assignments() → writes Cluster_ID back to the fact table
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────────
# Database connection
# ─────────────────────────────────────────────────────────────────────────────

def _get_connection() -> psycopg2.extensions.connection:
    """Open and return a psycopg2 connection using environment variables."""
    return psycopg2.connect(
        host     = os.environ.get("POSTGRES_HOST",     "localhost"),
        port     = int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname   = os.environ.get("POSTGRES_DB",       "datawarehouse"),
        user     = os.environ.get("POSTGRES_USER",     "dw_user"),
        password = os.environ.get("POSTGRES_PASSWORD", "dw_password"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Load feature vectors from the Data Warehouse
# ─────────────────────────────────────────────────────────────────────────────

def fetch_feature_vectors() -> pd.DataFrame:
    """
    Query Fact_Image_Analysis and return a DataFrame with two columns:

    Columns
    -------
    Image_ID       : int64  — primary key from the fact table
    Feature_Vector : object — each cell holds a list[float] of length 512

    This DataFrame is the direct input to your custom K-Means algorithm.

    Returns
    -------
    pd.DataFrame
        Shape: (N_images, 2)
        Column types: Image_ID → int64, Feature_Vector → list[float]
    """
    logger.info("Fetching feature vectors from Fact_Image_Analysis …")

    conn = _get_connection()
    try:
        # pgvector returns VECTOR columns as a Python list via psycopg2
        # when the pgvector psycopg2 adapter is registered (see note below).
        #
        # NOTE: If vectors arrive as strings like '[0.12, -0.34, …]', call:
        #   import pgvector.psycopg2; pgvector.psycopg2.register_vector(conn)
        # before executing the query, then each cell will be a numpy array.
        query = """
            SELECT
                Image_ID,
                Feature_Vector
            FROM
                Fact_Image_Analysis
            ORDER BY
                Image_ID;
        """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    logger.info("Loaded %d feature vectors (dimension: 512).", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – YOUR CUSTOM MACHINE LEARNING LOGIC GOES HERE
# ─────────────────────────────────────────────────────────────────────────────

def run_clustering(df: pd.DataFrame, k: int = 10) -> pd.Series:
    """
    Assign each image to a cluster using your custom K-Means implementation.

    Parameters
    ----------
    df : DataFrame returned by fetch_feature_vectors().
         df['Feature_Vector'] contains the 512-d vectors.
    k  : number of clusters (hyperparameter to tune later).

    Returns
    -------
    pd.Series
        Index matches df.index; values are integer cluster labels (0 … k-1).
        Name: 'Cluster_Label'

    # =========================================================================
    # TODO: Implement custom K-Means and Apriori logic here.
    # =========================================================================
    #
    # Suggested K-Means skeleton:
    # ---------------------------
    # import numpy as np
    #
    # X = np.stack(df['Feature_Vector'].values)   # shape: (N, 512)
    #
    # Step 1 – Initialisation (e.g., K-Means++ or random)
    # centroids = X[np.random.choice(len(X), k, replace=False)]
    #
    # for iteration in range(max_iterations):
    #
    #     Step 2 – Assignment: compute Euclidean distances, assign to nearest centroid
    #     distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # (N, k)
    #     labels = np.argmin(distances, axis=1)                              # (N,)
    #
    #     Step 3 – Update: recompute centroids as cluster means
    #     new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
    #
    #     Step 4 – Convergence check
    #     if np.allclose(centroids, new_centroids, atol=1e-6):
    #         logger.info("K-Means converged at iteration %d.", iteration)
    #         break
    #     centroids = new_centroids
    #
    # return pd.Series(labels, index=df.index, name='Cluster_Label')
    #
    # ---------------------------
    # Suggested Apriori skeleton (for CelebA boolean attributes):
    # ---------------------------
    # Load attribute rows:
    #   attr_df = pd.read_sql("SELECT * FROM Dim_Facial_Attributes", conn)
    #
    # Convert to transaction format (list of frozensets of attribute names).
    # Implement frequent itemset generation:
    #   L1 = {item: count / N for item in all_items if count/N >= min_support}
    #   Lk = join(L_{k-1}) → prune with subset check → scan for support
    # Generate association rules from frequent itemsets.
    #   confidence(A→B) = support(A∪B) / support(A)
    #   lift(A→B)       = confidence(A→B) / support(B)
    """
    raise NotImplementedError(
        "Custom K-Means logic not yet implemented. "
        "Fill in the TODO block in run_clustering()."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Write cluster assignments back to the Data Warehouse
# ─────────────────────────────────────────────────────────────────────────────

def save_cluster_assignments(
    image_ids: list[int],
    cluster_labels: list[int],
) -> None:
    """
    Persist K-Means cluster assignments to Fact_Image_Analysis.

    For each (Image_ID, Cluster_Label) pair:
      1. UPSERT a row in Dim_Cluster to get/create the Cluster_ID.
      2. UPDATE Fact_Image_Analysis.Cluster_ID for that Image_ID.

    Parameters
    ----------
    image_ids      : list of Image_ID values from the fact table.
    cluster_labels : parallel list of integer cluster labels (0 … k-1)
                     returned by run_clustering().

    Example
    -------
    >>> image_ids = [1, 2, 3]
    >>> cluster_labels = [0, 2, 0]   # images 1 and 3 are in cluster 0
    >>> save_cluster_assignments(image_ids, cluster_labels)
    """
    if len(image_ids) != len(cluster_labels):
        raise ValueError(
            f"image_ids and cluster_labels must have equal length. "
            f"Got {len(image_ids)} vs {len(cluster_labels)}."
        )

    conn = _get_connection()
    try:
        # Build Dim_Cluster rows for each unique label seen
        unique_labels = sorted(set(cluster_labels))
        cluster_id_map: dict[int, int] = {}   # label → Cluster_ID (DB PK)

        with conn.cursor() as cur:
            for label in unique_labels:
                # UPSERT cluster dimension row
                cur.execute(
                    """
                    INSERT INTO Dim_Cluster (Cluster_Label, Description)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    RETURNING Cluster_ID;
                    """,
                    (f"Cluster_{label}", f"Auto-generated by K-Means, label={label}"),
                )
                row = cur.fetchone()
                if row:
                    cluster_id_map[label] = row[0]
                else:
                    # Row already existed — fetch its ID
                    cur.execute(
                        "SELECT Cluster_ID FROM Dim_Cluster WHERE Cluster_Label = %s;",
                        (f"Cluster_{label}",),
                    )
                    cluster_id_map[label] = cur.fetchone()[0]

        # Bulk UPDATE fact rows using psycopg2 execute_batch for efficiency
        updates = [
            (cluster_id_map[lbl], img_id)
            for img_id, lbl in zip(image_ids, cluster_labels)
        ]

        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                "UPDATE Fact_Image_Analysis SET Cluster_ID = %s WHERE Image_ID = %s;",
                updates,
                page_size=500,
            )

        conn.commit()
        logger.info(
            "Saved %d cluster assignments across %d clusters.",
            len(image_ids), len(unique_labels),
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1 – Load data
    feature_df = fetch_feature_vectors()
    print(feature_df.head())
    print(f"\nDataFrame shape: {feature_df.shape}")

    # Steps 2 & 3 – Uncomment these once you have implemented run_clustering():
    # cluster_series = run_clustering(feature_df, k=10)
    # save_cluster_assignments(
    #     image_ids      = feature_df["Image_ID"].tolist(),
    #     cluster_labels = cluster_series.tolist(),
    # )
