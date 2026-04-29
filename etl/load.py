"""
etl/load.py
─────────────────────────────────────────────────────────────────────────────
Phase 2, Step 7 – Database Loading

Handles all PostgreSQL insertion logic using psycopg2.

Strategy
────────
• Dimension tables use UPSERT (ON CONFLICT DO NOTHING) so the pipeline is
  idempotent – re-running it won't create duplicate dimension rows.
• Fact rows use plain INSERT.  Duplicate fact detection is intentionally
  omitted here (it would require a unique index on File_Name + Source_ID),
  but the pipeline.py orchestrator only calls this after a fresh ingest.
• All DB credentials are read from environment variables (via .env) to keep
  secrets out of source code.
• Every public function accepts an optional psycopg2 connection so callers
  can share a transaction if needed.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator, Optional

import psycopg2
import psycopg2.extras

from etl.schemas import (
    DimClusterSchema,
    DimEmotionSchema,
    DimFacialAttributesSchema,
    DimSourceSchema,
    FactImageAnalysisSchema,
)

logger = logging.getLogger(__name__)


# ── Connection helpers ────────────────────────────────────────────────────────

def _get_dsn() -> str:
    """Build a libpq connection string from environment variables."""
    return (
        f"host={os.environ.get('POSTGRES_HOST', 'localhost')} "
        f"port={os.environ.get('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ.get('POSTGRES_DB', 'datawarehouse')} "
        f"user={os.environ.get('POSTGRES_USER', 'dw_user')} "
        f"password={os.environ.get('POSTGRES_PASSWORD', 'dw_password')}"
    )


def get_connection() -> psycopg2.extensions.connection:
    """
    Open and return a new psycopg2 connection.

    The caller is responsible for calling .close() or using it as a
    context manager.  For convenience, prefer using `managed_connection()`.
    """
    dsn = _get_dsn()
    logger.debug("Connecting to PostgreSQL …")
    return psycopg2.connect(dsn)


@contextmanager
def managed_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """
    Context manager that opens a connection and guarantees it is closed on exit.

    Usage
    -----
    >>> with managed_connection() as conn:
    ...     upsert_source(conn, DimSourceSchema(Source_Name="FER2013"))
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Dimension UPSERT functions ────────────────────────────────────────────────

def upsert_source(
    conn: psycopg2.extensions.connection,
    source: DimSourceSchema,
) -> int:
    """
    UPSERT a row into Dim_Source and return the Source_ID.

    Uses ON CONFLICT (Source_Name) DO NOTHING + a subsequent SELECT to
    reliably return the ID whether the row was inserted or already existed.

    Returns
    -------
    int
        The Source_ID of the (possibly pre-existing) row.
    """
    sql_upsert = """
        INSERT INTO Dim_Source (Source_Name, Description)
        VALUES (%s, %s)
        ON CONFLICT (Source_Name) DO NOTHING;
    """
    sql_select = "SELECT Source_ID FROM Dim_Source WHERE Source_Name = %s;"

    with conn.cursor() as cur:
        cur.execute(sql_upsert, (source.Source_Name, source.Description))
        cur.execute(sql_select, (source.Source_Name,))
        row = cur.fetchone()

    return row[0]


def upsert_emotion(
    conn: psycopg2.extensions.connection,
    emotion: DimEmotionSchema,
) -> int:
    """
    UPSERT a row into Dim_Emotion and return the Emotion_ID.

    Returns
    -------
    int
        The Emotion_ID of the (possibly pre-existing) row.
    """
    sql_upsert = """
        INSERT INTO Dim_Emotion (Emotion_Label)
        VALUES (%s)
        ON CONFLICT (Emotion_Label) DO NOTHING;
    """
    sql_select = "SELECT Emotion_ID FROM Dim_Emotion WHERE Emotion_Label = %s;"

    with conn.cursor() as cur:
        cur.execute(sql_upsert, (emotion.Emotion_Label,))
        cur.execute(sql_select, (emotion.Emotion_Label,))
        row = cur.fetchone()

    return row[0]


def upsert_facial_attributes(
    conn: psycopg2.extensions.connection,
    attrs: DimFacialAttributesSchema,
) -> int:
    """
    INSERT a CelebA facial attribute row into Dim_Facial_Attributes and
    return the Attribute_ID.

    NOTE: Unlike Source and Emotion, attribute rows are unique per image —
    there is no natural unique key across all 40 columns — so this uses a
    plain INSERT and returns the auto-generated Attribute_ID.

    Returns
    -------
    int
        The newly created Attribute_ID.
    """
    cols = list(attrs.model_fields.keys())
    placeholders = ", ".join(["%s"] * len(cols))
    col_names = ", ".join(cols)

    sql = f"""
        INSERT INTO Dim_Facial_Attributes ({col_names})
        VALUES ({placeholders})
        RETURNING Attribute_ID;
    """
    values = [getattr(attrs, col) for col in cols]

    with conn.cursor() as cur:
        cur.execute(sql, values)
        row = cur.fetchone()

    return row[0]


def upsert_cluster(
    conn: psycopg2.extensions.connection,
    cluster: DimClusterSchema,
) -> int:
    """
    INSERT a Dim_Cluster row and return the Cluster_ID.

    Called by data_mining.py after K-Means assigns cluster labels.

    Returns
    -------
    int
        The newly created Cluster_ID.
    """
    sql = """
        INSERT INTO Dim_Cluster (Cluster_Label, Description)
        VALUES (%s, %s)
        RETURNING Cluster_ID;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (cluster.Cluster_Label, cluster.Description))
        row = cur.fetchone()

    return row[0]


# ── Fact INSERT function ──────────────────────────────────────────────────────

def insert_fact(
    conn: psycopg2.extensions.connection,
    fact: FactImageAnalysisSchema,
) -> int:
    """
    Insert a single row into Fact_Image_Analysis.

    The Feature_Vector is passed as a pgvector literal string so psycopg2
    does not need to know about the vector type — it is treated as a plain
    text value that PostgreSQL then casts to VECTOR(512).

    Returns
    -------
    int
        The auto-generated Image_ID of the inserted row.
    """
    sql = """
        INSERT INTO Fact_Image_Analysis
            (Source_ID, Emotion_ID, Attribute_ID, Cluster_ID, File_Name, Feature_Vector)
        VALUES
            (%s, %s, %s, %s, %s, %s::vector)
        RETURNING Image_ID;
    """
    values = (
        fact.Source_ID,
        fact.Emotion_ID,        # None → NULL  (handled automatically by psycopg2)
        fact.Attribute_ID,      # None → NULL
        fact.Cluster_ID,        # None → NULL  (always None at ingestion time)
        fact.File_Name,
        fact.vector_as_pg_literal(),
    )

    with conn.cursor() as cur:
        cur.execute(sql, values)
        row = cur.fetchone()

    return row[0]


def insert_facts_batch(
    conn: psycopg2.extensions.connection,
    facts: list[FactImageAnalysisSchema],
    *,
    batch_size: int = 500,
) -> int:
    """
    Bulk-insert a list of fact rows, committing in batches to balance
    memory use against transaction overhead.

    Parameters
    ----------
    conn       : open psycopg2 connection (managed externally).
    facts      : list of validated FactImageAnalysisSchema objects.
    batch_size : number of rows per commit.

    Returns
    -------
    int
        Total number of rows successfully inserted.
    """
    inserted = 0
    for i in range(0, len(facts), batch_size):
        batch = facts[i : i + batch_size]
        for fact in batch:
            insert_fact(conn, fact)
            inserted += 1
        conn.commit()
        logger.info("Committed batch %d / %d rows", i + len(batch), len(facts))

    return inserted


# ── Cluster write-back (called from data_mining.py) ──────────────────────────

def update_cluster_id(
    conn: psycopg2.extensions.connection,
    image_id: int,
    cluster_id: int,
) -> None:
    """
    Write a Cluster_ID assignment back to a Fact_Image_Analysis row.

    Called by data_mining.py after custom K-Means assigns each image to
    a cluster.  Uses a parameterised UPDATE to prevent SQL injection.

    Parameters
    ----------
    conn       : open psycopg2 connection.
    image_id   : PK of the fact row to update.
    cluster_id : FK to Dim_Cluster to assign.
    """
    sql = """
        UPDATE Fact_Image_Analysis
        SET    Cluster_ID = %s
        WHERE  Image_ID   = %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (cluster_id, image_id))
