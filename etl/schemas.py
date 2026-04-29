"""
etl/schemas.py
─────────────────────────────────────────────────────────────────────────────
Phase 2, Step 6 – Pydantic Validation Schemas

All records are validated against these schemas BEFORE any database
insertion attempt.  If a record fails validation, a Pydantic ValidationError
is raised and the ETL pipeline can decide to skip or quarantine that record
without crashing.

Design notes
────────────
• Dimension schemas use strict types so bad data is caught early.
• FactImageAnalysisSchema uses Optional[int] for Emotion_ID and
  Attribute_ID.  This is the key architectural decision that lets a
  single Fact table gracefully store:
    – FER2013 rows  → Emotion_ID present,  Attribute_ID = None
    – CelebA rows   → Emotion_ID = None,   Attribute_ID present
• Feature_Vector is stored as a List[float] in Python but the insertion
  code converts it to the pgvector wire format (a string like '[0.1, …]').
• All schemas inherit from BaseModel with model_config = ConfigDict(
  strict=True) to prevent silent type coercion.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Strict base class ─────────────────────────────────────────────────────────
class _StrictBase(BaseModel):
    """All DW schemas inherit strict mode to prevent silent coercions."""
    model_config = ConfigDict(strict=True)


# ============================================================
# DIMENSION SCHEMAS
# ============================================================

class DimSourceSchema(_StrictBase):
    """
    Validates a row before insertion into Dim_Source.

    Fields
    ------
    Source_Name : unique human-readable label, e.g. 'FER2013', 'CelebA'.
    Description : optional free-text description of the source.
    """
    Source_Name: str = Field(..., min_length=1, max_length=100)
    Description: Optional[str] = None


class DimEmotionSchema(_StrictBase):
    """
    Validates a row before insertion into Dim_Emotion.

    Fields
    ------
    Emotion_Label : canonical lowercase emotion name, e.g. 'happy', 'sad'.
                    Must be non-empty and at most 50 characters.
    """
    Emotion_Label: str = Field(..., min_length=1, max_length=50)

    @field_validator("Emotion_Label")
    @classmethod
    def normalise_label(cls, v: str) -> str:
        """Force lowercase to ensure consistent DB keys."""
        return v.strip().lower()


class DimFacialAttributesSchema(_StrictBase):
    """
    Validates a CelebA attribute row before insertion into Dim_Facial_Attributes.
    All 40 CelebA binary attributes are represented.  CelebA encodes them as
    1 (present) or -1 (absent); the validator converts to booleans.
    """
    five_o_clock_shadow:  bool = False
    arched_eyebrows:      bool = False
    attractive:           bool = False
    bags_under_eyes:      bool = False
    bald:                 bool = False
    bangs:                bool = False
    big_lips:             bool = False
    big_nose:             bool = False
    black_hair:           bool = False
    blond_hair:           bool = False
    blurry:               bool = False
    brown_hair:           bool = False
    bushy_eyebrows:       bool = False
    chubby:               bool = False
    double_chin:          bool = False
    eyeglasses:           bool = False
    goatee:               bool = False
    gray_hair:            bool = False
    heavy_makeup:         bool = False
    high_cheekbones:      bool = False
    male:                 bool = False
    mouth_slightly_open:  bool = False
    mustache:             bool = False
    narrow_eyes:          bool = False
    no_beard:             bool = False
    oval_face:            bool = False
    pale_skin:            bool = False
    pointy_nose:          bool = False
    receding_hairline:    bool = False
    rosy_cheeks:          bool = False
    sideburns:            bool = False
    smiling:              bool = False
    straight_hair:        bool = False
    wavy_hair:            bool = False
    wearing_earrings:     bool = False
    wearing_hat:          bool = False
    wearing_lipstick:     bool = False
    wearing_necklace:     bool = False
    wearing_necktie:      bool = False
    young:                bool = False

    @classmethod
    def from_celeba_row(cls, raw: dict[str, int]) -> "DimFacialAttributesSchema":
        """
        Convenience constructor that converts the CelebA annotation convention
        (1 = present, -1 = absent) into booleans.

        Parameters
        ----------
        raw : dict mapping attribute name (snake_case) → int (1 or -1).

        Returns
        -------
        DimFacialAttributesSchema
        """
        bool_row = {k: (v == 1) for k, v in raw.items()}
        return cls(**bool_row)


class DimClusterSchema(_StrictBase):
    """
    Validates a cluster dimension row.  Cluster rows are written AFTER
    the data-mining step assigns cluster IDs.

    Fields
    ------
    Cluster_Label : short human-readable label (optional).
    Description   : longer description (optional).
    """
    Cluster_Label: Optional[str] = Field(None, max_length=100)
    Description:   Optional[str] = None


# ============================================================
# FACT SCHEMA
# ============================================================

class FactImageAnalysisSchema(_StrictBase):
    """
    Validates a row before insertion into Fact_Image_Analysis.

    Key design decision — Nullable dimension FKs
    ─────────────────────────────────────────────
    • Emotion_ID   is Optional[int]: present for FER2013, None for CelebA.
    • Attribute_ID is Optional[int]: present for CelebA,  None for FER2013.
    • Cluster_ID   is Optional[int]: None until data_mining.py writes it back.

    This schema models the heterogeneous nature of the two source datasets
    without requiring separate fact tables or NULL-heavy wide tables.

    Fields
    ------
    Source_ID      : FK → Dim_Source.Source_ID        (required)
    Emotion_ID     : FK → Dim_Emotion.Emotion_ID      (optional)
    Attribute_ID   : FK → Dim_Facial_Attributes       (optional)
    Cluster_ID     : FK → Dim_Cluster.Cluster_ID      (optional, set later)
    File_Name      : original filename for traceability (required)
    Feature_Vector : 512-d float list from ResNet18    (required)
    """
    Source_ID:      int            = Field(..., gt=0)
    Emotion_ID:     Optional[int]  = Field(None, gt=0)
    Attribute_ID:   Optional[int]  = Field(None, gt=0)
    Cluster_ID:     Optional[int]  = Field(None, gt=0)
    File_Name:      str            = Field(..., min_length=1, max_length=512)
    Feature_Vector: List[float]    = Field(..., min_length=512, max_length=512)

    @field_validator("Feature_Vector")
    @classmethod
    def validate_vector_length(cls, v: list[float]) -> list[float]:
        """Explicit guard – Pydantic's min_length/max_length already covers this,
        but an explicit check with a clear error message aids debugging."""
        if len(v) != 512:
            raise ValueError(
                f"Feature_Vector must have exactly 512 elements; got {len(v)}."
            )
        return v

    def vector_as_pg_literal(self) -> str:
        """
        Return the feature vector as a PostgreSQL pgvector literal string.

        pgvector expects the format: '[0.123, -0.456, …]'
        This is used directly in the psycopg2 INSERT statement.
        """
        return "[" + ",".join(f"{x:.8f}" for x in self.Feature_Vector) + "]"
