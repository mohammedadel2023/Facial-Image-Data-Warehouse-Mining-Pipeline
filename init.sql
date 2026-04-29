-- ================================================================
-- init.sql
-- Data Warehouse Star Schema Initialisation
-- Executed automatically on first PostgreSQL container start.
-- ================================================================

-- ----------------------------------------------------------------
-- 0. Enable the pgvector extension
-- ----------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;

-- ================================================================
-- DIMENSION TABLES
-- ================================================================

-- ----------------------------------------------------------------
-- 1. Dim_Source  – Where did the image come from?
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Dim_Source (
    Source_ID   SERIAL PRIMARY KEY,
    Source_Name VARCHAR(100) NOT NULL UNIQUE,    -- e.g. 'FER2013', 'CelebA'
    Description TEXT
);

-- ----------------------------------------------------------------
-- 2. Dim_Emotion – Emotion label (FER2013 specific)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Dim_Emotion (
    Emotion_ID    SERIAL PRIMARY KEY,
    Emotion_Label VARCHAR(50) NOT NULL UNIQUE     -- e.g. 'happy', 'sad', 'angry'
);

-- Pre-populate standard FER2013 emotion labels
INSERT INTO Dim_Emotion (Emotion_Label) VALUES
    ('angry'),
    ('disgust'),
    ('fear'),
    ('happy'),
    ('sad'),
    ('surprise'),
    ('neutral')
ON CONFLICT (Emotion_Label) DO NOTHING;

-- ----------------------------------------------------------------
-- 3. Dim_Facial_Attributes – 40 binary CelebA attribute columns
--    Each column stores TRUE/FALSE per the CelebA annotation file.
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Dim_Facial_Attributes (
    Attribute_ID            SERIAL  PRIMARY KEY,

    -- CelebA 40 binary attribute columns (snake_case mirrors CelebA header)
    five_o_clock_shadow     BOOLEAN DEFAULT FALSE,
    arched_eyebrows         BOOLEAN DEFAULT FALSE,
    attractive              BOOLEAN DEFAULT FALSE,
    bags_under_eyes         BOOLEAN DEFAULT FALSE,
    bald                    BOOLEAN DEFAULT FALSE,
    bangs                   BOOLEAN DEFAULT FALSE,
    big_lips                BOOLEAN DEFAULT FALSE,
    big_nose                BOOLEAN DEFAULT FALSE,
    black_hair              BOOLEAN DEFAULT FALSE,
    blond_hair              BOOLEAN DEFAULT FALSE,
    blurry                  BOOLEAN DEFAULT FALSE,
    brown_hair              BOOLEAN DEFAULT FALSE,
    bushy_eyebrows          BOOLEAN DEFAULT FALSE,
    chubby                  BOOLEAN DEFAULT FALSE,
    double_chin             BOOLEAN DEFAULT FALSE,
    eyeglasses              BOOLEAN DEFAULT FALSE,
    goatee                  BOOLEAN DEFAULT FALSE,
    gray_hair               BOOLEAN DEFAULT FALSE,
    heavy_makeup            BOOLEAN DEFAULT FALSE,
    high_cheekbones         BOOLEAN DEFAULT FALSE,
    male                    BOOLEAN DEFAULT FALSE,
    mouth_slightly_open     BOOLEAN DEFAULT FALSE,
    mustache                BOOLEAN DEFAULT FALSE,
    narrow_eyes             BOOLEAN DEFAULT FALSE,
    no_beard                BOOLEAN DEFAULT FALSE,
    oval_face               BOOLEAN DEFAULT FALSE,
    pale_skin               BOOLEAN DEFAULT FALSE,
    pointy_nose             BOOLEAN DEFAULT FALSE,
    receding_hairline       BOOLEAN DEFAULT FALSE,
    rosy_cheeks             BOOLEAN DEFAULT FALSE,
    sideburns               BOOLEAN DEFAULT FALSE,
    smiling                 BOOLEAN DEFAULT FALSE,
    straight_hair           BOOLEAN DEFAULT FALSE,
    wavy_hair               BOOLEAN DEFAULT FALSE,
    wearing_earrings        BOOLEAN DEFAULT FALSE,
    wearing_hat             BOOLEAN DEFAULT FALSE,
    wearing_lipstick        BOOLEAN DEFAULT FALSE,
    wearing_necklace        BOOLEAN DEFAULT FALSE,
    wearing_necktie         BOOLEAN DEFAULT FALSE,
    young                   BOOLEAN DEFAULT FALSE
);

-- ----------------------------------------------------------------
-- 4. Dim_Cluster – Cluster assignments written back after mining
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Dim_Cluster (
    Cluster_ID    SERIAL PRIMARY KEY,
    Cluster_Label VARCHAR(100),        -- Human-readable label (optional)
    Description   TEXT
);

-- ================================================================
-- FACT TABLE
-- ================================================================

-- ----------------------------------------------------------------
-- 5. Fact_Image_Analysis – Central fact table (Star Schema hub)
--    Feature_Vector dimension (512) matches ResNet18 penultimate layer.
--    Emotion_ID and Attribute_ID are NULLABLE to handle heterogeneous
--    datasets: FER2013 has emotions but no attributes; CelebA vice-versa.
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Fact_Image_Analysis (
    Image_ID        BIGSERIAL   PRIMARY KEY,
    Source_ID       INT         NOT NULL  REFERENCES Dim_Source(Source_ID),
    Emotion_ID      INT                   REFERENCES Dim_Emotion(Emotion_ID),    -- NULL for CelebA
    Attribute_ID    INT                   REFERENCES Dim_Facial_Attributes(Attribute_ID), -- NULL for FER2013
    Cluster_ID      INT                   REFERENCES Dim_Cluster(Cluster_ID),    -- NULL until mining runs
    File_Name       VARCHAR(512) NOT NULL,
    Feature_Vector  VECTOR(512)  NOT NULL,  -- ResNet18 avgpool output = 512-d
    Created_At      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------
-- 6. Indexes for common query patterns
-- ----------------------------------------------------------------
-- Standard B-tree indexes on FK columns used in GROUP BY / JOIN
CREATE INDEX IF NOT EXISTS idx_fact_source     ON Fact_Image_Analysis (Source_ID);
CREATE INDEX IF NOT EXISTS idx_fact_emotion    ON Fact_Image_Analysis (Emotion_ID);
CREATE INDEX IF NOT EXISTS idx_fact_attribute  ON Fact_Image_Analysis (Attribute_ID);
CREATE INDEX IF NOT EXISTS idx_fact_cluster    ON Fact_Image_Analysis (Cluster_ID);

-- IVFFlat ANN index for vector similarity search (cosine distance)
-- NOTE: Build this index AFTER bulk loading data for best performance.
--       LISTS parameter ≈ sqrt(row_count) is a common starting heuristic.
-- CREATE INDEX IF NOT EXISTS idx_fact_vector ON Fact_Image_Analysis
--     USING ivfflat (Feature_Vector vector_cosine_ops) WITH (lists = 100);

-- ================================================================
-- Confirmation notice (visible in Docker logs)
-- ================================================================
DO $$
BEGIN
    RAISE NOTICE 'Data Warehouse schema initialised successfully.';
END$$;
