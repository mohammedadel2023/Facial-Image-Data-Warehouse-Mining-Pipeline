# Da 370P — Facial Image Data Warehouse & Mining Pipeline

A full **Data Engineering + Machine Learning** project built from scratch to explore two things simultaneously:
1. How a professional **Data Warehouse (Star Schema)** is designed and populated via an ETL pipeline.
2. How **unsupervised machine learning** (K-Means clustering, Apriori association rules) can discover hidden patterns across heterogeneous datasets — with the math implemented by hand, not imported from a library.

---

## Project Concept

The core idea is to bridge two completely different facial image datasets — **FER2013** (emotion labels) and **CelebA** (40 binary facial attributes) — inside a single Star Schema data warehouse. Once unified, the raw pixel data is discarded and replaced by **512-dimensional feature vectors** extracted from a pre-trained ResNet18 CNN. These vectors become the mathematical foundation for all downstream mining.

```
FER2013 (emotions)  ──┐
                       ├─► ETL Pipeline ─► PostgreSQL + pgvector ─► Custom K-Means + Apriori
CelebA (attributes) ──┘
```

### Why two datasets?

| Dataset | Size | Label type | What it contributes |
|---|---|---|---|
| FER2013 | ~35k images | 7 emotion classes | Emotional expression patterns |
| CelebA | ~202k images | 40 binary attributes | Physical attribute co-occurrence |

By keeping both in **one fact table** (with nullable foreign keys for the heterogeneous fields), we can run a single K-Means pass over the entire combined feature space and discover clusters that span emotion and appearance — something neither dataset reveals alone.

---

## Architecture

### Star Schema

```
                    ┌─────────────────┐
                    │   Dim_Source    │  FER2013, CelebA
                    └────────┬────────┘
                             │
┌──────────────┐    ┌────────▼────────────────────┐    ┌──────────────────────────┐
│  Dim_Emotion │◄───│                             │───►│  Dim_Facial_Attributes   │
│  (7 labels)  │    │    Fact_Image_Analysis      │    │  (40 boolean columns)    │
└──────────────┘    │                             │    └──────────────────────────┘
                    │  Feature_Vector VECTOR(512) │
┌──────────────┐    │  Emotion_ID     (nullable)  │
│  Dim_Cluster │◄───│  Attribute_ID   (nullable)  │
│  (K-Means)   │    └─────────────────────────────┘
└──────────────┘
```

- `Emotion_ID` is **NULL** for CelebA rows (no emotion labels).
- `Attribute_ID` is **NULL** for FER2013 rows (no facial attributes).
- `Cluster_ID` is **NULL** until you run your custom K-Means.
- `Feature_Vector VECTOR(512)` — ResNet18 `avgpool` output, stored natively via **pgvector**.

### ETL Flow

```
[Kaggle] ──kagglehub──► [Raw Images]
                              │
                         OpenCV resize 224×224
                         → Grayscale (both datasets)
                         → Duplicate to 3-channel (pseudo-RGB)
                              │
                         ResNet18 (headless, pretrained)
                         → 512-d feature vector per image
                              │
                         Pydantic validation
                              │
                         psycopg2 UPSERT/INSERT
                              │
                    [PostgreSQL + pgvector]
```

> **Why grayscale duplication?** CelebA is colour; FER2013 is greyscale. Converting both to grayscale and then replicating the channel 3× ensures both datasets enter the CNN with identical tensor shapes **and** identical colour information (none) — eliminating colour as a confounding variable in the feature space.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Database | PostgreSQL 16 + pgvector extension |
| Containerisation | Docker / Docker Compose |
| Data ingestion | kagglehub |
| Image processing | OpenCV (`cv2`) |
| Feature extraction | PyTorch + torchvision (ResNet18) |
| GPU acceleration | CUDA 11.8 (RTX 3050 / any Ampere GPU) |
| Data validation | Pydantic v2 |
| DB driver | psycopg2 |
| ML stub | pandas + NumPy |
| Presentation | Streamlit *(planned)* |

---

## Prerequisites

Install these before starting:

- **Python 3.11.5** — [python.org/downloads](https://www.python.org/downloads/)
- **Docker Desktop** — [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
- **Git** — [git-scm.com](https://git-scm.com)
- **NVIDIA GPU drivers** — if you have an NVIDIA GPU (required for CUDA acceleration)
- **Kaggle account** — [kaggle.com](https://www.kaggle.com) (free)

---

## Setup Guide

### Step 1 — Clone the repository

```powershell
git clone <git@github.com:mohammedadel2023/Facial-Image-Data-Warehouse-Mining-Pipeline.git>
cd "Facial-Image-Data-Warehouse-Mining-Pipeline"
```

### Step 2 — Create and activate the virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal prompt.

### Step 3 — Install dependencies

> **Important:** PyTorch must be installed with the CUDA index URL — plain `pip install` gives you the CPU-only build.

```powershell
# Install everything except torch first
pip install -r requirements.txt

# Then reinstall torch with CUDA 11.8
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

If you **don't have an NVIDIA GPU**, skip the second command — the CPU build installed by `requirements.txt` will work, just slower.

### Step 4 — Configure environment variables

```powershell
copy .env.example .env
```

Open `.env` and fill in:

```env
# Your Kaggle API credentials
# Get from: kaggle.com → Account → API → Create New Token
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# PostgreSQL password (choose anything for local dev)
POSTGRES_PASSWORD=your_secure_password
```

### Step 5 — Start the database

```powershell
docker compose up -d
```

Wait ~10 seconds, then verify it's healthy:

```powershell
docker compose ps
```

You should see `dw_postgres` with status `healthy`. The `init.sql` file runs automatically and creates the entire Star Schema.

Verify the schema was created:

```powershell
docker exec -it dw_postgres psql -U dw_user -d datawarehouse -c "\dt"
```

Expected output:
```
              List of relations
 Schema |          Name          | Type  |  Owner
--------+------------------------+-------+---------
 public | dim_cluster            | table | dw_user
 public | dim_emotion            | table | dw_user
 public | dim_facial_attributes  | table | dw_user
 public | dim_source             | table | dw_user
 public | fact_image_analysis    | table | dw_user
```

---

## Running the Pipeline

### Quick test (100 images, no DB write)

Run this first to verify the entire stack works end-to-end without writing anything to the database:

```powershell
python pipeline.py --max-images 100 --dry-run
```

You should see logs for ingestion, annotation parsing, feature extraction, and validation — no errors.

### Full pipeline run

```powershell
# With NVIDIA GPU (recommended)
python pipeline.py --batch-size 128

# Without GPU (CPU only, slower)
python pipeline.py --batch-size 32
```

**What happens:**
1. Downloads FER2013 (~60 MB) and CelebA (~1.4 GB) from Kaggle into `~/.cache/kagglehub/`
2. Parses `list_attr_celeba.csv` (202,599 attribute rows)
3. Loads ResNet18 weights onto GPU/CPU
4. Iterates through all images in batches of `--batch-size`:
   - Resize → Grayscale → 3-channel duplicate (OpenCV)
   - Batch forward pass through headless ResNet18 → 512-d vectors
   - Pydantic validation
   - UPSERT dimension rows + INSERT fact rows (one commit per batch)

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--batch-size N` | `64` | Images per GPU/CPU forward pass. Increase for faster throughput; decrease if out of memory. |
| `--max-images N` | *(all)* | Cap images per dataset. Use `100` for testing. |
| `--dry-run` | off | Run everything except DB insertion. Safe for debugging. |
| `--data-root PATH` | `./data` | Override dataset storage location. |

### Expected throughput

| Hardware | Batch size | Speed |
|---|---|---|
| RTX 3050 Laptop | 128 | ~200–400 images/sec |
| CPU only | 32 | ~5–10 images/sec |

Full run (~237k images) takes **~10–20 min on GPU**, ~6–8 hours on CPU.

---

## Verify Data in the Database

After the pipeline completes:

```powershell
# Count total fact rows
docker exec -it dw_postgres psql -U dw_user -d datawarehouse -c "SELECT COUNT(*) FROM Fact_Image_Analysis;"

# Rows per source
docker exec -it dw_postgres psql -U dw_user -d datawarehouse -c "
  SELECT s.Source_Name, COUNT(*) AS image_count
  FROM Fact_Image_Analysis f
  JOIN Dim_Source s ON f.Source_ID = s.Source_ID
  GROUP BY s.Source_Name;"

# Rows per emotion (FER2013 only)
docker exec -it dw_postgres psql -U dw_user -d datawarehouse -c "
  SELECT e.Emotion_Label, COUNT(*) AS count
  FROM Fact_Image_Analysis f
  JOIN Dim_Emotion e ON f.Emotion_ID = e.Emotion_ID
  GROUP BY e.Emotion_Label
  ORDER BY count DESC;"
```

---

## Data Mining (planned)


### Step 1 — Load the data

```powershell
python data_mining.py
```

This runs `fetch_feature_vectors()` and prints the DataFrame shape. No clustering yet.

### Step 2 — Implement K-Means

Inside `run_clustering()`, the comment block gives you the full mathematical skeleton:

```python
# Step 1 – Initialisation (K-Means++ or random)
# Step 2 – Assignment: Euclidean distance to nearest centroid
# Step 3 – Update: recompute centroids as cluster means
# Step 4 – Convergence check: ||old_centroids - new_centroids|| < ε
```

### Step 3 — Save results

Once `run_clustering()` returns labels, uncomment the two lines at the bottom of `data_mining.py`:

```python
cluster_series = run_clustering(feature_df, k=10)
save_cluster_assignments(
    image_ids      = feature_df["Image_ID"].tolist(),
    cluster_labels = cluster_series.tolist(),
)
```

`save_cluster_assignments()` handles everything: creates `Dim_Cluster` rows and bulk-updates `Fact_Image_Analysis.Cluster_ID`.

---

## File Structure

```
Da 370P/
│
├── docker-compose.yml          # PostgreSQL + pgvector container
├── init.sql                    # Star Schema DDL (auto-runs on first boot)
├── pipeline.py                 # Main ETL orchestrator — run this
├── data_mining.py              # ML stub — implement your algorithms here
├── requirements.txt
├── .env.example                # Copy to .env and fill in secrets
├── .gitignore
│
└── etl/
    ├── ingest.py               # Kaggle dataset downloader (kagglehub)
    ├── preprocess.py           # OpenCV: resize → grayscale → 3-channel
    ├── extract_features.py     # Headless ResNet18 → 512-d vectors
    ├── schemas.py              # Pydantic validation schemas
    └── load.py                 # psycopg2 UPSERT/INSERT functions
```

---

## Troubleshooting

### `AssertionError: Torch not compiled with CUDA enabled`
You have the CPU build. Reinstall:
```powershell
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### `could not connect to server` (psycopg2)
The Docker container isn't running. Run `docker compose up -d` and wait for status `healthy`.

### Container starts but schema is missing
The `init.sql` only runs on first boot. If you started the container before the file existed:
```powershell
docker compose down -v   # ⚠️ deletes all data
docker compose up -d
```

### Out of memory during pipeline
Reduce `--batch-size`:
```powershell
python pipeline.py --batch-size 32
```
