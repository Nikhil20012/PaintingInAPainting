<div align="center">

# 🎨 Painting in a Painting

**Detecting hidden paintings beneath visible layers using deep learning on synthetic data**

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Azure](https://img.shields.io/badge/Azure-Data_Lake_Gen2-0078D4?style=flat-square&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com/)
[![Databricks](https://img.shields.io/badge/Databricks-PySpark-FF3621?style=flat-square&logo=databricks&logoColor=white)](https://www.databricks.com/)
[![Airflow](https://img.shields.io/badge/Airflow-Pipeline-017CEE?style=flat-square&logo=apacheairflow&logoColor=white)](https://airflow.apache.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000?style=flat-square&logo=pinecone&logoColor=white)](https://www.pinecone.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-RAG_Orchestration-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://www.langchain.com/langgraph)
[![Claude](https://img.shields.io/badge/Claude-LLM_Narrative-D97706?style=flat-square&logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Power BI](https://img.shields.io/badge/Power_BI-Dashboard-F2C811?style=flat-square&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

`IN PROGRESS`

</div>

---

## Overview

A deep learning system that detects hidden paintings beneath other paintings using only standard RGB images, no X-ray or specialized equipment required. The model is a Vision Transformer (ViT-B/16) with multi-task classification and spatial detection heads, trained entirely on synthetically generated composite images.

When a hidden layer is detected, the system retrieves relevant art history context from a Pinecone vector database using sentence-transformer embeddings, then passes the model predictions and retrieved context through a LangGraph-orchestrated RAG pipeline to Claude for grounded narrative generation.

**Core research contribution:** synthetic alpha compositing as a scalable training strategy for hidden layer detection, generalizable to medical imaging, satellite remote sensing, document forensics, and industrial inspection.

---

## Architecture

```
ADLS Gen2 (data lake - source of truth)
    raw / bronze / silver / gold / models
         ↓                    ↑
    Local PySpark ETL (read from ADLS, process, write back)
         ↓
Airflow DAG (local orchestration via Docker Compose)
    Upload Raw → Bronze → Silver → Gold
         ↓
Synthetic Dataset Generation (local)
         ↓
Model Training
    ViT-B/16 + Optuna HPO + MLflow tracking
         ↓
RAG Pipeline
    Model predictions → Pinecone retrieval → LangGraph → Claude narrative
         ↓
Deployment
    Flask API → Azure Container Apps
    Streamlit frontend → Streamlit Community Cloud
    CI/CD → GitHub Actions
```

The original design used Azure Databricks with Delta tables for the ETL pipeline. Due to Azure student subscription compute limitations, the ETL logic was migrated into standalone PySpark jobs while preserving the same staged architecture. The Databricks notebooks remain in the repository as the reference implementation.

Airflow orchestrates the ETL pipeline locally. The downstream ML pipeline (synthetic generation, training, evaluation) can be executed independently or integrated into the DAG depending on the execution environment.

---

## Design Decisions

- **Why synthetic data?** Real X-ray ground truth for hidden paintings is scarce (a few hundred known examples worldwide). Alpha compositing provides scalable, labeled supervision without specialized imaging equipment.
- **Why ViT?** Global self-attention captures long-range visual relationships across the full painting surface, and the dual output (CLS token + patch embeddings) naturally supports both classification and spatial localization in a single backbone.
- **Why ADLS + PySpark?** Cloud-backed staged ETL preserves the medallion architecture despite compute constraints. The original design targeted Azure Databricks with Delta tables; the local PySpark implementation maintains the same staged pipeline while keeping ADLS as the source of truth.
- **Why RAG?** Grounding narratives in retrieved art-history context produces specific, factual outputs instead of generic LLM completions. The retrieval step makes the system auditable since you can inspect what context informed each narrative.
- **Why multi-task learning?** Joint supervision across style, artist, genre, hidden detection, and spatial localization forces the encoder to learn richer shared representations than any single task would produce. The auxiliary classification tasks regularize the hidden detection objective.

---

## Model

The model uses a pretrained ViT-B/16 backbone with two output heads:

**Classification head** (from CLS token):
- Style classification (27 classes)
- Artist classification (856 classes)
- Genre classification (27 classes)
- Hidden layer detection (binary)

**Detection head** (from patch embeddings):
- Spatial heatmap (224x224) localizing where hidden content bleeds through

Multi-task loss combines cross-entropy (style, artist, genre), BCE (hidden detection), and Dice + BCE (heatmap), with configurable task weights.

---

## RAG Pipeline

After the model produces predictions, the system generates a grounded narrative about the painting:

1. **Embed** - predicted style, artist, and genre are used to query a Pinecone vector index containing art history context chunks (artist bios, style descriptions, period information) embedded with sentence-transformers
2. **Retrieve** - top-k relevant context chunks are pulled from Pinecone
3. **Generate** - LangGraph orchestrates a workflow that passes retrieved context + model predictions to Claude API for narrative generation

This replaces a blind LLM call with a production RAG architecture where every generated narrative is grounded in real art history context.

---

## Data Pipeline

**Source:** WikiArt dataset (81,444 images, 27 styles, 1,119 artists)

Built using a medallion architecture with Azure Data Lake Gen2 as the storage layer. Each ETL stage reads its input layer from ADLS, performs the transformation locally using PySpark, and persists the next layer back to ADLS:

| Layer | Rows | What happens |
|---|---|---|
| Bronze | 80,042 | Raw audit: class imbalance analysis, duplicate detection, dimension profiling |
| Silver | 79,989 | Remove 22 phash duplicates + 44 uncertain artists, clean genres, filter extreme dimensions |
| Gold | 47,780 | Cap large styles at 3,000, create label mappings, stratified 80/10/10 split |

**Synthetic generation:** 50,000 composite triplets created locally by alpha-blending pairs of Gold paintings with spatially varying transparency (0.60-0.90 top opacity, 10-40% hidden bleed-through) and Gaussian-smoothed spatial noise. Each triplet produces a composite image, ground truth mask, and full label metadata. Pairs are sampled within the same split to prevent data leakage.

---

## Project Structure

```
PaintingInAPainting/
├── app.py                        # Flask API for inference + RAG narrative
├── streamlit_app.py              # Streamlit frontend
├── configs/
│   └── default.yaml              # All config: Azure, model, training, MLflow, Optuna
├── dags/
│   └── painting_pipeline.py      # Airflow orchestration DAG
├── data/
│   ├── wikiart/                   # 81K raw images (gitignored)
│   ├── raw/                       # Local cache (downloaded from ADLS raw)
│   ├── bronze/                    # Local cache (downloaded from ADLS bronze)
│   ├── silver/                    # Local cache (downloaded from ADLS silver)
│   └── gold/labels/               # Local cache (downloaded from ADLS gold)
├── src/
│   ├── data/
│   │   ├── blend.py               # Synthetic alpha compositing
│   │   └── dataset.py             # PyTorch Dataset
│   ├── models/
│   │   ├── encoder.py             # ViT-B/16 backbone
│   │   ├── classifier.py          # Multi-task classification head
│   │   ├── detector.py            # Heatmap detection head
│   │   └── model.py               # Unified model
│   ├── rag/
│   │   ├── embeddings.py          # Sentence-transformer embedding pipeline
│   │   ├── retriever.py           # Pinecone vector search
│   │   └── graph.py               # LangGraph RAG orchestration workflow
│   ├── training/
│   │   ├── losses.py              # Dice + BCE + CrossEntropy
│   │   └── trainer.py             # Training loop + MLflow logging
│   └── utils/
│       └── datalake.py            # Azure Data Lake upload/download utility
├── scripts/
│   ├── upload_raw.py              # Upload raw metadata to ADLS
│   ├── bronze_ingest.py           # ADLS raw → profile → ADLS bronze
│   ├── bronze_to_silver.py        # ADLS bronze → clean → ADLS silver
│   ├── silver_to_gold.py          # ADLS silver → balance/label → ADLS gold
│   ├── generate_dataset.py        # Synthetic data generation entry point
│   ├── evaluate.py                # Test metrics + Grad-CAM visualizations
│   ├── index_pinecone.py          # Embed and load art history into Pinecone
│   ├── train.py                   # Training + Optuna HPO
│   └── upload_checkpoint.py       # Upload model checkpoint to ADLS
├── notebooks/
│   ├── 01_bronze_ingest.ipynb     # Databricks Bronze layer audit
│   ├── 02_bronze_to_silver.ipynb  # Databricks Silver layer cleaning
│   └── 03_silver_to_gold.ipynb    # Databricks Gold layer balancing + labeling
├── docker-compose.yml             # Local Airflow setup
└── requirements.txt               # All dependencies
```

---

## Setup

```bash
git clone https://github.com/Nikhil20012/PaintingInAPainting.git
cd PaintingInAPainting
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
AZURE_STORAGE_ACCOUNT_NAME=<your-account>
AZURE_STORAGE_ACCOUNT_KEY=<your-key>
PINECONE_API_KEY=<your-key>
ANTHROPIC_API_KEY=<your-key>
```

**Start Airflow (requires Docker):**
```bash
docker compose up -d
```
Dashboard at `http://localhost:8081` (admin / admin)

---

## Usage

**Upload raw data to ADLS:**
```bash
PYTHONPATH=. python scripts/upload_raw.py
```

**Run ETL pipeline (or trigger via Airflow):**
```bash
PYTHONPATH=. python scripts/bronze_ingest.py
PYTHONPATH=. python scripts/bronze_to_silver.py
PYTHONPATH=. python scripts/silver_to_gold.py
```

**Generate synthetic dataset:**
```bash
python -m scripts.generate_dataset
```

**Index art history into Pinecone:**
```bash
PYTHONPATH=. python scripts/index_pinecone.py
```

**Train:**
```bash
python -m scripts.train
```

**Evaluate:**
```bash
PYTHONPATH=. python scripts/evaluate.py
```

**Run Flask API:**
```bash
PYTHONPATH=. python app.py
```

**Run Streamlit frontend:**
```bash
streamlit run streamlit_app.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data storage | Azure Data Lake Gen2 |
| Data processing | PySpark (Databricks Free Edition) |
| Orchestration | Apache Airflow (Docker Compose) |
| ML framework | PyTorch + torchvision |
| Model | ViT-B/16 (pretrained, fine-tuned) |
| Experiment tracking | MLflow |
| Hyperparameter search | Optuna + SQLite |
| Explainability | Grad-CAM |
| Vector database | Pinecone |
| Embeddings | Sentence-transformers |
| RAG orchestration | LangGraph |
| LLM | Claude API (Anthropic) |
| API | Flask |
| Frontend | Streamlit |
| Dashboard | Power BI |
| Deployment | Azure Container Apps |
| CI/CD | GitHub Actions |

---

## Roadmap

- [x] Data engineering pipeline (Bronze/Silver/Gold)
- [x] Model architecture (ViT-B/16 multi-task)
- [x] Synthetic data generation pipeline
- [x] Azure Data Lake integration
- [x] Airflow DAG orchestration
- [x] Pinecone indexing + sentence-transformer embeddings
- [x] LangGraph RAG pipeline + Claude integration
- [x] Flask API
- [x] Streamlit frontend
- [ ] Model training + MLflow tracking + Optuna HPO
- [ ] Evaluation + Grad-CAM visualizations
- [ ] Power BI dashboard
- [ ] Docker + Azure Container Apps deployment
- [ ] GitHub Actions CI/CD

---

## Author

**Nikhil Bharadwaj Yellapragada**
<br>
MS Data Analytics Engineering, Northeastern University

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nikhil-bharadwaj-yellapragada-48321a211)
[![Email](https://img.shields.io/badge/-Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:yellapragada.n@northeastern.edu)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.