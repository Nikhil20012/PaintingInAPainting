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
Databricks Free Edition
    Bronze → Silver → Gold (PySpark notebooks)
         ↓
Airflow DAG (local orchestration)
    Ingest → Clean → Label → Upload → Generate → Train → Evaluate → Deploy
         ↓
Azure Data Lake Gen2
    painting-data / bronze / silver / gold / synthetic / models
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

1. **Embed** — predicted style, artist, and genre are used to query a Pinecone vector index containing 80K+ art history context chunks (artist bios, style descriptions, period information) embedded with sentence-transformers
2. **Retrieve** — top-k relevant context chunks are pulled from Pinecone
3. **Generate** — LangGraph orchestrates a workflow that passes retrieved context + model predictions to Claude API for narrative generation

This replaces a blind LLM call with a production RAG architecture where every generated narrative is grounded in real art history context.

---

## Data Pipeline

**Source:** WikiArt dataset (81,444 images, 27 styles, 1,119 artists)

Built using a medallion architecture in Databricks Free Edition:

| Layer | Rows | What happens |
|---|---|---|
| Bronze | 80,042 | Raw audit: class imbalance analysis, duplicate detection, dimension profiling |
| Silver | 79,989 | Remove 22 phash duplicates + 44 uncertain artists, clean genres, filter extreme dimensions |
| Gold | 47,780 | Cap large styles at 3,000, create label mappings, stratified 80/10/10 split |

**Synthetic generation:** 50,000 composite triplets created by alpha-blending pairs of Gold paintings with spatially varying transparency (0.10-0.40) and aging noise. Each triplet produces a composite image, ground truth mask, and full label metadata. Pairs are sampled within the same split to prevent data leakage.

---

## Project Structure

```
PaintingInAPainting/
├── configs/
│   └── default.yaml              # All config: Azure, model, training, MLflow, Optuna
├── dags/
│   └── painting_pipeline.py      # Airflow DAG with 9 tasks
├── data/
│   ├── wikiart/                   # 81K raw images (gitignored)
│   └── gold/labels/               # Gold CSVs with label mappings
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
│   └── training/
│       ├── losses.py              # Dice + BCE + CrossEntropy
│       └── trainer.py             # Training loop + MLflow logging
├── scripts/
│   ├── bronze_ingest.py           # PySpark audit of raw data
│   ├── bronze_to_silver.py        # PySpark data cleaning
│   ├── silver_to_gold.py          # PySpark balancing + label mapping
│   ├── generate_dataset.py        # Synthetic data generation entry point
│   ├── index_pinecone.py          # Embed and load art history into Pinecone
│   ├── train.py                   # Training + Optuna HPO
│   └── upload_gold_to_datalake.py # Azure Data Lake upload
├── docker-compose.yml             # Local Airflow setup
└── notebooks/                     # Databricks notebooks (Bronze/Silver/Gold)
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
```

**Start Airflow (requires Docker):**
```bash
docker compose up -d
```
Dashboard at `http://localhost:8081` (admin / admin)

---

## Usage

**Generate synthetic dataset:**
```bash
python -m scripts.generate_dataset
```

**Upload Gold labels to Azure Data Lake:**
```bash
python scripts/upload_gold_to_datalake.py
```

**Index art history into Pinecone:**
```bash
python scripts/index_pinecone.py
```

**Train:**
```bash
python -m scripts.train
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
- [ ] Model training + MLflow tracking + Optuna HPO
- [ ] Evaluation + Grad-CAM visualizations
- [ ] Power BI dashboard
- [ ] Pinecone indexing + sentence-transformer embeddings
- [ ] LangGraph RAG pipeline + Claude integration
- [ ] Flask API
- [ ] Streamlit frontend
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

---