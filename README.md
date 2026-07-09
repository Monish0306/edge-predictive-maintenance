<div align="center">

# ⚙️ Edge AI Predictive Maintenance System

### Industry 4.0 · NASA Turbofan · Dual-Head Transformer · RAG Copilot

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.134-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?logo=typescript&logoColor=white)](https://typescriptlang.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Vercel](https://img.shields.io/badge/Vercel-Deployed-000000?logo=vercel&logoColor=white)](https://vercel.com)
[![Render](https://img.shields.io/badge/Render-API-46E3B7?logo=render&logoColor=white)](https://render.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**[🚀 Live Demo](https://edge-predictive-maintenance.vercel.app) · [📡 API Docs](https://edge-ai-fastapi.onrender.com/docs) · [👤 LinkedIn](https://linkedin.com/in/monish-valiveti) · [💻 GitHub](https://github.com/Monish0306)**

> An end-to-end AI-powered web platform that predicts NASA turbofan jet engine failures **up to 45 days in advance** using a custom Dual-Head Transformer model running at **0.20ms inference speed** directly on factory floor devices — zero cloud dependency.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Key Results](#-key-results)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [AI Model Details](#-ai-model-details)
- [RAG Chatbot](#-rag-maintenance-copilot-chatbot)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Local Setup](#-local-setup)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)
- [Business Impact](#-business-impact)
- [Author](#-author)

---

## 🎯 Overview

The **Edge AI Predictive Maintenance System** is a full-stack Industry 4.0 platform built to solve one of manufacturing's biggest problems — **unplanned equipment failures that cost $260,000 per hour**.

Traditional cloud AI solutions suffer from high latency (200–500ms), monthly subscription costs ($2,000+/month), and internet dependency. This system eliminates all three by running a highly optimized Transformer model **directly on factory floor devices** using ONNX Runtime — achieving **0.20ms inference**, **$0/month** running cost, and **zero internet requirement**.

### What it does
- Monitors **15 real-time sensors** from NASA turbofan jet engines
- Predicts failures **12–45 days in advance** using a Dual-Head Transformer
- Classifies severity into **5 levels** (NORMAL → LOW → MEDIUM → HIGH → CRITICAL)
- Predicts **Remaining Useful Life (RUL)** in operational cycles
- Provides **RAG-powered AI chatbot** for maintenance guidance
- Shows **3D Digital Twin** engine visualization
- Monitors **12 global factories** on an interactive world map
- Tracks **OEE metrics** with financial impact analysis

---

## 🌐 Live Demo

| Interface | URL |
|-----------|-----|
| 🖥️ React Dashboard | [edge-predictive-maintenance.vercel.app](https://edge-predictive-maintenance.vercel.app) |
| 📡 FastAPI Backend | [edge-ai-fastapi.onrender.com](https://edge-ai-fastapi.onrender.com) |
| 📚 API Documentation | [edge-ai-fastapi.onrender.com/docs](https://edge-ai-fastapi.onrender.com/docs) |

> **Note:** Backend runs on Render free tier — first load takes ~15 seconds to wake up. A loading screen shows automatically during this time.

---

## 📊 Key Results

| Metric | Value | Benchmark |
|--------|-------|-----------|
| Test Accuracy (FD001) | **98.82%** | Industry avg: ~85% |
| AUC-ROC Score | **0.997** | Near perfect = 1.0 |
| Validation Accuracy | **97.68%** | At epoch 18/25 |
| ONNX Inference Speed | **0.20ms** | Requirement: <50ms |
| Speed vs Requirement | **250× faster** | — |
| Speed vs Cloud AI | **250–2500× faster** | Cloud: 200–500ms |
| False Alarm Rate | **0.7%** | 1 per 143 predictions |
| Failure Catch Rate | **79.2%** | — |
| Model Parameters | **18,690** | Lightweight edge model |
| Model Size (PyTorch) | **145 KB** | — |
| Model Size (ONNX) | **181 KB** | — |
| Cloud Cost Saved | **$24,000/year** | vs $2,000/month cloud |
| Power Reduction | **95%** | 5W edge vs 250W cloud GPU |

---

## 🏗️ System Architecture

```
NASA CMAPSS Dataset (709 engines)
          │
          ▼
┌─────────────────────┐
│  Data Preprocessing │  MinMaxScaler, sliding window (30 cycles × 15 sensors)
│  src/data_processing│  Remove zero-variance sensors (6 removed → 15 kept)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Dual-Head          │  PyTorch Transformer
│  Transformer Model  │  d_model=32, nhead=4, num_layers=2
│  src/model/train.py │  BCEWithLogitsLoss + MSELoss (dual output)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  ONNX Conversion    │  torch.onnx.export → 0.20ms CPU inference
│  convert_to_onnx.py │  250× faster than 50ms industry requirement
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌──────────────────────┐
│  FastAPI Backend    │────▶│  RAG Chatbot System   │
│  start_api.py       │     │  ChromaDB + BM25      │
│  Port: 8000         │     │  Claude claude-sonnet-4-6│
└─────────┬───────────┘     └──────────────────────┘
          │
          ▼
┌─────────────────────┐     ┌──────────────────────┐
│  React Dashboard    │     │  Streamlit Dashboard  │
│  Vite + TypeScript  │     │  9 analytical pages   │
│  Port: 8080         │     │  Port: 8501           │
│  Deployed: Vercel   │     └──────────────────────┘
└─────────────────────┘

MLflow Experiment Tracking (localhost:5000)
ModelMonitor Drift Detection (every 50 predictions)
```

---

## ✨ Features

### 🖥️ React Web Dashboard (13 Pages)

#### Core Monitoring
| Page | Description |
|------|-------------|
| **Live Monitor** | Real-time anomaly probability + health score charts, mode selector (Normal/Warning/Fault), agent recommendations, metric cards |
| **Fleet Overview** | 50 engine cards sorted by risk, color-coded severity badges, real-time ONNX predictions |
| **Digital Twin** | Interactive 3D turbofan engine in Three.js — spinning fan blades, pulsing combustion chamber, components turn red when failing |
| **Plant Map** | Interactive world map with 12 factory locations across 12 countries, 512 total engines, clickable plant detail cards |

#### AI & Analytics
| Page | Description |
|------|-------------|
| **Analytics** | Cross-dataset AUC-ROC comparison (FD001–FD004), confusion matrix, domain shift explanation |
| **Sensor Heatmap** | Explainable AI — 15 sensor attention weights show which sensors triggered the alert |
| **Failure Timeline** | RUL converted to calendar dates, Gantt chart with Safe/Warning/Danger zones, milestone markers |
| **Model Info** | Architecture details, benchmark table, edge vs cloud comparison, parameter counts |
| **Dataset Stats** | NASA CMAPSS dataset breakdown, per-dataset performance, sliding window explanation |

#### Operations
| Page | Description |
|------|-------------|
| **OEE Dashboard** | Live Availability × Performance × Quality gauges, Six Big Losses breakdown, industry benchmark comparison, financial impact |
| **Reports** | Auto-generated plain English maintenance reports, downloadable, cost estimates + recommended actions |
| **Cost Savings** | Edge vs Cloud financial comparison, ROI calculator, per-severity savings breakdown |

#### Alerts & Logs
| Page | Description |
|------|-------------|
| **Notifications** | Alert settings, escalation rules, severity thresholds, notification history with bar chart |
| **Agent Log** | Complete alert history, expandable detail drawers, severity breakdown |

### 🤖 RAG Maintenance Copilot Chatbot
- Floating bot button (bottom-right) visible on all pages
- Hybrid search: ChromaDB semantic + BM25 keyword (Reciprocal Rank Fusion)
- Claude claude-sonnet-4-6 generates accurate, project-scoped answers
- Smart fallback responses without API key
- Domain scope guard — refuses off-topic questions
- Query expansion for vague questions
- Engine context awareness (severity, RUL, anomaly probability)
- Conversation history (last 6 messages)
- 15 knowledge sections + 45 Q&A pairs in knowledge base
- Suggestions panel with 6 quick-question buttons
- Inline bold + code rendering in chat bubbles

### 🔔 Notification System
- Real-time alert bell with unread counter badge
- Toast popups with sound for HIGH and CRITICAL alerts
- 5 severity levels: NORMAL / LOW / MEDIUM / HIGH / CRITICAL
- Escalation rules: Supervisor → Lead → Plant Manager → CEO
- Alert acknowledge and clear functionality
- Full alert history with timestamp

### 📊 Streamlit Dashboard (9 Pages)
| Page | Description |
|------|-------------|
| Live Monitoring | Real-time charts, 1s refresh |
| Model & Edge Stats | PyTorch vs ONNX size/speed proof |
| MLOps & Retraining | Drift detection, one-click retrain |
| Agent Log | Alert history with severity chart |
| Cost & Power Savings | ROI calculator, edge vs cloud |
| Dataset Comparison | All 4 datasets evaluated |
| Sensor Heatmap | Attention weight visualization |
| Maintenance Report | Auto-generated downloadable PDF |
| Failure Timeline | Gantt chart with milestone markers |

---

## 🛠️ Tech Stack

### Machine Learning & AI
| Technology | Purpose |
|-----------|---------|
| **PyTorch 2.0** | Dual-Head Transformer model training |
| **ONNX Runtime** | Edge inference at 0.20ms on CPU |
| **Scikit-learn** | Data preprocessing, evaluation metrics |
| **NumPy / Pandas** | Data processing and transformation |
| **MLflow** | Experiment tracking and model registry |
| **Anthropic Claude API** | RAG chatbot LLM (claude-sonnet-4-6) |

### RAG System
| Technology | Purpose |
|-----------|---------|
| **ChromaDB** | Vector store for semantic search |
| **Sentence-Transformers** | all-MiniLM-L6-v2 text embeddings |
| **LangChain** | RAG pipeline orchestration |
| **Rank-BM25** | Keyword search index |
| **Hybrid Search (RRF)** | Merges semantic + BM25 results |
| **Query Expansion** | Enhances vague queries before retrieval |

### Backend
| Technology | Purpose |
|-----------|---------|
| **Python 3.10** | Core language |
| **FastAPI** | REST API framework |
| **Uvicorn** | ASGI server |
| **Pydantic** | Request/response validation |

### Frontend
| Technology | Purpose |
|-----------|---------|
| **React 18** | UI framework |
| **TypeScript** | Type safety |
| **Vite** | Build tool |
| **Tailwind CSS** | Styling |
| **Framer Motion** | Page transitions and animations |
| **Three.js + R3F** | 3D Digital Twin engine |
| **Recharts** | Real-time charts and graphs |
| **React Leaflet** | Interactive world plant map |
| **TanStack Query** | API state management |
| **React Router DOM** | Client-side routing |
| **Shadcn/UI** | Component library |
| **Lucide React** | Icon library |

### Dashboard & Monitoring
| Technology | Purpose |
|-----------|---------|
| **Streamlit** | Python analytics dashboard |
| **Plotly** | Interactive charts |

### Deployment
| Technology | Purpose |
|-----------|---------|
| **Vercel** | React frontend hosting |
| **Render** | FastAPI backend hosting |
| **Git + GitHub** | Version control |
| **Conda (Anaconda)** | Python environment management |

---

## 🧠 AI Model Details

### Architecture: Dual-Head Transformer

```
Input: (batch, 30 cycles, 15 sensors) = 450 values
         │
         ▼
Linear Projection: 15 → 32 dimensions (d_model=32)
         │
         ▼
Positional Encoding: sine/cosine waves (encodes cycle order)
         │
         ▼
TransformerEncoderLayer #1: nhead=4, dim_feedforward=64
         │
         ▼
TransformerEncoderLayer #2: nhead=4, dim_feedforward=64
         │
         ▼
Global Average Pooling: (batch, 30, 32) → (batch, 32)
         │
    ┌────┴────┐
    ▼         ▼
Head 1:    Head 2:
Anomaly    RUL
Classifier Regressor
(Sigmoid)  (Linear)
    │         │
    ▼         ▼
Probability  Cycles
  (0–1)    (0–125)
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 25 (early stop at 18) |
| Batch Size | 64 |
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau (patience=3) |
| Anomaly Loss | BCEWithLogitsLoss (pos_weight=5.0) |
| RUL Loss | MSELoss |
| Class Imbalance | 83% normal / 17% anomaly → pos_weight fixes |
| Training Time | ~10 minutes on CPU |

### Cross-Dataset Performance
| Dataset | Engines | Conditions | Fault Modes | AUC-ROC | Accuracy |
|---------|---------|------------|-------------|---------|----------|
| **FD001** | 100 | 1 | 1 (HPC) | **0.997** | **98.82%** |
| FD002 | 260 | 6 | 1 (HPC) | 0.541 | — |
| FD003 | 100 | 1 | 2 (HPC+Fan) | 0.793 | — |
| FD004 | 249 | 6 | 2 (HPC+Fan) | 0.554 | — |

> FD002/FD004 lower due to **domain shift** — model trained on 1 operating condition, tested on 6.

### Why Transformer over LSTM/CNN?
| Aspect | Transformer ✅ | LSTM ❌ | CNN ❌ |
|--------|--------------|---------|--------|
| Sees all cycles | Simultaneously | One at a time | Local windows |
| Early data memory | Full attention | Forgets | Limited |
| Explainability | Attention weights | None | None |
| Training speed | 10 min | 30 min | 15 min |
| Dual output | Native | Two models | Two models |

---

## 💬 RAG Maintenance Copilot Chatbot

### Architecture
```
User Question
     │
     ▼
Domain Scope Guard (regex word-boundary patterns)
     │ in-scope?
     ▼
Query Expansion (vague → technical terms)
     │
     ├──▶ ChromaDB Semantic Search (all-MiniLM-L6-v2)
     │
     ├──▶ BM25 Keyword Search
     │
     ▼
Reciprocal Rank Fusion (merge top-5 chunks)
     │
     ▼
Claude claude-sonnet-4-6 (max_tokens=600)
     │
     ▼
Formatted Response (markdown-lite rendering)
```

### Knowledge Base Coverage
| Section | Topics |
|---------|--------|
| Project Overview | System goals, edge AI benefits, who built it |
| Model Architecture | Transformer layers, training config, dual-head design |
| ONNX Deployment | Conversion process, speed benchmarks, hardware support |
| Dataset & Sensors | All 15 sensors with ranges, fault signatures |
| Fault Modes | HPC degradation, fan degradation, root causes |
| Severity & Alerts | 5 levels, response procedures, escalation rules |
| RUL & Scheduling | Maintenance windows, parts lead times, urgency tiers |
| OEE Metrics | Formula, benchmarks, Six Big Losses, financial impact |
| MLOps Pipeline | MLflow tracking, drift detection, retraining steps |
| Dashboard Features | All 13 React pages + 9 Streamlit pages described |
| API Endpoints | All routes, request/response formats |
| Cost Savings | ROI calculations, edge vs cloud, per-severity savings |
| Plant Map | 12 factories, 512 engines, global monitoring |
| Troubleshooting | Common errors + exact fixes |
| Quick Reference | All commands, ports, URLs, folder structure |

---

## 📁 Project Structure

```
D:\PredictiveMaintenance\
│
├── start_api.py                    # FastAPI entry point
├── init_rag.py                     # Build RAG knowledge base (run once)
├── requirements.txt                # Python dependencies
│
├── src\
│   ├── data_processing\
│   │   └── preprocess.py           # Sliding window, normalization, labeling
│   ├── model\
│   │   ├── train.py                # Transformer training with MLflow
│   │   ├── convert_to_onnx.py      # PyTorch → ONNX conversion
│   │   └── evaluate.py             # Cross-dataset evaluation
│   ├── agent\
│   │   ├── maintenance_agent.py    # Severity classification + recommendations
│   │   ├── timeline.py             # RUL → calendar date conversion
│   │   └── report_generator.py     # Auto-generate maintenance reports
│   ├── mlops\
│   │   └── monitor_and_retrain.py  # Drift detection + retraining pipeline
│   └── rag\
│       ├── __init__.py
│       ├── knowledge_base.py       # ChromaDB + BM25 + hybrid search
│       └── query_engine.py         # Claude API + fallback responses
│
├── models\
│   ├── saved\                      # PyTorch .pth checkpoints
│   └── onnx\
│       └── model_fp32.onnx         # Production ONNX model (181 KB)
│
├── data\
│   ├── raw\                        # NASA CMAPSS text files
│   ├── processed\                  # Numpy arrays + metadata JSON
│   ├── rag_db\                     # ChromaDB vector store
│   └── bm25_index\                 # BM25 pickle index
│
├── dashboard\
│   └── app.py                      # Streamlit dashboard (9 pages)
│
├── mlruns\                         # MLflow experiment data
│
└── frontend\                       # React TypeScript application
    ├── src\
    │   ├── App.tsx                  # Root with routes + keep-alive + loader
    │   ├── components\
    │   │   ├── AppLayout.tsx        # Layout with sidebar + chatbot
    │   │   ├── AppSidebar.tsx       # Navigation sidebar
    │   │   ├── ChatbotWidget.tsx    # RAG chatbot floating widget
    │   │   ├── LoadingScreen.tsx    # Backend warm-up loading screen
    │   │   ├── NotificationBell.tsx # Alert bell with unread badge
    │   │   └── AlertToast.tsx       # Toast notification popup
    │   ├── hooks\
    │   │   └── useKeepAlive.ts      # Backend keep-alive ping hook
    │   ├── lib\
    │   │   └── alertStore.ts        # Global alert state management
    │   └── pages\                   # 13 dashboard pages
    └── public\
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.10 (Anaconda recommended)
- Node.js 18+
- Git

### Step 1 — Clone the repository
```bash
git clone https://github.com/Monish0306/edge-predictive-maintenance.git
cd edge-predictive-maintenance
```

### Step 2 — Set up Python environment
```bash
conda create -n predmaint python=3.10 -y
conda activate predmaint
pip install -r requirements.txt
```

### Step 3 — Download NASA dataset
Download from [Kaggle NASA CMAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and place files in `data/raw/`:
```
data/raw/train_FD001.txt
data/raw/test_FD001.txt
data/raw/RUL_FD001.txt
```

### Step 4 — Train the model
```bash
python src/data_processing/preprocess.py    # ~30 seconds
python src/model/train.py                   # ~10 minutes
python src/model/convert_to_onnx.py         # ~10 seconds
python src/model/evaluate.py                # ~1 minute
```

### Step 5 — Build RAG knowledge base (run once)
```bash
python init_rag.py
```
Expected output:
```
✅ Vector store : 175 documents
✅ BM25 index   : 175 documents
✅ Q&A pairs    : 45
✅ Sections     : 15
Knowledge base is ready!
```

### Step 6 — Set environment variable (optional, for Claude AI chatbot)
```bash
# Windows
set ANTHROPIC_API_KEY=your_api_key_here

# Linux/Mac
export ANTHROPIC_API_KEY=your_api_key_here
```
> Without this, the chatbot uses built-in keyword fallback responses.

### Step 7 — Start the backend
```bash
python -m uvicorn start_api:app --reload --port 8000
```
Test: open [http://localhost:8000/health](http://localhost:8000/health)

### Step 8 — Start the frontend
```bash
cd frontend
npm install
npm run dev
```
Open: [http://localhost:8080](http://localhost:8080)

### Optional — Start Streamlit dashboard
```bash
conda activate predmaint
streamlit run dashboard/app.py
```
Open: [http://localhost:8501](http://localhost:8501)

### Optional — View MLflow experiments
```bash
mlflow ui
```
Open: [http://localhost:5000](http://localhost:5000)

---

## 📡 API Reference

Base URL: `https://edge-ai-fastapi.onrender.com`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API status and version |
| `GET` | `/health` | Health check with timestamp |
| `GET` | `/simulate?mode=normal&engine_id=1` | Single engine prediction |
| `GET` | `/fleet?count=20` | Fleet overview sorted by risk |
| `GET` | `/metadata` | Model performance metadata |
| `GET` | `/evaluation` | Cross-dataset evaluation results |
| `POST` | `/chat` | RAG chatbot response |

### Simulate Endpoint
```bash
curl "https://edge-ai-fastapi.onrender.com/simulate?mode=fault&engine_id=47"
```
```json
{
  "engine_id": 47,
  "anomaly_probability": 0.9234,
  "rul_cycles": 8.3,
  "health_score": 7.7,
  "severity": "CRITICAL",
  "root_cause": "HPC Degradation detected",
  "maintenance_schedule": "Emergency maintenance within 24 hours",
  "cost_saved": 350000,
  "recommended_actions": ["Shutdown engine", "Notify Plant Manager", "Expedite parts order"]
}
```

### Chat Endpoint
```bash
curl -X POST "https://edge-ai-fastapi.onrender.com/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What does sensor 4 T30 measure?", "engine_id": 1, "mode": "normal"}'
```

---

## ☁️ Deployment

### Frontend — Vercel
```
Root Directory:  frontend
Framework:       Vite
Build Command:   npm run build
Output Dir:      dist
Environment:     VITE_API_URL=https://edge-ai-fastapi.onrender.com
```

### Backend — Render
```
Build Command:  pip install -r requirements.txt
Start Command:  uvicorn start_api:app --host 0.0.0.0 --port $PORT
Environment:    ANTHROPIC_API_KEY=your_key_here
```

---

## 💰 Business Impact

| Scenario | Cost |
|----------|------|
| LOW alert repair (planned) | $750 |
| LOW alert if ignored (failure) | $5,000–$15,000 |
| MEDIUM alert repair | $10,000–$25,000 |
| MEDIUM alert if ignored | $50,000–$100,000 |
| HIGH alert repair | $50,000–$150,000 |
| HIGH alert if ignored | $200,000–$400,000 |
| CRITICAL repair (emergency) | $150,000–$350,000 |
| CRITICAL if ignored (catastrophic) | $350,000–$500,000 |
| **Best ROI example** | **35,600%** ($980 repair → $350K failure prevented) |
| Cloud AI cost (annual) | $24,000 |
| Edge AI cost (annual) | **$0** |
| Power reduction | **95%** (5W vs 250W GPU) |
| OEE improvement | **8–15 percentage points** |
| Annual savings (mid-size plant) | **$1.5M–$3M** |

---

## 📈 Severity Classification

| Severity | Probability | Health | Action |
|----------|------------|--------|--------|
| 🟢 NORMAL | 0–30% | 70–100% | Continue operations |
| 🟡 LOW | 30–50% | 50–70% | Daily monitoring, inspect in 2 weeks |
| 🟠 MEDIUM | 50–70% | 30–50% | Order parts, repair within 7 days |
| 🔴 HIGH | 70–90% | 10–30% | Reduce load 30%, emergency maintenance 72h |
| 🟣 CRITICAL | 90–100% | 0–10% | **SHUTDOWN IMMEDIATELY** |

---

## 👤 Author

**Monish Valiveti**
B.Tech — Computer and Communication Engineering
Amrita Vishwa Vidyapeetham, Chennai (Graduating 2028)

[![GitHub](https://img.shields.io/badge/GitHub-Monish0306-181717?logo=github)](https://github.com/Monish0306)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-monish--valiveti-0A66C2?logo=linkedin)](https://linkedin.com/in/monish-valiveti)

### Certifications
- Google Cloud Gen AI Academy 2.0
- IBM AI Certification
- Forage GenAI Analytics Simulation
- 1M1B Green Internship (AICTE–Salesforce)

### Experience
- IBM SkillsBuild Internship
- 1M1B Foundation Internship
- ShadowFox Internship

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repo if you found it useful!**

Built with ❤️ by [Monish Valiveti](https://github.com/Monish0306)

</div>