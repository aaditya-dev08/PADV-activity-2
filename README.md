# 🎵 Spotify Music Intelligence

> Decode what makes a hit song — an interactive ML analytics dashboard

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)]()
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)]()
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=flat&logo=plotly&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-1DB954?style=flat)]()

---

## 📌 Project Overview

An interactive, premium-styled ML dashboard analyzing **114,000+ Spotify tracks** to understand what audio features drive song popularity. Built as **PADV Activity 2 — Data Storytelling Through Visualization**.

The dashboard predicts whether a track is a "hit" (popularity > 50) using 9 audio features, and presents the findings through 7 fully custom-styled pages that feel like a music streaming app — not a generic data tool.


---

## 🎯 Problem Statement

What makes a song a hit?

This dashboard investigates whether audio features (energy, danceability, loudness, etc.) can predict whether a track crosses the popularity threshold of 50, and visualizes the patterns behind music success across 114+ genres and 89,000+ cleaned tracks.

**Why it matters:**
- A&R teams can use audio signals to pre-screen tracks
- Artists can optimize production based on feature benchmarks
- Playlist curators can filter by audio profile
- Marketers can prioritize genres with high hit rates

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle) |
| **Raw Size** | ~114,000 tracks × 20 columns |
| **After Cleaning** | ~89,741 tracks |
| **Genres** | 114+ unique genres |
| **Target Variable** | `is_hit` = 1 if popularity > 50, else 0 |
| **Class Distribution** | ~34% Hit / ~66% Not Hit |
| **Audio Features Used** | 9 (danceability, energy, loudness, valence, tempo, acousticness, instrumentalness, speechiness, liveness) |

---

## 🖥️ Dashboard Modules (7 Pages)

| # | Page | Description |
|---|------|-------------|
| 🏠 | **Home** | Landing page with KPIs, feature dictionary, dataset snapshot |
| 1 | **Executive Overview** | 6 KPI cards, popularity histogram, top genres, hit/not-hit donut, audio radar |
| 2 | **Track Explorer** | Filter, search, download 89k tracks with glassmorphism UI |
| 3 | **Feature Analysis** | Scatter, correlation heatmap, box plots, trend lines (4 pill-tab views) |
| 4 | **Model Evaluation** | 3-model leaderboard, confusion matrix, ROC curve, feature importance |
| 5 | **Hit Prediction** | Slider-driven live ML prediction with gauge, radar, 5 similar tracks |
| 6 | **Genre Intelligence** | Market map, rankings, audio fingerprint radar comparison, genre deep-dive |
| 7 | **Data Quality** | Missing values, outlier detection, violin distributions, column stats |

---

## 🤖 Model Development

### Models Trained
| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | ~0.65 | ~0.38 | ~0.64 | ~0.48 | ~0.62 |
| **Random Forest ✅** | **~0.72** | **~0.45** | **~0.64** | **~0.53** | **~0.75** |
| Gradient Boosting | ~0.86 | ~0.50 | ~0.01 | ~0.02 | ~0.68 |

> ⚠️ _Note: Exact values are computed live from your trained model. Run `python src/train_model.py` and check `outputs/model_comparison.csv` for actual figures._

### Why Random Forest Won
- Best **F1-score balance** between precision and recall
- Catches **~64% of actual hits** (high recall — critical for A&R use case)
- Robust to the skewed class distribution
- Handles non-linear feature interactions natively

### Imbalance Handling
Used `class_weight='balanced'` to penalize misclassification of the minority (hit) class. SMOTE was considered but skipped due to library conflicts with the Python 3.14 environment.

---

## 💡 Key Insights

| # | Insight | Pearson r |
|---|---------|-----------|
| 1 | **Loudness ↔ Popularity** — louder tracks trend better | +0.31 |
| 2 | **Acousticness ↔ Popularity** — acoustic tracks struggle | −0.35 |
| 3 | **Energy ↔ Popularity** — high energy correlates with hits | +0.27 |
| 4 | **Instrumentalness** collapses to near zero in hit tracks | −0.29 |
| 5 | **Genre matters** — Pop, Hip-Hop, Dance genres dominate | — |

> **Critical caveat:** Audio features explain only **~10–15% of popularity variance**. Artist brand, label marketing, playlist placement, and social virality drive the rest.

---

## ⚠️ Limitations

- Audio features alone have limited predictive power (AUC ~0.75 ceiling)
- Dataset is a **snapshot** — no time-series trends or release-date metadata
- No artist popularity, label tier, or social media signals
- Not intended for real A&R decisions — **educational use only**
- Class imbalance (~34% hit rate) makes recall optimization non-trivial

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.10+ |
| **Dashboard** | Streamlit 1.32+ |
| **Visualization** | Plotly 5.18+ |
| **ML** | Scikit-learn 1.3+ |
| **Data** | Pandas, NumPy |
| **Model Persistence** | Joblib |
| **Dataset Download** | KaggleHub |
| **Styling** | Vanilla CSS (800+ lines, glassmorphism) |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10+
- Kaggle API credentials (for data download)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/spotify-intelligence.git
cd spotify-intelligence

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configure Kaggle Credentials

```powershell
# Windows (PowerShell)
$env:KAGGLE_USERNAME = "your_kaggle_username"
$env:KAGGLE_KEY      = "your_kaggle_api_key"
```

```bash
# macOS/Linux
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"
```

### Prepare Data & Train Models (one-time setup)

```bash
python src/prepare_data.py   # Downloads dataset, cleans, saves to data/
python src/train_model.py    # Trains 3 models, saves best to models/
```

### Launch Dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
spotify-intelligence/
├── app.py                          # Landing page (Home)
├── requirements.txt                # Python dependencies
├── Procfile                        # Heroku/Render deployment
├── .gitignore
├── README.md
├── PRESENTATION.md                 # 8-slide presentation outline
├── VIVA_QA.md                      # 20 Q&A pairs for viva
│
├── .streamlit/
│   └── config.toml                 # Dark theme + server config
│
├── assets/
│   └── style.css                   # 800+ line custom CSS (glassmorphism)
│
├── src/
│   ├── prepare_data.py             # Data pipeline (download → clean → save)
│   ├── train_model.py              # ML training (3 models → outputs)
│   ├── utils.py                    # Shared helpers (load, theme, KPI HTML)
│   └── ui.py                       # top_nav() and page_footer() helpers
│
├── pages/
│   ├── 1_Overview.py               # Executive Overview
│   ├── 2_Track_Explorer.py         # Track search & filter
│   ├── 3_Feature_Analysis.py       # Audio feature deep-dive
│   ├── 4_Model_Evaluation.py       # ML model comparison
│   ├── 5_Hit_Prediction.py         # Live hit predictor
│   ├── 6_Genre_Intelligence.py     # Genre market intelligence
│   └── 7_Data_Quality.py           # Dataset health report
│
├── data/
│   └── spotify_cleaned.csv         # Generated by prepare_data.py
│
├── models/
│   ├── best_model.pkl              # Best trained classifier
│   └── scaler.pkl                  # StandardScaler fitted on train set
│
├── outputs/
│   ├── model_comparison.csv        # Metrics for all 3 models
│   └── feature_importance.csv      # Feature importance from best model
│
└── screenshots/
    ├── overview.png                # (Take manually after deployment)
    ├── model-eval.png
    └── prediction.png
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push to GitHub (commit `data/`, `models/`, `outputs/` if under 100 MB)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point
4. Add Kaggle secrets in **Settings → Secrets**:
   ```toml
   KAGGLE_USERNAME = "your_username"
   KAGGLE_KEY = "your_api_key"
   ```
5. Click **Deploy**

> **Alternative:** Deploy on Render using the `Procfile`. Set environment variables in the Render dashboard.


---

## 👤 Author

**Aaditya A**
- 🎓 MCA — Jain (Deemed-to-be University)
- 📚 Subject: Predictive Analytics and Data Visualization (PADV)
- 📋 Activity 2 — Data Storytelling Through Visualization
- 📅 2026

---

## 📄 License

This project is licensed under the **MIT License** — free to use for educational purposes.

---

