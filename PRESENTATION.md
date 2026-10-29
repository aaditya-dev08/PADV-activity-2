# 🎵 Spotify Music Intelligence — 8-Slide Presentation Outline

> **PADV Activity 2 | MCA | Jain (Deemed-to-be University) | 2026**

---

## Slide 1 — Title Slide

**Title:** What Makes a Hit Song?

**Subtitle:** Spotify Music Intelligence Dashboard — Predictive Analytics on 114k Tracks

**Presented by:** Aaditya A
**Programme:** MCA
**Subject:** Predictive Analytics and Data Visualization (PADV)
**Activity:** 2 — Data Storytelling Through Visualization
**Date:** 2026

_[Show landing page hero screen as background]_

---

## Slide 2 — The Problem

**Headline:** 114,000 songs. Only ~34% become hits. Can we predict which ones?

**Key Points:**
- Spotify hosts 100M+ songs; hits are rare and valuable
- A&R teams manually screen thousands of demos every year
- Audio features (energy, danceability, loudness, etc.) are machine-readable
- **Research question:** Do audio features predict commercial success?

**Why It Matters:**
| Stakeholder | Use Case |
|-------------|----------|
| Artists | Optimize production before release |
| Labels (A&R) | Pre-screen demos at scale |
| Playlist curators | Filter by audio profile |
| Marketers | Identify high-potential tracks early |

_[Visual: Donut chart — 34% Hit vs 66% Not Hit]_

---

## Slide 3 — Dataset & Preparation

**Headline:** 114,000 raw tracks → 89,741 clean, analysis-ready records

**Dataset Details:**
- **Source:** Kaggle — maharshipandya/spotify-tracks-dataset
- **Raw size:** 114,000 tracks × 20 columns
- **After cleaning:** 89,741 tracks
- **Genres:** 114+ unique categories

**Preprocessing Pipeline:**
1. Downloaded via `kagglehub` API
2. Removed duplicates (same track_id)
3. Dropped rows with null `track_name` or `artists`
4. Capped outliers using IQR method for bounded features
5. Encoded `explicit` as 0/1 boolean
6. Created `is_hit` target: 1 if popularity > 50, else 0
7. Created `popularity_segment`: Low / Medium / High

**Target Variable Distribution:**
- `is_hit = 1` (Hit): ~34%
- `is_hit = 0` (Not Hit): ~66%

_[Visual: Popularity histogram with hit threshold line at 50]_

---

## Slide 4 — Exploratory Insights

**Headline:** Acousticness kills hits. Loudness and Energy drive them.

**Top Correlations with Popularity:**

| Feature | Pearson r | Direction |
|---------|-----------|-----------|
| Acousticness | −0.35 | Negative ↓ |
| Loudness | +0.31 | Positive ↑ |
| Instrumentalness | −0.29 | Negative ↓ |
| Energy | +0.27 | Positive ↑ |
| Valence (mood) | +0.05 | Weak |

**Key Findings:**
- High-energy + loud + low-acoustic = formula for popularity
- Instrumentalness near-zero in virtually all hit tracks → **vocals matter**
- Valence (happy vs. sad) barely correlates → mood doesn't predict success
- Genre matters more than any individual audio feature

_[Visual: Correlation heatmap from Feature Analysis page]_

---

## Slide 5 — Model Development

**Headline:** Random Forest wins — F1 balance beats pure accuracy

**Three Models Trained:**

| Model | F1 | Recall | AUC | Notes |
|-------|----|--------|-----|-------|
| Logistic Regression | ~0.48 | ~0.64 | ~0.62 | Good recall, low precision |
| **Random Forest ✅** | **~0.53** | **~0.64** | **~0.75** | **Best overall balance** |
| Gradient Boosting | ~0.02 | ~0.01 | ~0.68 | Collapsed to majority class |

**Why F1 over Accuracy?**
- Dataset is imbalanced (~34% positive)
- Accuracy of 66% achievable by always predicting "not hit"
- F1 balances precision (are predicted hits real?) and recall (are we catching hits?)

**Why Random Forest?**
- Best F1 score across all thresholds
- Catches 64% of actual hits (high recall)
- Handles non-linear relationships natively
- Provides interpretable feature importance

**Imbalance Strategy:** `class_weight='balanced'` — penalizes missing hits more than false positives

_[Visual: Model comparison bar chart from Model Evaluation page]_

---

## Slide 6 — Key Visualizations

**Headline:** From data to insight — 7 pages, 30+ interactive charts

**Featured Visualizations:**

1. **Energy vs Danceability Scatter** — 5,000 track sample colored by popularity
   - Hits cluster in top-right quadrant (high energy + high dance)

2. **Genre Market Map** — Bubble chart: popularity vs energy, size = track count
   - Pop/Hip-Hop dominate upper right; Folk/Classical bottom left

3. **Audio Fingerprint Radar** — Compare up to 6 genres simultaneously
   - Dance music: high energy+dance, low acoustic
   - Classical: inverse pattern

4. **Feature Importance Bar** — From the Random Forest model
   - Loudness, Acousticness, Energy top 3

5. **ROC Curve** — AUC ~0.75, well above random baseline of 0.50

6. **Hit Prediction Gauge** — Live slider → gauge → similar tracks

_[Screenshot carousel of the dashboard pages]_

---

## Slide 7 — Insights & Recommendations

**Headline:** Audio features are a signal, not a crystal ball — but they're actionable

**For Artists:**
- Target loudness between −5 and −8 dB for streaming optimization
- Keep acousticness below 0.3 for mainstream appeal
- High energy (> 0.7) and danceability (> 0.6) correlate with popularity
- Ensure vocal presence — instrumental tracks rarely hit

**For Labels (A&R):**
- Use audio profile as a **first-pass filter**, not a final gate
- Focus genre acquisition on Pop, Hip-Hop, Dance — highest hit rates
- Don't dismiss slow-tempo tracks — valence barely correlates with success

**For Platforms:**
- Audio fingerprint matching enables content-based recommendation
- Genre radar comparison supports playlist curation

**Critical Limitation:**
> Audio features explain only **~10–15% of popularity variance**. Artist fame, playlist placement, label marketing, social virality, and release timing drive the remaining 85–90%.

---

## Slide 8 — Limitations & Future Work

**Headline:** What this model can't see — and where to go next

**Current Limitations:**
| Limitation | Impact |
|-----------|--------|
| No artist popularity signal | High-importance predictor missing |
| No playlist placement data | Major driver of streams untracked |
| No release date / timing | Weekend releases perform differently |
| Static snapshot dataset | No time-series trends |
| AUC ceiling ~0.75 | Cannot improve further without external features |

**Future Work:**
1. **Add external features** — Spotify artist followers, playlist adds, social mentions
2. **Time-series modeling** — Track popularity trajectory post-release
3. **Deep learning** — Use raw audio waveforms (CNN on mel-spectrograms)
4. **Genre-specific models** — Separate model per genre for better precision
5. **Deploy as REST API** — Integrate with Spotify Web API for real-time scoring

**Live Demo:** _[Streamlit Cloud URL]_

**Questions?**

---

*Aaditya A | MCA | PADV Activity 2 | Jain (Deemed-to-be University) | 2026*
