# 🎓 Viva Q&A — Spotify Music Intelligence Dashboard

> **PADV Activity 2 | 20 Expected Questions with Model Answers**

---

## Q1. What dataset did you use, and where did you get it?

**Answer:**
I used the **Spotify Tracks Dataset** by maharshipandya from Kaggle, downloaded programmatically using the `kagglehub` library. The raw dataset contains ~114,000 tracks across 114+ genres with 20 columns including 9 audio features extracted by Spotify's API. After cleaning (removing duplicates and nulls), we work with ~89,741 tracks.

---

## Q2. How did you define a "hit" song?

**Answer:**
I defined a hit as any track with a **popularity score greater than 50** out of 100. Spotify's popularity score is computed based on recent stream counts, save rates, and playlist adds. This gave us approximately 34% positive (hit) and 66% negative (not hit) — a moderately imbalanced binary classification problem. The threshold of 50 was chosen as it represents the upper half of the 0–100 scale.

---

## Q3. Why did you choose Random Forest over Logistic Regression or Gradient Boosting?

**Answer:**
Three models were trained and evaluated on a stratified 80/20 train-test split. **Random Forest won on F1-score**, which we prioritized over accuracy because of class imbalance.

- **Logistic Regression** had comparable recall (~0.64) but lower F1
- **Gradient Boosting** collapsed — it achieved high accuracy by predicting the majority class, giving near-zero recall and F1 (~0.02)
- **Random Forest** provided the best balance: catches 64% of actual hits while limiting false positives

Additionally, Random Forest handles non-linear feature interactions natively and provides interpretable feature importance rankings.

---

## Q4. Why did you use F1-score as the primary metric instead of accuracy?

**Answer:**
Because the dataset is **class-imbalanced** (~34% hits). A naive model that always predicts "not hit" would achieve ~66% accuracy — which looks good but catches zero hits. F1-score is the **harmonic mean of precision and recall**, so it penalizes both missing hits (low recall) and false alarms (low precision). F1 forces the model to genuinely learn to identify the minority class, making it the correct metric for this use case.

---

## Q5. Why is your F1-score only ~0.42–0.53? Is the model bad?

**Answer:**
No — it reflects the inherent difficulty of the problem. Audio features alone explain roughly **10–15% of popularity variance**. The remaining 85–90% is driven by:
- Artist brand recognition and follower count
- Spotify editorial playlist placement
- Label marketing spend and promotional timing
- Social media virality (TikTok, Instagram Reels)
- Release timing (weekends, seasons, events)

None of these signals are present in the audio metadata. An AUC of ~0.75 is well above the random baseline of 0.50, meaning the model does learn real patterns — just with fundamental signal limitations.

---

## Q6. What is `class_weight='balanced'` and why did you use it?

**Answer:**
`class_weight='balanced'` is a Scikit-learn parameter that automatically adjusts the **loss penalty** for each class inversely proportional to its frequency. Since hits are ~34% of data, the model would naturally bias toward predicting "not hit." With balanced weights, misclassifying a hit is penalized ~2x more than misclassifying a non-hit. This forces the model to prioritize recall on the minority class. We chose this over SMOTE (synthetic oversampling) due to library compatibility constraints with Python 3.14.

---

## Q7. What does AUC-ROC = 0.75 mean in plain English?

**Answer:**
AUC-ROC (Area Under the Receiver Operating Characteristic curve) measures the probability that the model will **rank a randomly chosen hit higher than a randomly chosen non-hit**. An AUC of 0.75 means: if I pick one random hit and one random non-hit, 75% of the time the model correctly assigns a higher probability to the actual hit. Random guessing gives AUC = 0.50; a perfect model gives 1.0. So 0.75 represents meaningful but imperfect learning.

---

## Q8. What is a confusion matrix, and what does yours show?

**Answer:**
A confusion matrix is a 2×2 table showing the counts of:
- **True Positives (TP):** Predicted hit, actually hit ✅
- **True Negatives (TN):** Predicted not-hit, actually not-hit ✅
- **False Positives (FP):** Predicted hit, actually not-hit ❌ (Type I error)
- **False Negatives (FN):** Predicted not-hit, actually hit ❌ (Type II error)

In our case, recall (~0.64) means we catch 64% of actual hits but miss 36%. The FP count is higher than ideal — meaning we flag some non-hits as potential hits, which is acceptable in an A&R screening context (better to review a false alarm than miss a hit).

---

## Q9. What is feature importance, and what did it reveal?

**Answer:**
Feature importance (from Random Forest) measures how much each feature **reduces impurity** (Gini impurity) across all decision tree splits. Our top features were:
1. **Loudness** — most predictive of popularity
2. **Acousticness** — strong negative predictor
3. **Energy** — positive predictor
4. **Danceability** — moderate positive signal
5. **Instrumentalness** — strong negative (hits have vocals)

This aligns with our correlation analysis: loud, energetic, non-acoustic, vocal tracks are most likely to be hits.

---

## Q10. What is a correlation heatmap (correlogram)?

**Answer:**
A correlation heatmap (or correlogram) displays the **Pearson correlation coefficient** between every pair of variables as a color-coded grid. Values range from −1 (perfect negative correlation, shown in red) to +1 (perfect positive, shown in green). The diagonal is always 1.0 (each variable perfectly correlated with itself). In our dashboard, it reveals which audio features move together and which features correlate most strongly with popularity — the key analysis for feature selection and insight generation.

---

## Q11. What is a violin plot, and why did you use it?

**Answer:**
A violin plot combines a **box plot** (median, IQR, whiskers) with a **kernel density estimate (KDE)** shown as a mirrored area. It shows both the distribution shape and summary statistics simultaneously. We used violin plots in the Data Quality page to compare feature distributions across popularity segments (Low/Medium/High), which reveals not just the median shift but also the shape change — for example, whether hits have a narrower or wider spread in energy values.

---

## Q12. What is a treemap?

**Answer:**
A treemap is a hierarchical visualization where data is displayed as **nested rectangles**. Each rectangle's area is proportional to a quantitative value (e.g., track count), and color can encode a second metric (e.g., hit rate). It allows comparison of many categories simultaneously without the clutter of a bar chart. We referenced treemaps in the context of genre volume comparison — where genre size (track count) and hit rate can be shown together in one compact view.

---

## Q13. Why is Gradient Boosting's accuracy misleadingly high?

**Answer:**
Gradient Boosting achieved ~86% accuracy but near-zero recall (~0.01) and F1 (~0.02). This happens because it found that **always predicting the majority class** ("not hit") gives 66% accuracy, and with its boosting iterations, it converged on a model that barely predicts hits at all. The high accuracy is a statistical artifact of the class imbalance — it looks impressive but is useless for the actual task of identifying hits. This is why we use F1, not accuracy, as our selection criterion.

---

## Q14. What is the difference between precision and recall?

**Answer:**
- **Precision** = TP / (TP + FP) — "Of all tracks I predicted as hits, what fraction actually were hits?" High precision = few false alarms.
- **Recall** = TP / (TP + FN) — "Of all actual hit tracks, what fraction did I correctly identify?" High recall = few missed hits.

For A&R screening, **recall matters more** — missing a hit is more costly than reviewing a false alarm. Our model prioritizes recall (~0.64) at some cost to precision.

---

## Q15. Why did you use a train-test split of 80/20 with stratification?

**Answer:**
**80/20** is a standard split giving enough training data while retaining a meaningful test set (~17,948 samples in our case). **Stratification** (`stratify=y`) ensures both the train and test sets maintain the same class ratio (~34% hits). Without stratification, random sampling might accidentally give the test set 40% hits or 20% hits, making evaluation misleading. Using `random_state=42` ensures reproducibility across runs.

---

## Q16. What is overfitting, and did your model overfit?

**Answer:**
Overfitting occurs when a model memorizes training data patterns but fails to generalize to new data — high training accuracy, low test accuracy. Random Forest is **prone to overfitting** on individual trees but reduces it through **ensemble averaging** (bagging). We didn't explicitly compare train vs. test F1 in this project, but the relatively modest test F1 (~0.42–0.53) suggests the model is not dramatically overfitting — it's simply limited by the signal quality of the available features. Cross-validation would be the next step to verify.

---

## Q17. How did you handle missing values?

**Answer:**
In `prepare_data.py`:
1. Loaded raw data and checked for nulls column by column
2. Dropped rows where `track_name` or `artists` were null (structural nulls)
3. For numeric audio features — Spotify's API rarely returns nulls for these since they're computed; our cleaned dataset showed 0% missing across all 9 audio features
4. Verified completeness in the Data Quality page's "Missing Values" tab

---

## Q18. What is IQR-based outlier detection, and did you clip outliers?

**Answer:**
The **Interquartile Range (IQR)** method defines outliers as values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR. We computed these fences for all numeric features and reported the outlier counts in the Data Quality page. However, for **bounded features** (0–1 scale like danceability, energy), true outliers are rare by definition. For **loudness** (unbounded dB scale), extreme values exist but were retained because Random Forest handles them natively without requiring normalization. Features were scaled only for the ML models using `StandardScaler`.

---

## Q19. Why did you choose Plotly over Matplotlib or Seaborn?

**Answer:**
**Plotly** was chosen for four reasons:
1. **Interactive charts** — users can hover, zoom, filter, and pan directly in the browser
2. **Streamlit native integration** — renders perfectly with `st.plotly_chart()`
3. **Custom theming** — full control over colors, backgrounds, and layout for our dark glassmorphism aesthetic
4. **No static rendering** — Matplotlib produces static PNGs which look out of place in a web dashboard

Seaborn is excellent for statistical analysis notebooks but lacks Plotly's web interactivity.

---

## Q20. What business value does this dashboard provide?

**Answer:**
The dashboard provides value across four stakeholder groups:

| Stakeholder | Specific Value |
|-------------|---------------|
| **Artists** | Before recording, tune audio profile (energy, loudness) against benchmark targets from successful tracks in their genre |
| **A&R Teams** | First-pass automated screening — flag tracks with hit-like audio profiles for human review, reducing manual listening time |
| **Playlist Curators** | Audio fingerprint radar enables finding tracks similar to known hits for discovery playlists |
| **Music Educators** | Understand what the data shows — and doesn't show — about commercial success, illustrating the limits of purely data-driven decision making |

**Key caveat to always mention:** This is a **signal, not a decision**. The model explains ~15% of variance. All recommendations require human judgment, genre knowledge, and contextual understanding that no ML model can replace.

---

*Aaditya A | MCA | PADV Activity 2 | Jain (Deemed-to-be University) | 2026*
