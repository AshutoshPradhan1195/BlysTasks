# Customer behavior analysis — Blys platform

This report summarizes the analysis in `Section1.ipynb` on **5,000** customer records (`customers_5000.csv`), aligned with the technical assessment: preprocessing, NLP-derived signals, segmentation, and actionable recommendations.

---

## 1. Data preprocessing

### 1.1 Missing values


| Area                | Missing | Treatment                                                                                                                                                                                                              |
| ------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Preferred_Service` | 131     | Missing values replaced with an empty string so downstream grouping behaves consistently.                                                                                                                              |
| `Review_Text`       | 305     | Missing values replaced with an empty string; empty reviews yield `compound = 0` via VADER (treated as neutral — not satisfied).                                                                                       |
| `Avg_Spending`      | 200     | Imputed using the **global median** scaled by a mild booking-frequency factor: `median × (1 + Booking_Frequency / 100)` so higher-frequency customers without a recorded spend are not collapsed to a single constant. |


### 1.2 Feature engineering

- `Days_Since_Last_Visit`: Derived from `Last_Activity` versus the analysis date to capture engagement recency.
- **Review NLP:** NLTK **VADER** (`SentimentIntensityAnalyzer`) on `Review_Text`, producing a `compound` score (overall polarity in [-1, 1]). Empty reviews yield `compound = 0` (neutral), which is important when interpreting clusters that include many non-reviewers.

### 1.3 Normalization (clustering inputs)

Numerical columns used for clustering differ in scale (e.g. spending in dollars vs. compound in [-1, 1]). **K-Means is distance-based**, so features were **standardized with `StandardScaler`** (zero mean, unit variance) before fitting.

---

## 2. Customer segmentation (K-Means)

### 2.1 Inputs

Clustering used four features:

1. `Booking_Frequency`
2. `Avg_Spending`
3. `compound` (VADER sentiment)
4. `Days_Since_Last_Visit`

### 2.2 Method

- **Algorithm:** scikit-learn `KMeans`.
- **k = 5:** An **elbow plot** of inertia for **k = 2 … 9** (on standardized features) guided the choice. Five segments produce interpretable, well-separated groups across frequency, spend, sentiment, and recency.
- **Reproducibility:** `random_state=42`, `n_init="auto"`.

### 2.3 Cluster sizes


| Cluster | Customers | Share |
| ------- | --------- | ----- |
| 0       | 1,655     | 33.1% |
| 1       | 460       | 9.2%  |
| 2       | 1,205     | 24.1% |
| 3       | 765       | 15.3% |
| 4       | 915       | 18.3% |


### 2.4 Segment profiles (median / mean of raw-scale attributes)


| Cluster | Booking frequency (med) | Avg spending (med) | Compound (med) | Days since last visit (med) |
| ------- | ----------------------- | ------------------ | -------------- | --------------------------- |
| **0**   | 7.0                     | 164.40             | 0.765          | 98                          |
| **1**   | 24.0                    | 214.75             | 0.000          | 202                         |
| **2**   | 5.0                     | 97.03              | 0.202          | 235                         |
| **3**   | 23.0                    | 266.99             | 0.791          | 76                          |
| **4**   | 5.0                     | 76.22              | -0.520         | 306                         |


### 2.5 Interpretation

```
Cluster 0 — Engaged regulars: Moderate frequency (7 bookings) and spend ($164), strong positive sentiment (compound 0.77), and fairly recent activity (~98 days).
Cluster 1 — High-frequency, disengaged: High booking frequency (24) but neutral sentiment (compound ~0.0) and notably less recent activity. 
These customers book often but sentiment has dropped — a watch group.
Cluster 2 — Low-value inactives: Low frequency (5.7), low spend ($98), mild positive sentiment, and extended inactivity (~237 days). 
Moderate churn risk.
Cluster 3 — Premium champions: Highest spend ($267), high frequency (23), strongest positive sentiment (compound 0.79), and the most recent activity (~76 days). 
The core high-value segment.
Cluster 4 — At-risk / churned: Lowest spend ($82), low frequency (5.8), clearly negative sentiment (compound -0.48), and the longest inactivity (~306 days). 
Highest churn risk.
```

---

## 3. Rule-based overlays

### 3.1 High-value customers

Customers above the **75th percentile** for both `Avg_Spending` and `Booking_Frequency`: **845 customers** total.


| Cluster | High-value count |
| ------- | ---------------- |
| 3       | 609              |
| 1       | 236              |


The strict "high spend + high frequency" definition maps almost entirely to **clusters 3 and 1**, confirming cluster 3 as the premium champion group.

### 3.2 Churn / disengagement risk score

Score = count of: (1) `Days_Since_Last_Visit > 90` and (2) `compound < -0.3`, mapped to **Low / Medium / High** risk.


| Cluster | Low | Medium | High |
| ------- | --- | ------ | ---- |
| 0       | 687 | 965    | 3    |
| 1       | 16  | 342    | 102  |
| 2       | 2   | 1,186  | 17   |
| 3       | 480 | 285    | 0    |
| 4       | 0   | 194    | 721  |


Key observations:

- **Cluster 4** concentrates **721 High-risk customers** (79% of the cluster) — the primary churn target.
- **Cluster 1** has **102 High-risk** despite high booking frequency — declining sentiment combined with growing inactivity makes this a secondary watch group.
- **Cluster 3** has **zero High-risk** customers, confirming it as the healthiest segment.
- **Cluster 2** is largely **Medium-risk** (1,186) — infrequent but not yet critically negative.

---

## 4. Insights and recommendations

### 4.1 Premium champions — Cluster 3 (retention)

**Who:** 765 customers; highest spend and frequency, strongest positive sentiment, most recent activity.

**Retention ideas:**

1. **Loyalty and recognition:** Priority access, bundles, or tiered perks tied to frequency and spend to prevent drift to competitors.
2. **Personalized upsell:** Use `Preferred_Service` to recommend adjacent high-margin services for this high-booking group.
3. **Advocacy programs:** Strong positive sentiment — mine review language for marketing and selective referral programs (with consent).

### 4.2 High-frequency disengaged — Cluster 1 (re-engagement)

**Who:** 460 customers; books frequently but sentiment has gone neutral and recency is fading (202 days median). Contains **102 High-risk** customers.

**Tactics:**

1. **Service quality audit:** Neutral VADER scores on frequent bookers signal dissatisfaction without explicit complaint — proactive NPS outreach can surface root causes.
2. **Win-back offers:** Targeted promotions to pull recency back before the gap becomes permanent.
3. **Preferred service rematch:** Check if `Preferred_Service` has shifted and align recommendations accordingly.

### 4.3 At-risk / churned — Cluster 4 (urgent intervention)

**Who:** 915 customers; lowest spend, negative sentiment, 306-day median inactivity. **721 classified High-risk.**

**Engagement tactics:**

1. **Time-limited win-back offers** — especially for customers whose historical `Avg_Spending` was non-trivial before decline.
2. **Service recovery outreach:** Where `compound < -0.3`, pair re-engagement with an operational follow-up (quality, scheduling, hygiene).
3. **Reduce friction to rebook:** Reminders and one-click flows aligned with past `Preferred_Service`.
4. **Review collection at rebook:** Prompt short post-visit feedback to confirm recovery and update sentiment signals.

### 4.4 Low-value inactives — Cluster 2 (nurture)

**Who:** 1,205 customers; low frequency and spend, largely Medium-risk (1,186 of 1,205).

**Tactics:**

1. **Entry-level reactivation:** Lower-cost service promotions to re-establish the booking habit without deep discounting.
2. **Education-led content:** Service guides or "first time" packages to build confidence and preference.

### 4.5 Engaged regulars — Cluster 0 (grow)

**Who:** 1,655 customers (largest segment); moderate spend and frequency, positive sentiment, ~98 days recency.

**Tactics:**

1. **Frequency incentives:** Loyalty stamps, milestone rewards, or subscription plans to push booking cadence upward.
2. **Upsell pathways:** Cross-sell adjacent services to grow average spend toward Cluster 3 levels.

