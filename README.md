# Exploring daily emotional well-being through personalized data analysis

Final Degree Project (Proyecto fin de grado) — Grado en Ciencia de Datos e Inteligencia Artificial, University of Deusto · San Sebastián, May 2026  
Author: **Adriana Rodríguez Rodríguez** · GitHub: [@adrianaryr2004](https://github.com/adrianaryr2004)  
Supervisors: Alex Barco (University of Deusto) and Dr Aleksandar Matic (Koa Health)

This project analyses **daily emotional well-being from a personalized perspective**, using records of emotional states and contextual information derived from daily life. It is based on the **StudentLife** dataset (Dartmouth College), which combines smartphone sensing data, ecological momentary assessments (EMA) and psychological surveys.

The starting idea: well-being is not a fixed value. It varies from day to day, and it can be better understood in relation to each person's individual context and behaviour.

📄 Full thesis: [`docs/thesis.pdf`](docs/thesis.pdf)

---

## Key findings

| Analysis | Result |
|---|---|
| Global mean baseline (same prediction for everyone: the overall mean happiness) | MAE = 0.71 |
| Linear regression (sleep, stress, social interaction and phone usage features) | Error similar to the global mean baseline (MAE ≈ 0.70) |
| **Per-user mean baseline** (each person's own average happiness) | **MAE = 0.53** |
| Survey-based personality and well-being profiles (K-Means) | 2 profiles, highest silhouette score (0.19) |
| Random Forest — distinguishing better-than-usual days, best feature group | Academic context, ROC-AUC = 0.648 |

**Takeaway:** emotional well-being is very personal and changes from day to day. Between-person differences explain a large proportion of the variance in happiness scores, so general models had limited value, while personalized approaches captured these differences much better. No single contextual variable dominates: many features contribute small amounts of information. The project has an exploratory scope, focused on understanding patterns rather than building a fully predictive system.

---

## Dataset

- **Source:** StudentLife study, Dartmouth College, Spring term 2013 (60 students initially involved)
- **Final analytical dataset:** 4,753 daily user-level observations · 49 participants · 184 distinct days · 77 columns (75 analytical variables + 2 identifiers)
- **Outcome variable:** daily happiness, from EMA mood reports
- **Data sources integrated:**
  - *Behavioural logs:* app usage, calendar, call log, dining
  - *Academic data:* enrolment and workload, grades, deadlines, online platform engagement (Piazza)
  - *EMA:* activity, class, behaviour, comment, dimensions, dining halls, events, mood, sleep, social, stress, study spaces
  - *Passive smartphone sensing:* activity inference, audio, Bluetooth, dark screen, phone charging, phone lock, Wi-Fi, SMS
  - *Surveys:* Big Five (BFI-44), Flourishing Scale, UCLA Loneliness Scale, PANAS, PSS-10, PHQ-9

The data is **not included** in this repository. See [`dataset/README.md`](dataset/README.md) for where to download it and where to place it.

---

## Repository structure

```
.
├── src/
│   ├── 01_data_cleaning/
│   │   └── build_daily_dataset.py          # Raw StudentLife files → one row per user-day (final_dataset_complete.csv)
│   ├── 02_exploratory_analysis/
│   │   ├── 01_dataset_overview.py          # Shape, duplicates, missing values, users and days
│   │   ├── 02_happiness_baselines.py       # Global and per-user baselines, linear regression, better vs worse days
│   │   └── 03_emotional_daily_map.py       # Emotional daily map of one user: happiness and context over time
│   ├── 03_emotional_variability/
│   │   └── emotional_stability_vs_level.py # Mean vs standard deviation of happiness per user
│   ├── 04_survey_profiles/
│   │   └── survey_profiles_clustering.py   # K-Means profiles, silhouette, PCA, comparison with EMA and behaviour
│   ├── 05_recommendation/
│   │   ├── 01_similarity_based.py          # Most similar past days (K = 10) → personalized recommendations
│   │   └── 02_rule_based.py                # Within-user rules (support / confidence)
│   └── 06_machine_learning/
│       └── feature_groups_random_forest.py # Random Forest by feature group, user-level split, permutation importance
├── dataset/                                # Place the StudentLife data here (not tracked by git)
├── docs/
│   └── thesis.pdf
├── requirements.txt
└── LICENSE
```

---

## Methods

1. **Data preparation and cleaning** — timestamps converted to the America/New_York time zone, daily aggregation per user, integration of all data sources into one dataset, and missing values filled with 0 only for counts and durations (left as missing when a variable was not measured or not answered).
2. **Exploratory emotional analysis** — distribution of daily happiness, baseline models, and within-person classification of each day as better than usual (happiness above the user's own average) or not.
3. **Individual emotional variability** — each user's mean happiness (level) vs its standard deviation (stability).
4. **Survey-based profiles** — survey scales standardized, K-Means with the number of profiles chosen by silhouette score, PCA visualization, and comparison of profiles with EMA and behavioural variables.
5. **Personalized recommendation**
   - *Similarity-based:* features normalized within each user (z-score), Euclidean distance between days, and comparison of today with similar past days that were better than usual.
   - *Rule-based:* z-scores discretized into LOW (< −0.5), NORMAL and HIGH (> 0.5), and rules of one or two conditions filtered by minimum support (4) and confidence (0.60), e.g. `stress_level=LOW → better day` (user 49, confidence 0.75).
6. **Machine learning** — Random Forest trained on each feature group (academic context, sleep, psychological factor, social interaction, phone behaviour, mobility and activity) and on all features together, with a user-level split (`GroupShuffleSplit`) so no participant appears in both training and test sets, evaluated with ROC-AUC and permutation importance.

---

## How to run

From the root folder of the repository:

```bash
pip install -r requirements.txt
```

All scripts use paths relative to the **repository root** (`dataset/...`), so run them from there:

```bash
# 1. Build the daily dataset from the raw StudentLife files
python src/01_data_cleaning/build_daily_dataset.py

# 2. Run any analysis
python src/02_exploratory_analysis/02_happiness_baselines.py
python src/06_machine_learning/feature_groups_random_forest.py
```

> Note: some scripts rely on behaviour that changed in pandas 3.0 and matplotlib 3.10, so `requirements.txt` pins versions below those.

---

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib

**Keywords:** personalized well-being analysis · daily contextual data · emotional data analysis · multimodal data

---

## Acknowledgements

Data from: R. Wang, F. Chen, Z. Chen, T. Li, G. Harari, S. Tignor, X. Zhou, D. Ben-Zeev and A. T. Campbell, "StudentLife: Assessing mental health, academic performance and behavioral trends of college students using smartphones," in *Proceedings of the 2014 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp '14)*, 2014.
