# Chapter 3 — Survey-based personality & well-being profiles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

CSV_PATH = "dataset/final_dataset_complete.csv"
USER_COL = "user_id"
SURVEY_FEATURES = [
    # Big Five
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "neuroticism",
    "openness",

    # Well-being & mental health
    "flourishing_score",
    "loneliness_score",
    "panas_positive",
    "panas_negative",
    "stress_score",
    "phq_score"
]

K_RANGE = range(2, 5)   # 2–4 profiles is more interpretable (number of clusters)
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[USER_COL])
df[USER_COL] = df[USER_COL].astype(int)
# keep only survey columns that actually exist
SURVEY_FEATURES = [c for c in SURVEY_FEATURES if c in df.columns]
if len(SURVEY_FEATURES) < 3:
    raise ValueError(f"Not enough survey features found: {SURVEY_FEATURES}")
#BUILD SURVEY TABLE (1 ROW PER USER)
survey = (
    df[[USER_COL] + SURVEY_FEATURES]
    .groupby(USER_COL)
    .mean()
    .reset_index()
) #It groups all rows that belong to the same user_id and calculates the average in case.
# force numeric
survey[SURVEY_FEATURES] = survey[SURVEY_FEATURES].apply(
    pd.to_numeric, errors="coerce"
)
# drop users with too many missing values
survey = survey.dropna(thresh=int(0.8 * len(SURVEY_FEATURES)))
# fill remaining missing with median
for c in SURVEY_FEATURES:
    survey[c] = survey[c].fillna(survey[c].median()) #If a value is missing, I leave it as something typical to avoid breaking the analysis. And since I already filtered users with too many NaN values, this is rarely applicable. To ensure compatibility with clustering algorithms
print("Users used:", len(survey))
print("Survey features:", SURVEY_FEATURES)
#  NORMALIZATION
scaler = StandardScaler()
X = scaler.fit_transform(survey[SURVEY_FEATURES]) #puts all variables on the same scale
# Choosing the number of clusters with Silhouette
sil_scores = {}
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10) #It tries 10 different initializations and keeps the best one.
    labels = km.fit_predict(X)
    sil_scores[k] = silhouette_score(X, labels)

best_k = max(sil_scores, key=sil_scores.get)
print("Silhouette scores:", sil_scores)
print("Chosen k:", best_k)

plt.figure()
plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker="o")
plt.xlabel("Number of profiles (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette analysis (survey profiles)")
plt.show()
# FINAL CLUSTERING
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
survey["profile"] = kmeans.fit_predict(X) + 1 #Profiles were labelled starting from 1 instead of 0 for easier interpretation
print("\nProfile distribution:")
print(survey["profile"].value_counts().sort_index())
# PROFILE SUMMARY (for interpretation)
profile_summary = (
    survey
    .groupby("profile")[SURVEY_FEATURES]
    .mean()
    .round(2)
)

print("\nProfile summary (mean values):")
print(profile_summary)

# PCA VISUALIZATION
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
plt.figure()
for p in sorted(survey["profile"].unique()):
    mask = survey["profile"] == p #marks which users belong to that profile
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Profile {p}",
        alpha=0.8
    )
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Survey-based personality & well-being profiles")
plt.legend()
plt.show()

profile_summary_z = (
    pd.DataFrame(X, columns=SURVEY_FEATURES) #x is already standardished
    .assign(profile=survey["profile"].values) #separate into profiles
    .groupby("profile")
    .mean()
)
plt.figure(figsize=(10, 4))
plt.imshow(profile_summary_z, aspect="auto", cmap="coolwarm")
plt.colorbar(label="Standardized mean")
plt.xticks(range(len(SURVEY_FEATURES)), SURVEY_FEATURES, rotation=45, ha="right")
plt.yticks(range(len(profile_summary_z.index)), [f"Profile {i}" for i in profile_summary_z.index])
plt.title("Standardized survey profile patterns")
plt.tight_layout()
plt.show()





# Next — Compare profiles with EMA & behavior variables
DATE_COL = "date"  # if you have it, we use it for #days with records
# Merge the profile back to the (daily) dataset
dfp = df.merge(survey[[USER_COL, "profile"]], on=USER_COL, how="inner") #Only the rows whose users: exist in df and also have a profile remain
# If date exists, parse it
if DATE_COL in dfp.columns:
    dfp[DATE_COL] = pd.to_datetime(dfp[DATE_COL], errors="coerce").dt.date
print("\n===Profiles vs other variables ===")
print("Users in merged dataset:", dfp[USER_COL].nunique())
print("Rows:", len(dfp))

# 1.Survey profiles vs emotions (EMA)
EMA_VARS = ["happy", "how", "stress_level","social_people"]
EMA_VARS = [c for c in EMA_VARS if c in dfp.columns]
if len(EMA_VARS) == 0:
    print("\nNo EMA variables found.")
else:
    # numeric
    dfp[EMA_VARS] = dfp[EMA_VARS].apply(pd.to_numeric, errors="coerce")
    print("\n--- 1.1 EMA mean by profile ---")
    ema_mean = dfp.groupby("profile")[EMA_VARS].mean().round(2)
    print(ema_mean)
    # Boxplot for one main variable (happy if exists, else first available)
    target = "happy" if "happy" in EMA_VARS else EMA_VARS[0]
    box = dfp[[USER_COL, target, "profile"]].dropna()
    print("Daily observations used:", len(box))
    print("Users contributing:", box[USER_COL].nunique())
    plt.figure()
    groups = [box[box["profile"] == p][target].values for p in sorted(box["profile"].unique())]#For each profile p: select only rows from that profile, take the target values ​​and convert them into an array
    plt.boxplot(groups, labels=[f"Profile {p}" for p in sorted(box["profile"].unique())])
    plt.ylabel(target)
    plt.title(f"{target} distribution by profile")
    plt.show()

    # Emotional variability: std per user, then average by profile
    print("\n--- 1.2 Emotional variability (std) by profile ---")
    user_std = dfp.groupby([USER_COL, "profile"])[EMA_VARS].std() #For each user, calculate the standard deviation of their EMA variables.
    prof_std = user_std.groupby("profile").mean().round(2)
    print(prof_std)
# 1.2 Survey profiles vs behavior (descriptive)
BEHAVIOR_VARS = [
    "sleep_hours",
    "sleep_quality",
    "tiempo_uso_total",
    "prop_active",
    "total_apps_used"
]
BEHAVIOR_VARS = [c for c in BEHAVIOR_VARS if c in dfp.columns]
if len(BEHAVIOR_VARS) == 0:
    print("\nNo behavior vars found in BEHAVIOR_VARS.")
else:
    dfp[BEHAVIOR_VARS] = dfp[BEHAVIOR_VARS].apply(pd.to_numeric, errors="coerce")
    print("\n--- 1.3 Behavior mean by profile ---")
    beh_mean = dfp.groupby("profile")[BEHAVIOR_VARS].mean().round(2)
    print(beh_mean)
    # Plot one behavior variable (sleep_hours if exists, else first)
    btarget = "sleep_hours" if "sleep_hours" in BEHAVIOR_VARS else BEHAVIOR_VARS[0]
    beh_data = dfp[["profile", btarget]].dropna()
    print(f"\nObservations used for {btarget} plot:", len(beh_data))
    vals = dfp.groupby("profile")[btarget].mean()
    plt.figure()
    plt.bar([f"Profile {p}" for p in vals.index], vals.values)
    plt.ylabel(f"Mean {btarget}")
    plt.title(f"Mean {btarget} by profile")
    plt.show()







