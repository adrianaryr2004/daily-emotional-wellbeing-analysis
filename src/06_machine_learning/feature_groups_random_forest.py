import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance


csv_path = "dataset/final_dataset_complete.csv"
TARGET = "happy"
RANDOM_STATE = 42
df = pd.read_csv(csv_path)

df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
df = df.dropna(subset=["user_id", "date"])
df["user_id"] = df["user_id"].astype(int)

df = df[df[TARGET].notna()].copy() #remain only rows with a value in happy
print("\nDATASET INFO")
print("Rows:", len(df))
print("Users:", df["user_id"].nunique())
#DAYS PER USER 
days_per_user = df.groupby("user_id")["date"].nunique() #Group by user and count how many different days each one has.
print("\nDATASET COVERAGE PER USER")
print("Number of users:", days_per_user.shape[0])
print("Mean days per user:", round(days_per_user.mean(), 2))
print("Median days per user:", round(days_per_user.median(), 2))
print("Minimum days:", int(days_per_user.min()))
print("Maximum days:", int(days_per_user.max()))

# FEATURE GROUPS
academic_features = [
    "calendar_event_count",
    "num_deadlines_today",
    "hours"
]

sleep_features = [
    "sleep_hours",
    "sleep_quality"
]

psychological_features = [
    "stress_level"
]

social_features = [
    "social_people",
    "bt_unique_devices",
    "total_calls",
    "num_sms"
]

phone_features = [
    "tiempo_uso_total",
    "total_apps_used",
    "dark_minutes",
    "locked_minutes"
]

mobility_features = [
    "prop_active",
    "prop_stationary",
    "unique_wifi"
]

academic_features = [c for c in academic_features if c in df.columns]
sleep_features = [c for c in sleep_features if c in df.columns]
psychological_features = [c for c in psychological_features if c in df.columns]
social_features = [c for c in social_features if c in df.columns]
phone_features = [c for c in phone_features if c in df.columns]
mobility_features = [c for c in mobility_features if c in df.columns]
#Combines all variable lists, removes duplicates, maintains order, and returns a final list
all_features = list(dict.fromkeys(
    academic_features +
    sleep_features +
    psychological_features +
    social_features +
    phone_features +
    mobility_features
))

feature_groups = {
    "Academic context": academic_features,
    "Sleep-related factors": sleep_features,
    "Psychological factor": psychological_features,
    "Social interaction": social_features,
    "Phone behaviour": phone_features,
    "Mobility and activity": mobility_features,
    "All features": all_features
}

print("\nFEATURE GROUPS USED")
for group_name, cols in feature_groups.items():
    print(group_name, ":", cols)

#NUMERIC CONVERSION
# Here de df has all the user_id,date,happy and all the features. I want the ones from all_features to be numeric.Then I will create a subset of the df to work more efficiently.
df[all_features] = df[all_features].apply(pd.to_numeric, errors="coerce")
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

# CREATE better_day
df["user_mean_happy"] = df.groupby("user_id")[TARGET].transform("mean")
df["better_day"] = (df[TARGET] > df["user_mean_happy"]).astype(int)
print("\nBETTER DAY DISTRIBUTION")
print(df["better_day"].value_counts())

#WITHIN-USER NORMALIZATION
for f in all_features:
    mu = df.groupby("user_id")[f].transform("mean")
    sd = df.groupby("user_id")[f].transform("std").replace(0, np.nan)
    df[f] = (df[f] - mu) / sd
print("\nFEATURES NORMALIZED WITHIN USER (z-score)")

#RANDOM FOREST FUNCTION
def run_rf_model(group_name, features, plot_importance=False):

    print("\n" + "=" * 70)
    print("GROUP:", group_name)
    print("FEATURES:", features)
    model_df = df[["user_id", "better_day"] + features].copy() #new df
    X = model_df[features]
    y = model_df["better_day"]
    groups = model_df["user_id"] #User to whom each row belongs
#This creates an object whose job is to divide the dataset into train and test, but with one important condition: the users do not mix.
    splitter = GroupShuffleSplit(
        test_size=0.2, #20% of the groups will go to testing
        n_splits=1,
        random_state=RANDOM_STATE #always the same division
    )

    train_idx, test_idx = next(splitter.split(X, y, groups)) #It gives you: which rows go to train and which rows go to test
    #select rows by position.
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    print("\nTRAIN TEST SPLIT")
    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Train users:", groups.iloc[train_idx].nunique())
    print("Test users:", groups.iloc[test_idx].nunique())

    model = Pipeline([ #step chain
        ("imputer", SimpleImputer(strategy="median")), #This means that if a value is missing from a column, it is replaced by the median of that column.
        #It calculates the median of all Train users, and then uses it in the test if there are any missing values.
        ("rf", RandomForestClassifier(
            n_estimators=300, #This means the forest will have 300 trees.
            max_depth=8, #This helps prevent the tree from memorizing too much of the training data.
            min_samples_leaf=5, #Each final leaf of the tree must have at least 5 observations.
            class_weight="balanced", #Pay more attention to the minority class so as not to ignore it.
            random_state=RANDOM_STATE
        ))
    ])
    print("\nMissing values imputed using MEDIAN")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)#Returns a vector with one prediction for each row of X_test
    y_prob = model.predict_proba(X_test)[:, 1] #Returns a vector with the estimated probability that that day will be better_day.
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\nAccuracy:", round(acc, 3)) #Compare y_test with y_pred and calculate the percentage of correct predictions.
    print("ROC AUC:", round(auc, 3)) #It measures the model's ability to separate the two classes using probabilities.

    print("\nConfusion matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report")
    print(classification_report(y_test, y_pred, digits=3))

    X_test_imp = model.named_steps["imputer"].transform(X_test) #Take x_test and only apply the imputor step to it
    #He fills them with the medians calculated in train
    rf = model.named_steps["rf"] #Extract only the already trained Random Forest.
    #Here I am calculating the importance of each variable using the permutation method.
    perm = permutation_importance(
        rf,
        X_test_imp, #Once the random forest is trained, you run the imputed test through it and compare it with the actual predictions to get the roc_auc.
        y_test,
        n_repeats=20, #For each variable, you don't do the mixing just once, but 20 times.
        random_state=RANDOM_STATE,
        scoring="roc_auc" #It measures the importance of the variables by seeing how much the ROC AUC worsens.
    )
    #returns a vector with an importance per variable
    importance_df = pd.DataFrame({
        "feature": features,
        "importance_mean": perm.importances_mean #the average drop in performance when swapping each variable
        #For each variable, its values ​​are rearranged several times (20), the ROC AUC is recalculated in each case, and the decrease is averaged with respect to the original AUC to measure its importance.
    }).sort_values("importance_mean", ascending=False)

    print("\nPermutation importance")
    print(importance_df.to_string(index=False))

    if plot_importance:

        plot_df = importance_df.sort_values("importance_mean")

        plt.figure(figsize=(8, 5))
        plt.barh(plot_df["feature"], plot_df["importance_mean"])
        plt.title("All features")
        plt.xlabel("Permutation importance (ROC AUC decrease)")
        plt.tight_layout()
        plt.show()

    return acc, auc, importance_df

# RUN MODELS
results = []
all_features_importance = None
for name, features in feature_groups.items():
    if len(features) == 0:
        continue
    show_plot = (name == "All features")
    acc, auc, importance_df = run_rf_model(name, features, plot_importance=show_plot)
    if name == "All features":
        all_features_importance = importance_df.copy() #save the importance table of the model that uses all the variables.
    results.append([name, acc, auc])

# FINAL COMPARISON
results_df = pd.DataFrame(results, columns=["group", "accuracy", "roc_auc"])
results_df = results_df.sort_values("roc_auc", ascending=False)

print("\nFINAL COMPARISON")
print(results_df.to_string(index=False))
plot_df = results_df.sort_values("roc_auc", ascending=True)
plt.figure(figsize=(8, 4))
plt.barh(plot_df["group"], plot_df["roc_auc"])
plt.title("Model performance by feature group")
plt.xlabel("ROC AUC")
plt.tight_layout()
plt.show()

# MAIN TAKEAWAYS
print("\n" + "=" * 70)
print("MAIN TAKEAWAYS")
best_group = results_df.iloc[0]["group"] #Row 0 is the best group
best_auc = results_df.iloc[0]["roc_auc"]
print(f"Best-performing group: {best_group} (ROC AUC = {best_auc:.3f})")

if all_features_importance is not None:
    print("\nTop variables in ALL FEATURES model:")
    print(all_features_importance.head(5).to_string(index=False))



