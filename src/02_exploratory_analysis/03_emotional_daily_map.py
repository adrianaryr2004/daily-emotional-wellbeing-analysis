import pandas as pd
import matplotlib.pyplot as plt
path = "dataset/final_dataset_complete.csv"
df = pd.read_csv(path)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
target = "happy"
features = [
    "stress_level",
    "sleep_hours",
    "sleep_quality",
    "total_apps_used",
    "tiempo_uso_total",
    "prop_active"
] 
cols = ["user_id", "date", target] + features
df_happy = df[df[target].notna()].copy()
df_happy = df_happy[cols].copy()
example_user = 59
df_u = df_happy[df_happy["user_id"] == example_user].sort_values("date").copy()
print("User:", example_user)
print("Days with happy:", df_u.shape[0])
df_u = df_u.dropna(subset=features, how="all").copy()
print("Days after removing days with all features missing:", df_u.shape[0])
#Create within-person label: better_day
user_mean_happy = df_u[target].mean()
df_u["better_day"] = (df_u[target] > user_mean_happy).astype(int)
print("User mean happy:", round(user_mean_happy, 3))
print("Better vs worse days:")
print(df_u["better_day"].value_counts())
# EMOTIONAL MAP - PLOT 1: Time series (happy + main context)
plt.figure()
plt.plot(df_u["date"], df_u["happy"])
plt.title(f"User {example_user} - Happiness over time")
plt.xlabel("Date")
plt.ylabel("Happy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
for v in ["stress_level", "sleep_hours", "prop_active"]:
    if v in df_u.columns:
        plt.figure()
        plt.plot(df_u["date"], df_u[v])
        plt.title(f"User {example_user} - {v} over time")
        plt.xlabel("Date")
        plt.ylabel(v)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
# EMOTIONAL MAP - PLOT 2: Relationship with happy (scatter)
for v in ["stress_level","prop_active"]:
    # only plot if there is data
    tmp = df_u[[target, v]].dropna()
    if tmp.shape[0] >= 5:
        plt.figure()
        plt.scatter(tmp[v], tmp[target])
        plt.title(f"User {example_user} - Happy vs {v}")
        plt.xlabel(v)
        plt.ylabel("Happy")
        plt.tight_layout()
        plt.show()
# EMOTIONAL MAP - PLOT 3: Better vs Worse (mean comparison)
summary = df_u.groupby("better_day")[features].mean()#Separate the variable into two groups and calculate its mean
summary.index = ["Worse_or_equal_than_mean", "Better_than_mean"]
print("\nMean context on worse vs better days (User only):")
print(summary)
diff = summary.loc["Better_than_mean"] - summary.loc["Worse_or_equal_than_mean"] #calculate the difference
print("\nDifference (Better - Worse) (User only):")
print(diff.sort_values(ascending=False))


#Top users with most complete days
# Completely valid days (without missing any relevant variable)
df_complete = df_happy.dropna(subset=features, how="any")
# Count per user
user_counts_complete = df_complete.groupby("user_id").size().sort_values(ascending=False)
print("\nTop users with the most FULL days (all variables):")
print(user_counts_complete.head(10))