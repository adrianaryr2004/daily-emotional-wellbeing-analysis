import os
import pandas as pd
import matplotlib.pyplot as plt
csv_path = "dataset/final_dataset_complete.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"No csv path found in: {csv_path}")
df = pd.read_csv(csv_path)
df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["happy"] = pd.to_numeric(df["happy"], errors="coerce")

df = df.dropna(subset=["user_id", "date", "happy"])
df["user_id"] = df["user_id"].astype(int)

print("Total users in the CSV:", df["user_id"].nunique())
min_days = 3
user_counts = df.groupby("user_id")["happy"].count()
print("\nValid days per user (happy):")
print(user_counts.describe())
valid_users = user_counts[user_counts >= min_days].index
df_valid = df[df["user_id"].isin(valid_users)].copy()

print(f"\nUsers with ≥ {min_days} valid days:", len(valid_users))
summary = df_valid.groupby("user_id")["happy"].agg(
    mean_happy="mean",
    sd_happy="std",
    n_days="count"
).reset_index()

print("Number of valid users:", len(valid_users))
print("Number of rows in summary:", len(summary))
print(summary)

print("\nSummary by user (first rows):")
print(summary.head())


print("\n--- CHECK ---")
print("Unique users in summary:", summary["user_id"].nunique())
print("Total points to plot:", len(summary))
# check if there are exactly the same (overlapping) points
duplicates = summary.duplicated(subset=["mean_happy", "sd_happy"]).sum()
print("Number of overlapping points (same mean & sd):", duplicates)



# mean vs sd
plt.figure()
plt.scatter(summary["mean_happy"], summary["sd_happy"])
plt.xlabel("Mean happiness (per user)")
plt.ylabel("SD happiness (per user)")
plt.title("Level vs Stability of Happiness (users)")
plt.tight_layout()
plt.show()
# Top 10 more variable users
top = summary.sort_values("sd_happy", ascending=False).head(10)
plt.figure()
plt.bar(top["user_id"].astype(str), top["sd_happy"])
plt.xlabel("User ID")
plt.ylabel("SD happiness")
plt.title("Most emotionally variable users")
plt.tight_layout()
plt.show()
