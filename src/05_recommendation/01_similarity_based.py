import pandas as pd
import numpy as np
csv_path= "dataset/final_dataset_complete.csv"
TARGET = "happy" 
K = 10
FEATURES = [
    "sleep_hours", "sleep_quality",
    "stress_level", "social_people",
    "tiempo_uso_total", "total_apps_used",
    "prop_active", "prop_stationary",
    "bt_unique_devices",
    "dark_minutes", "locked_minutes",
    "unique_wifi",
    "total_calls", "num_sms",
    "calendar_event_count", "num_deadlines_today","hours"
]
df = pd.read_csv(csv_path)
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
df = df.dropna(subset=["user_id", "date"])
df["user_id"] = df["user_id"].astype(int)
# Leave only existing columns
FEATURES = [c for c in FEATURES if c in df.columns]
#Let's focus on days with a target (if there is no target we cannot define "good day")
df = df[df[TARGET].notna()].copy()
print("\nDataset after preprocessing:")
print("Rows:", len(df))
print("Users:", df["user_id"].nunique())
print("Days with target:", df.shape[0])
#Force EVERYTHING to numeric to avoid warnings
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
#Define "better_day" (a day better than your average)
df["user_mean"] = df.groupby("user_id")[TARGET].transform("mean")
df["better_day"] = (df[TARGET] > df["user_mean"]).astype(int)
#Z-score per user
for f in FEATURES:
    mu = df.groupby("user_id")[f].transform("mean") #For each row:calculate the mean of that variable for that user
    sd = df.groupby("user_id")[f].transform("std").replace(0, np.nan)#How much does that variable usually vary for that user, for example if they always sleep similarly → small SD
    df[f] = (df[f] - mu) / sd #Change the value of this variable to indicate whether that day is above or below normal FOR THAT USER
#Choose example: user with more data and last day
user_id = int(df["user_id"].value_counts().index[0])
today_date = df[df["user_id"] == user_id]["date"].max()
u = df[df["user_id"] == user_id].copy()
today = u[u["date"] == today_date]
hist = u[u["date"] < today_date] #Focus only on the days before today
print("Historical days available:", len(hist))
if today.empty or hist.empty:
    print("There is not enough data for that user/date.")
    raise SystemExit
today_vec = today[FEATURES].fillna(0).to_numpy()[0] #convert the chosen day into a vector
hist_mat  = hist[FEATURES].fillna(0).to_numpy()#converts the entire history into a numerical matrix

# Distance (smaller = more similar)
dists = np.linalg.norm(hist_mat - today_vec, axis=1) #This needs numbers, not Nans
#For each feature, subtract the previous day from today, then calculate the Euclidean distance of each row of hist from that of today.
#You then obtain a vector with the distances; the smaller the distance, the closer the days are.
hist = hist.copy()
hist["dist"] = dists #Add the calculated distance as a new column.
neighbors = hist.sort_values("dist").head(K)#Now we choose the k most similar days, ordering them first by distance.
good_neighbors = neighbors[neighbors["better_day"] == 1] #Days like today when you felt better than usual
print("Good neighbors found:", len(good_neighbors))
print(f"\nUSER: {user_id}  DATE: {today_date}  {TARGET}={float(today[TARGET])}")
print("\nNeighbors who look most alike:")
print(neighbors[["date", "dist", TARGET, "better_day"]].head(5))
if good_neighbors.empty:
    print("\nThere are no 'good' neighbors among the most similar. Try with K=20.")
else:
    avg_good = good_neighbors[FEATURES].mean()#From those similar good days, I take the average of each feature, for example, of all the sleep hours
    diff = (avg_good - pd.Series(today_vec, index=FEATURES)).dropna()
    #Now you compare that to today and it turns out + → there were more on good days and − → there were fewer on good days
    top = diff.reindex(diff.abs().sort_values(ascending=False).index).head(6)
    #Show me only the variables that most differentiate my similar good days from today.
    print("\nRecommendations (on similar good days it's usually MORE/LESS):")
    for f, d in top.items():
        print(f"- {f}: {d:+.2f}")

