import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


##Analysis of Happiness(EMA)
path = "dataset/final_dataset_complete.csv"
df = pd.read_csv(path)
# Convert it to real date
df["date"] = pd.to_datetime(df["date"], errors="coerce")
# Target variable
target = "happy"
# Keep only days with valid happy EMA
df_happy = df[df[target].notna()].copy()
print("Rows with happy:", df_happy.shape[0])
print("Users with happy:", df_happy["user_id"].nunique())
print("Days with happy:", df_happy["date"].nunique())
# Basic statistics of happy
print(df_happy["happy"].describe())
# Distribution (counts)
print(df_happy["happy"].value_counts().sort_index())
# Global mean baseline
global_mean = df_happy["happy"].mean() #the average of happy using all available observations
# Prediction
df_happy["pred_global_mean"] = global_mean#I assign the same prediction to all rows
# MAE (Mean Absolute Error)
mae_global = (df_happy["happy"] - df_happy["pred_global_mean"]).abs().mean() #For each observation:Calculate the difference between the actual value and the prediction and take the absolute value.
print("Global mean:", round(global_mean, 3))
print("MAE - Global mean baseline:", round(mae_global, 3))

#Linear Regression
features = [
    "sleep_hours",
    "sleep_quality",
    "total_apps_used",
    "tiempo_uso_total",
    "social_people",
    "stress_level"
]
# Keep only target + features
cols = ["user_id", "date", "happy"] + features
df_model = df_happy[cols].dropna() #If a variable is missing on a day, that day cannot be used for the model.
print("Rows for modeling:", df_model.shape[0])
print("Users for modeling:", df_model["user_id"].nunique())

X = df_model[features]
y = df_model["happy"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict
y_pred = model.predict(X_test)
# Evaluate
mae_lr = mean_absolute_error(y_test, y_pred)
print("MAE - Linear Regression:", round(mae_lr, 3))




# Baseline: per-user mean (personalization simple)
user_mean = df_model.groupby("user_id")["happy"].mean() #For each user, you calculate their average happiness on the days when there is complete data.
df_model["pred_user_mean"] = df_model["user_id"].map(user_mean)
mae_user_mean = (df_model["happy"] - df_model["pred_user_mean"]).abs().mean()
print("MAE - User mean baseline:", round(mae_user_mean, 3))


#Within-Person Analysis of Daily Happiness
# Keep only needed columns
df_sub = df_happy[cols].copy()
# Drop rows where ALL features are missing 
df_sub = df_sub.dropna(subset=features, how="all").copy()
print("\nRows after removing days with all features missing:", df_sub.shape[0])
print("Users after filtering:", df_sub["user_id"].nunique())
# Compute each user's mean happy 
user_mean_happy = df_sub.groupby("user_id")[target].mean()
# Map it back to each row
df_sub["happy_user_mean"] = df_sub["user_id"].map(user_mean_happy)
# Label each day as "better" or "worse" than that user's mean 
# better_day = 1 if happy > user mean, else 0
df_sub["better_day"] = (df_sub[target] > df_sub["happy_user_mean"]).astype(int)
# Counts
print("\nBetter vs Worse days (overall):")
print(df_sub["better_day"].value_counts())

#Compare feature averages between better and worse days
summary = df_sub.groupby("better_day")[features].mean() #Divide the two groups into one for worse days and another for better days, and for each group calculate the average of those features.
# Rename rows for readability
summary.index = ["Worse_or_equal_than_mean", "Better_than_mean"]
print("\nMean of features for worse vs better days:")
print(summary)

# Difference (Better - Worse) 
diff = summary.loc["Better_than_mean"] - summary.loc["Worse_or_equal_than_mean"]
print("\nDifference (Better - Worse):")
print(diff.sort_values(ascending=False))
