import pandas as pd
import numpy as np
from itertools import combinations #to create combinations

csv_path = "dataset/final_dataset_complete.csv"
TARGET = "happy"
FEATURES = [
    "stress_level",
    "tiempo_uso_total",
    "prop_active",
    "prop_stationary",
    "unique_wifi",
    "total_calls",
    "num_sms",
    "calendar_event_count",
    "num_deadlines_today",
    
]

MIN_SUPPORT = 4 #A rule must appear at least 4 times.For example , if num_sms=HIGH only shows 2 days, it's not valid. If it shows 5 days, it can be valid.
MIN_CONFIDENCE = 0.60 #Of all the days the rule appears, in what proportion was the day better?For example ,if num_sms=HIGH appears 5 times and in 4 of those instances the day was better, then: Since 0.80 > 0.60, the rule is accepted.
MAX_RULE_SIZE = 2 #Look for rules with a maximum size of 2.

df = pd.read_csv(csv_path)
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
df = df.dropna(subset=["user_id", "date"])
df["user_id"] = df["user_id"].astype(int)
FEATURES = [c for c in FEATURES if c in df.columns]
df = df[df[TARGET].notna()].copy()
print("\nDataset after preprocessing:")
print("Rows:", len(df))
print("Users:", df["user_id"].nunique())
print("Days with target:", df.shape[0])
#Ensure that all variables are numeric.
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
# better_day = day better than personal average
df["user_mean"] = df.groupby("user_id")[TARGET].transform("mean")
df["better_day"] = (df[TARGET] > df["user_mean"]).astype(int)
# z-score within each user
for f in FEATURES:
    mu = df.groupby("user_id")[f].transform("mean") #The transform function calculates the average within each group and returns it aligned with the original rows.
    sd = df.groupby("user_id")[f].transform("std").replace(0, np.nan) #to avoid division by zero in each user
    df[f] = (df[f] - mu) / sd

print("\nFeatures normalized within user (z-score).")

#Discretize variables
def categorize_z(x):
    if pd.isna(x):
        return np.nan
    elif x < -0.5:
        return "LOW"
    elif x > 0.5:
        return "HIGH"
    else:
        return "NORMAL"
#This creates a new column for each feature
for f in FEATURES:
    df[f + "_cat"] = df[f].apply(categorize_z)

#Function to extract rules from a user
def mine_rules_for_user(user_df, min_support=MIN_SUPPORT, min_conf=MIN_CONFIDENCE, max_rule_size=MAX_RULE_SIZE):
    cat_cols = [c for c in user_df.columns if c.endswith("_cat")] #Look for all columns that end in _cat. These are the categorized variables that will be used for the rules.
    results = [] #I will save the rules found
    if user_df.empty:
        return pd.DataFrame()
    #Rules of 1 variable
    for col in cat_cols: #For each column
        values = user_df[col].dropna().unique() #Unique values : Low , High , Normal
        for val in values:
            subset = user_df[user_df[col] == val] #This filters only the days where that variable takes that value.
            support = len(subset) #Count how many days meet that condition.
            if support < min_support:
                continue #Skip this iteration
            confidence = subset["better_day"].mean()#The average of that column is exactly the proportion of better days.
            if confidence < min_conf:
                continue 
            rule_name = col.replace("_cat", "") + "=" + str(val) #We create the name of the rule
            results.append({
                "rule": rule_name,
                "support": support,
                "confidence": round(confidence, 3)
            })

    # rules of two variables
    if max_rule_size >= 2:
        pairs = list(combinations(cat_cols, 2)) #This creates all possible pairs of variables.No inverted duplicates are created
        for col1, col2 in pairs: #I go through each pair.
            temp = user_df[[col1, col2, "better_day"]].dropna() #Here you remove any row that has NaN in col1, col2 or better_day.
            unique_pairs = temp[[col1, col2]].drop_duplicates() #Find different combinations of values.
            
            for _, row in unique_pairs.iterrows(): #_ is index , we iter the rows of the dataframe. There is index due to the function iterrrows.
                val1 = row[col1]
                val2 = row[col2] #Take the value from the second column in that row.
                subset = temp[(temp[col1] == val1) & (temp[col2] == val2)] #Give me all days where that combination occurs.
                support = len(subset)
                if support < min_support:
                    continue
                confidence = subset["better_day"].mean() #Calculate the mean of better_day within that subset.
                if confidence < min_conf:
                    continue
                rule_name = (
                    col1.replace("_cat", "") + "=" + str(val1)
                    + " AND " +
                    col2.replace("_cat", "") + "=" + str(val2)
                )

                results.append({
                    "rule": rule_name,
                    "support": support,
                    "confidence": round(confidence, 3)
                })

    rules_df = pd.DataFrame(results)

    if not rules_df.empty:
        rules_df = rules_df.sort_values(
            by=["confidence", "support"],
            ascending=False
        )

    return rules_df

# Valid users
valid_users = []
for uid in df["user_id"].unique():
    u_temp = df[df["user_id"] == uid]
    if len(u_temp) >= 8 and u_temp["better_day"].sum() >= 2:
        valid_users.append(uid)

print("Valid users:", valid_users)

#Change here the user you want
user_id = 49
u = df[df["user_id"] == user_id].copy()
print(f"\nSelected user: {user_id}")
print("Days available:", len(u))
print("Better-day rate:", round(u["better_day"].mean(), 3))

if len(u) < 8:
    print("Not enough data for this user.")
    raise SystemExit

rules = mine_rules_for_user(
    u,
    min_support=MIN_SUPPORT,
    min_conf=MIN_CONFIDENCE,
    max_rule_size=MAX_RULE_SIZE
)

if rules.empty:
    print("\nNo strong rules found for this user.")
else:
    print("\nTop within-user rules associated with better days:\n")
    print(rules.head(10).to_string(index=False))