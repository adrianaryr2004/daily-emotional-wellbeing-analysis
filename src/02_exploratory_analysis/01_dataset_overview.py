import pandas as pd

path = "dataset/final_dataset_complete.csv"
df = pd.read_csv(path)

# Convert it to real date
df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("Shape:", df.shape)
print(df.head(5))

df.info()

dup = df.duplicated().sum()
print("Duplicate rows:", dup)

nulls = df.isna().mean().sort_values(ascending=False) * 100
print(nulls.head(15))

print("Num users:", df["user_id"].nunique())
print("Num days:", df["date"].nunique())

users_present = sorted(df["user_id"].dropna().astype(int).unique())

print("Users present in dataset:")
print(users_present)

all_ids = set(range(60))
present_ids = set(users_present)

missing_users = sorted(all_ids - present_ids)

print("Users missing from final dataset:")
print(missing_users)


