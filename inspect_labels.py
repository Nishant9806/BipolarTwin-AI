from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "dataset"

df = pd.read_csv(DATA_DIR / "daily_data.csv")

df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("=" * 60)
print("1. MOOD STATE BY FUTURE RELAPSE LABEL")
print("=" * 60)

print(pd.crosstab(
    df["state"],
    df["relapse_7d"],
    margins=True,
    dropna=False
))

print("\n" + "=" * 60)
print("2. MOOD STATE BY NEXT EPISODE TYPE")
print("=" * 60)

print(pd.crosstab(
    df["state"],
    df["next_episode_type"],
    margins=True,
    dropna=False
))

print("\n" + "=" * 60)
print("3. FUTURE LABELS BY DATE")
print("=" * 60)

print("First 15 rows:")
print(
    df[["date", "state", "relapse_7d", "next_episode_type"]]
    .head(15)
    .to_string(index=False)
)

print("\nLast 15 rows:")
print(
    df[["date", "state", "relapse_7d", "next_episode_type"]]
    .tail(15)
    .to_string(index=False)
)

print("\n" + "=" * 60)
print("4. LABEL TRANSITIONS")
print("=" * 60)

df["previous_state"] = df["state"].shift(1)

print(pd.crosstab(
    df["previous_state"],
    df["state"],
    margins=True
))

print("\n" + "=" * 60)
print("5. FUTURE LABEL CONSISTENCY")
print("=" * 60)

valid = df.dropna(subset=["relapse_7d"])

print("Rows with valid relapse labels:", len(valid))

print("\nRelapse label counts:")
print(valid["relapse_7d"].value_counts().sort_index())

print("\nNext episode type counts:")
print(df["next_episode_type"].value_counts(dropna=False))

print("\n" + "=" * 60)
print("6. LABELS AROUND STATE CHANGES")
print("=" * 60)

state_change = df["state"].ne(df["state"].shift(1))

columns = [
    "date",
    "previous_state",
    "state",
    "relapse_7d",
    "next_episode_type",
]

print(df.loc[state_change, columns].head(30).to_string(index=False))

print("\nLabel inspection completed.")
