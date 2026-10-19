from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "dataset"

df = pd.read_csv(DATA_DIR / "daily_data.csv")

df["date"] = pd.to_datetime(df["date"], errors="raise")
df = df.sort_values(["patient_id", "date"]).reset_index(drop=True)

# Examine the recorded state during the following seven calendar days.
df["future_episode_within_7d"] = False
df["future_state_available"] = False

for patient_id, group in df.groupby("patient_id", sort=False):
    indices = group.index

    for idx in indices:
        current_date = df.at[idx, "date"]

        future = group[
            (group["date"] > current_date)
            & (group["date"] <= current_date + pd.Timedelta(days=7))
        ]

        if future.empty:
            continue

        # Only treat this as a complete seven-day window if observations
        # exist for every calendar day in the following seven days.
        expected_dates = pd.date_range(
            current_date + pd.Timedelta(days=1),
            current_date + pd.Timedelta(days=7),
            freq="D",
        )

        observed_dates = set(future["date"].dt.normalize())

        if not set(expected_dates).issubset(observed_dates):
            continue

        df.at[idx, "future_state_available"] = True

        df.at[idx, "future_episode_within_7d"] = (
            future["state"].isin(["depression", "mania"]).any()
        )

print("=== FUTURE LABEL VALIDATION CHECK ===")

complete = df[df["future_state_available"]].copy()

print("\nRows with complete seven-day future windows:", len(complete))

print("\nRecorded relapse label counts:")
print(complete["relapse_7d"].value_counts(dropna=False).to_string())

print("\nFuture episode observed within seven days:")
print(complete["future_episode_within_7d"].value_counts().to_string())

valid = complete[complete["relapse_7d"].isin([0, 1])].copy()

print("\nRows with both a complete future window and a known relapse label:", len(valid))

if not valid.empty:
    print("\nRelapse label vs observed future episode:")
    print(
        pd.crosstab(
            valid["relapse_7d"],
            valid["future_episode_within_7d"],
            rownames=["Recorded relapse_7d"],
            colnames=["Episode observed in next 7 days"],
            margins=True,
        )
    )

    agreement = (
        valid["relapse_7d"].astype(int)
        == valid["future_episode_within_7d"].astype(int)
    ).mean()

    print(f"\nLabel agreement with this proxy check: {agreement:.3f}")

print(
    "\nIMPORTANT: This is a consistency check, not proof that the labels "
    "are clinically valid. A future mood-state transition is only a proxy "
    "for relapse and may not match the dataset's original label definition."
)
print("\n=== RELAPSE LABEL VS NEXT EPISODE TYPE ===")

print(
    pd.crosstab(
        df["relapse_7d"],
        df["next_episode_type"],
        dropna=False,
        margins=True,
    )
)

print("\nRows where relapse_7d = 1 but next_episode_type = none:")

print(
    df[
        (df["relapse_7d"] == 1)
        & (df["next_episode_type"] == "none")
    ][
        ["date", "state", "relapse_7d", "next_episode_type"]
    ].to_string(index=False)
)

print("\nRows where relapse_7d = 0 but next_episode_type is an episode:")

print(
    df[
        (df["relapse_7d"] == 0)
        & (df["next_episode_type"].isin(["depression", "mania"]))
    ][
        ["date", "state", "relapse_7d", "next_episode_type"]
    ].head(15).to_string(index=False)
)