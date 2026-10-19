from pathlib import Path

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).resolve().parent / "dataset"

FEATURES = [
    "sleep_hours",
    "sleep_onset_hour",
    "sleep_midpoint_hour",
    "steps",
    "resting_hr",
    "hrv_rmssd",
    "hr_amplitude",
    "activity_peak_hour",
]

TARGET = "state"


def main():
    df = pd.read_csv(DATA_DIR / "daily_data.csv")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values(["patient_id", "date"]).reset_index(drop=True)

    # Construct a target from the state exactly seven calendar days later.
    # Match by date rather than shifting rows, so missing dates cannot
    # silently change the forecast horizon.
    future = df[["patient_id", "date", "state"]].copy()
    future["date"] = future["date"] - pd.Timedelta(days=7)
    future = future.rename(columns={"state": "future_state_7d"})

    df = df.merge(
        future,
        on=["patient_id", "date"],
        how="left",
        validate="one_to_one",
    )

    # Only observations with known future labels can train or test the model.
    df = df.dropna(subset=["future_state_7d"]).copy()

    # Remove rows with unknown current mood labels for this initial experiment.
    df = df.dropna(subset=["state"]).copy()

    X = df[FEATURES]
    y = df["future_state_7d"]

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "The chronological training and testing periods must each "
            "contain at least two mood classes."
        )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    # Baseline: assume the current recorded mood persists for seven days.
    baseline_predictions = df["state"].iloc[split_index:].to_numpy()

    print("\n=== PERSISTENCE BASELINE ===")
    print(
        "Baseline accuracy:",
        round(accuracy_score(y_test, baseline_predictions), 4),
    )

    print("\nBaseline classification report:")
    print(
        classification_report(
            y_test,
            baseline_predictions,
            labels=["stable", "depression", "mania"],
            target_names=["Stable", "Depression", "Mania"],
            zero_division=0,
        )
    )

    print("\n=== BIPOLARTWIN SEVEN-DAY MOOD FORECAST EXPERIMENT ===")
    print("Target: mood state exactly seven calendar days later")
    print("Training observations:", len(X_train))
    print("Testing observations:", len(X_test))

    print("\nTraining target distribution:")
    print(y_train.value_counts().to_string())

    print("\nTesting target distribution:")
    print(y_test.value_counts().to_string())

    print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["stable", "depression", "mania"],
            target_names=["Stable", "Depression", "Mania"],
            zero_division=0,
        )
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["stable", "depression", "mania"],
        )
    )

    latest = df.iloc[[-1]]
    probabilities = model.predict_proba(latest[FEATURES])[0]

    print("\nLatest eligible observation date:", latest["date"].iloc[0].date())
    print("Forecast horizon: seven calendar days")

    print("\nExperimental future mood probabilities:")
    for mood, probability in zip(model.classes_, probabilities):
        print(f"{mood}: {probability:.4f}")

    print(
        "\nLIMITATIONS: This is a preliminary experiment using one patient's "
        "data. The dataset's provenance and label definitions are unverified. "
        "The results do not establish clinical validity or generalization "
        "to other patients."
    )


if __name__ == "__main__":
    main()