from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
    "on_lithium",
    "on_valproate",
    "on_lamotrigine",
    "on_antipsychotic",
    "on_antidepressant",
    "on_beta_blocker",
    "on_prednisolone",
]

TARGET = "state"


def main():
    df = pd.read_csv(DATA_DIR / "daily_data.csv")
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    df = df.sort_values("date").reset_index(drop=True)

    # Only use rows with an observed mood label.
    df = df.dropna(subset=[TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    # Keep the earliest 80% for training and latest 20% for testing.
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data contains fewer than two mood classes."
        )

    numeric_features = [
        "sleep_hours",
        "sleep_onset_hour",
        "sleep_midpoint_hour",
        "steps",
        "resting_hr",
        "hrv_rmssd",
        "hr_amplitude",
        "activity_peak_hour",
    ]

    categorical_features = [
        "on_lithium",
        "on_valproate",
        "on_lamotrigine",
        "on_antipsychotic",
        "on_antidepressant",
        "on_beta_blocker",
        "on_prednisolone",
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n=== BIPOLARTWIN MOOD CLASSIFIER ===")
    print(f"Training observations: {len(X_train)}")
    print(f"Testing observations: {len(X_test)}")

    print("\nTraining class distribution:")
    print(y_train.value_counts().to_string())

    print("\nTesting class distribution:")
    print(y_test.value_counts().to_string())

    print("\nAccuracy:")
    print(round(accuracy_score(y_test, predictions), 4))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=["stable", "depression", "mania"],
        zero_division=0,
    ))

    print("\nConfusion matrix:")
    print(pd.DataFrame(
        confusion_matrix(
            y_test,
            predictions,
            labels=["stable", "depression", "mania"],
        ),
        index=["Actual stable", "Actual depression", "Actual mania"],
        columns=["Predicted stable", "Predicted depression", "Predicted mania"],
    ))

    # Show estimated class probabilities for the latest observation.
    latest = df.iloc[[-1]]
    probabilities = model.predict_proba(latest[FEATURES])[0]

    print("\nLatest recorded observation:")
    print("Date:", latest["date"].iloc[0].date())

    print("\nEstimated class probabilities:")
    for label, probability in zip(
        model.named_steps["classifier"].classes_,
        probabilities,
    ):
        print(f"{label}: {probability:.3f}")

    print("\nPredicted class:", model.predict(latest[FEATURES])[0])

    print(
        "\nResearch limitation: this dataset contains one patient. "
        "Results do not establish generalization to other patients."
    )


if __name__ == "__main__":
    main()
