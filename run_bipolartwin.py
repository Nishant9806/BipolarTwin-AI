from pathlib import Path

import pandas as pd

from mood_classifier import FEATURES as MOOD_FEATURES
from mood_classifier import TARGET as MOOD_TARGET
from mood_classifier import LogisticRegression
from mood_classifier import Pipeline
from mood_classifier import ColumnTransformer
from mood_classifier import SimpleImputer
from mood_classifier import StandardScaler
from mood_classifier import OneHotEncoder

from mood_forecaster import FEATURES as FORECAST_FEATURES
from mood_forecaster import TARGET as FORECAST_TARGET
from mood_forecaster import make_pipeline

from digital_twin import DigitalTwin


DATA_DIR = Path(__file__).resolve().parent / "dataset"


def train_mood_model():
    df = pd.read_csv(DATA_DIR / "daily_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=[MOOD_TARGET])

    numeric = [
        "sleep_hours", "sleep_onset_hour", "sleep_midpoint_hour",
        "steps", "resting_hr", "hrv_rmssd",
        "hr_amplitude", "activity_peak_hour",
    ]

    categorical = [
        "on_lithium", "on_valproate", "on_lamotrigine",
        "on_antipsychotic", "on_antidepressant",
        "on_beta_blocker", "on_prednisolone",
    ]

    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )),
    ])

    model.fit(df[MOOD_FEATURES], df[MOOD_TARGET])
    return model


def train_forecast_model():
    df = pd.read_csv(DATA_DIR / "daily_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=[FORECAST_TARGET])

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
    )

    model.fit(
        df[FORECAST_FEATURES],
        df[FORECAST_TARGET].astype(int),
    )

    return model


def main():
    print("=" * 60)
    print("BIPOLARTWIN AI — INTEGRATED PROTOTYPE")
    print("=" * 60)

    twin = DigitalTwin("P001")

    mood_model = train_mood_model()
    forecast_model = train_forecast_model()

    # Use the latest recorded observation as the input.
    latest = twin.history.iloc[[-1]]

    print("\n1. DIGITAL TWIN STATE")
    print(twin.update_state())

    print("\n2. CURRENT MOOD ESTIMATE")
    mood_probabilities = mood_model.predict_proba(
        latest[MOOD_FEATURES]
    )[0]

    for label, probability in zip(
        mood_model.named_steps["classifier"].classes_,
        mood_probabilities,
    ):
        print(f"{label}: {probability:.3f}")

    print("Estimated class:", mood_model.predict(
        latest[MOOD_FEATURES]
    )[0])

    print("\n3. FUTURE OUTCOME ESTIMATE")
    probability = forecast_model.predict_proba(
        latest[FORECAST_FEATURES]
    )[0, 1]

    print(
        "Probability of positive relapse_7d label:",
        round(float(probability), 3),
    )

    print("\n4. PERSONAL BASELINE")
    print(twin.calculate_baseline())

    print("\n5. BASELINE DEVIATIONS")
    print(twin.calculate_deviations())

    print("\nPrototype execution completed.")
    print(
        "Research prototype only. Future-label provenance is "
        "unverified, and generalization to other patients "
        "has not been established."
    )


if __name__ == "__main__":
    main()
