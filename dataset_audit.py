from pathlib import Path

import pandas as pd

# Locate the dataset folder relative to this script.
DATA_DIR = Path(__file__).resolve().parent / "dataset"

FILES = [
    "patients.csv",
    "daily_data.csv",
    "ehr_meds.csv",
    "ehr_labs.csv",
]


def audit_file(filename):
    file_path = DATA_DIR / filename

    print("\n" + "=" * 70)
    print(f"FILE: {filename}")
    print("=" * 70)

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {df.columns.tolist()}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values.")
    else:
        for column, count in missing.items():
            percentage = count / len(df) * 100 if len(df) else 0
            print(f"{column}: {count} ({percentage:.2f}%)")

    # Identify likely patient identifier columns.
    patient_columns = [
        column for column in df.columns
        if column.lower() in {"patient_id", "patientid", "subject_id"}
    ]

    for column in patient_columns:
        print(
            f"\nUnique patients in {column}: "
            f"{df[column].nunique(dropna=True)}"
        )

    # Inspect the current mood labels.
    if "state" in df.columns:
        print("\nMood-state distribution:")
        print(df["state"].value_counts(dropna=False).to_string())

    # Inspect potential future-outcome labels.
    for column in ["relapse_7d", "next_episode_type"]:
        if column in df.columns:
            print(f"\nDistribution of {column}:")
            print(
                df[column]
                .value_counts(dropna=False)
                .to_string()
            )

    # Inspect date ranges using common date-column names.
    date_columns = [
        column for column in df.columns
        if any(
            word in column.lower()
            for word in ["date", "timestamp", "datetime"]
        )
    ]

    for column in date_columns:
        parsed = pd.to_datetime(df[column], errors="coerce")
        valid = parsed.dropna()

        if not valid.empty:
            print(
                f"\n{column} range: "
                f"{valid.min()} to {valid.max()}"
            )

    print("\nFirst five rows:")
    print(df.head().to_string(index=False))


def main():
    print("BIPOLARTWIN AI - DATASET AUDIT")
    print(f"Dataset directory: {DATA_DIR}")

    for filename in FILES:
        audit_file(filename)

    print("\n" + "=" * 70)
    print("AUDIT FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()