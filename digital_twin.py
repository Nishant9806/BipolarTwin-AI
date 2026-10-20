from pathlib import Path

import pandas as pd


class DigitalTwin:
    """Maintain an evolving representation of one patient's health state."""

    def __init__(self, patient_id):
        self.data_dir = Path(__file__).resolve().parent / "dataset"
        self.patient_id = patient_id

        # Load historical clinical information.
        patients = pd.read_csv(self.data_dir / "patients.csv")
        medications = pd.read_csv(self.data_dir / "ehr_meds.csv")
        labs = pd.read_csv(self.data_dir / "ehr_labs.csv")
        daily_data = pd.read_csv(self.data_dir / "daily_data.csv")

        # Validate the requested patient.
        patient = patients.loc[
            patients["patient_id"] == patient_id
        ]

        if patient.empty:
            raise ValueError(f"Patient {patient_id} was not found.")

        self.profile = patient.iloc[0].to_dict()

        # Retain this patient's longitudinal observations.
        daily_data["date"] = pd.to_datetime(
            daily_data["date"], errors="coerce"
        )

        self.history = (
            daily_data.loc[
                daily_data["patient_id"] == patient_id
            ]
            .sort_values("date")
            .reset_index(drop=True)
        )

        self.medications = medications.loc[
            medications["patient_id"] == patient_id
        ].copy()

        self.labs = labs.loc[
            labs["patient_id"] == patient_id
        ].copy()

        if self.history.empty:
            raise ValueError(
                f"No daily observations found for {patient_id}."
            )

        self.state = {}
        self.update_state()
        
    def ingest_observation(self, observation):
        """Add a new wearable observation and refresh the digital state."""

        required_fields = [
            "date",
            "sleep_hours",
            "steps",
            "resting_hr",
            "hrv_rmssd",
        ]

        missing_fields = [
            field for field in required_fields
            if field not in observation
        ]

        if missing_fields:
            raise ValueError(
                f"Missing required fields: {missing_fields}"
            )

        new_row = observation.copy()

        new_row["patient_id"] = self.patient_id
        new_row["date"] = pd.to_datetime(
            new_row["date"], errors="raise"
        )

        # Do not allow duplicate or out-of-order observations.
        if new_row["date"] <= self.history["date"].max():
            raise ValueError(
                "New observation must be later than the latest record."
            )

        # Preserve the existing schema; unknown fields remain missing.
        new_row = pd.DataFrame([new_row]).reindex(
            columns=self.history.columns
        )

        self.history = pd.concat(
            [self.history, new_row],
            ignore_index=True
        )

        self.history = self.history.sort_values(
            "date"
        ).reset_index(drop=True)

        # Recalculate the digital representation.
        return self.update_state()

    def update_state(self):
        """Refresh the digital representation from available observations."""

        latest = self.history.iloc[-1]

        self.state = {
            "patient_id": self.patient_id,
            "last_observation_date": latest["date"].strftime("%Y-%m-%d"),
            "current_recorded_state": latest["state"],
            "latest_sleep_hours": latest["sleep_hours"],
            "latest_steps": latest["steps"],
            "latest_resting_hr": latest["resting_hr"],
            "latest_hrv_rmssd": latest["hrv_rmssd"],
        }

        return self.state

    def calculate_baseline(self, days=30):
        """Calculate personal averages from the preceding observation window."""

        latest_date = self.history["date"].max()
        start_date = latest_date - pd.Timedelta(days=days - 1)

        window = self.history.loc[
            self.history["date"].between(start_date, latest_date)
        ]

        features = [
            "sleep_hours",
            "steps",
            "resting_hr",
            "hrv_rmssd",
        ]

        baseline = window[features].mean()

        return baseline.to_dict()

    def calculate_deviations(self, days=30):
        """Compare the latest observation with recent personal averages."""

        baseline = self.calculate_baseline(days)
        deviations = {}

        latest = self.history.iloc[-1]

        for feature, average in baseline.items():
            current = latest[feature]

            if pd.isna(current) or pd.isna(average):
                deviations[feature] = None
            else:
                deviations[feature] = round(
                    float(current - average), 3
                )

        return deviations

    def get_data_quality(self):
        """Summarize missing measurements in the recent observation window."""

        latest_date = self.history["date"].max()
        start_date = latest_date - pd.Timedelta(days=29)

        window = self.history.loc[
            self.history["date"].between(start_date, latest_date)
        ]

        features = [
            "sleep_hours",
            "steps",
            "resting_hr",
            "hrv_rmssd",
        ]

        return {
            "observations_in_window": len(window),
            "missing_measurements": window[features]
            .isna()
            .sum()
            .to_dict(),
        }


if __name__ == "__main__":
    twin = DigitalTwin("P001")

    print("\n=== BEFORE NEW OBSERVATION ===")
    print(twin.update_state())

    simulated_observation = {
        "date": "2026-09-01",
        "sleep_hours": 5.5,
        "steps": 3200,
        "resting_hr": 76,
        "hrv_rmssd": 32.0,
    }

    print("\n=== INGESTING SIMULATED OBSERVATION ===")

    updated_state = twin.ingest_observation(
        simulated_observation
    )

    print("\n=== UPDATED DIGITAL STATE ===")
    print(updated_state)

    print("\n=== UPDATED 30-DAY BASELINE ===")
    print(twin.calculate_baseline())

    print("\n=== UPDATED DEVIATIONS ===")
    print(twin.calculate_deviations())

    print("\n=== ORIGINAL CSV REMAINS UNCHANGED ===")
    print("New observation exists only in the in-memory Digital Twin.")