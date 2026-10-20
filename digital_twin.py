from pathlib import Path
import numpy as np
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
        patient = patients.loc[patients["patient_id"] == patient_id]

        if patient.empty:
            raise ValueError(f"Patient {patient_id} was not found.")

        self.profile = patient.iloc[0].to_dict()

        # Retain this patient's longitudinal observations.
        daily_data["date"] = pd.to_datetime(
            daily_data["date"], errors="coerce"
        )

        self.history = (
            daily_data.loc[daily_data["patient_id"] == patient_id]
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

    def calculate_bci(self, alpha=1.0, beta=0.8, gamma=0.6):
        """
        Compute the Bipolar Circadian Index (BCI):
            BCI(t) = alpha * Phi(t) + beta * Delta(t) - gamma * Psi(t)
        
        Components:
            Phi(t): Phase alignment (inverse normalized sleep onset deviation from baseline)
            Delta(t): Circadian amplitude (normalized heart rate amplitude / activity)
            Psi(t): Sleep-wake variability (7-day rolling std of sleep duration)
        """
        latest_idx = len(self.history) - 1
        latest = self.history.iloc[latest_idx]

        # 1. Psi(t): Sleep-wake variability over rolling 7-day window
        window_7d = self.history.iloc[max(0, latest_idx - 6) : latest_idx + 1]
        sleep_std = window_7d["sleep_hours"].std()
        psi_t = float(sleep_std) if not pd.isna(sleep_std) else 0.5

        # 2. Delta(t): Circadian amplitude (normalized hr_amplitude or steps proxy)
        if "hr_amplitude" in self.history.columns and not pd.isna(latest.get("hr_amplitude")):
            raw_amp = float(latest["hr_amplitude"])
            baseline_amp = self.history["hr_amplitude"].tail(30).mean()
            delta_t = float(raw_amp / (baseline_amp + 1e-5)) if baseline_amp else 1.0
        else:
            # Fallback to normalized daily steps relative to 30-day baseline
            recent_steps_mean = self.history["steps"].tail(30).mean()
            delta_t = float(latest["steps"] / (recent_steps_mean + 1e-5)) if recent_steps_mean else 1.0
        delta_t = float(np.clip(delta_t, 0.0, 2.0))

        # 3. Phi(t): Circadian phase alignment
        if "sleep_onset_hour" in self.history.columns and not pd.isna(latest.get("sleep_onset_hour")):
            onset = float(latest["sleep_onset_hour"])
            baseline_onset = self.history["sleep_onset_hour"].tail(30).mean()
            diff = abs(onset - baseline_onset)
            phi_t = float(max(0.0, 1.0 - (diff / 4.0)))  # Declines if onset deviates by > 4 hours
        else:
            phi_t = 1.0

        # BCI calculation
        bci_score = (alpha * phi_t) + (beta * delta_t) - (gamma * psi_t)

        return {
            "bci_score": round(float(bci_score), 3),
            "phi_phase_alignment": round(float(phi_t), 3),
            "delta_amplitude": round(float(delta_t), 3),
            "psi_variability": round(float(psi_t), 3),
            "interpretation": (
                "Optimal" if bci_score >= 1.2
                else "Moderate Circadian Disruption" if bci_score >= 0.7
                else "Severe Circadian Misalignment"
            )
        }

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
        new_row["date"] = pd.to_datetime(new_row["date"], errors="raise")

        if new_row["date"] <= self.history["date"].max():
            raise ValueError(
                "New observation must be later than the latest record."
            )

        new_row = pd.DataFrame([new_row]).reindex(
            columns=self.history.columns
        )

        self.history = pd.concat([self.history, new_row], ignore_index=True)
        self.history = self.history.sort_values("date").reset_index(drop=True)

        return self.update_state()

    def update_state(self):
        """Refresh the digital representation from available observations."""
        latest = self.history.iloc[-1]
        bci_metrics = self.calculate_bci()

        self.state = {
            "patient_id": self.patient_id,
            "last_observation_date": latest["date"].strftime("%Y-%m-%d"),
            "current_recorded_state": latest.get("state", None),
            "latest_sleep_hours": latest.get("sleep_hours", None),
            "latest_steps": latest.get("steps", None),
            "latest_resting_hr": latest.get("resting_hr", None),
            "latest_hrv_rmssd": latest.get("hrv_rmssd", None),
            "bci": bci_metrics,
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

        available = [f for f in features if f in window.columns]
        baseline = window[available].mean()
        return baseline.to_dict()

    def calculate_deviations(self, days=30):
        """Compare the latest observation with recent personal averages."""
        baseline = self.calculate_baseline(days)
        deviations = {}
        latest = self.history.iloc[-1]

        for feature, average in baseline.items():
            current = latest.get(feature)
            if pd.isna(current) or pd.isna(average):
                deviations[feature] = None
            else:
                deviations[feature] = round(float(current - average), 3)

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

        available = [f for f in features if f in window.columns]
        return {
            "observations_in_window": len(window),
            "missing_measurements": window[available].isna().sum().to_dict(),
        }


if __name__ == "__main__":
    twin = DigitalTwin("P001")
    state = twin.update_state()
    print("\n=== DIGITAL TWIN STATE WITH BCI ===")
    print(f"Date: {state['last_observation_date']}")
    print(f"BCI Score: {state['bci']['bci_score']}")
    print(f"Phi(t) Alignment: {state['bci']['phi_phase_alignment']}")
    print(f"Delta(t) Amplitude: {state['bci']['delta_amplitude']}")
    print(f"Psi(t) Variability: {state['bci']['psi_variability']}")
    print(f"Status: {state['bci']['interpretation']}")