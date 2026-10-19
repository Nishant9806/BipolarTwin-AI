from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from run_bipolartwin import (
    train_mood_model,
    train_forecast_model,
)
from digital_twin import DigitalTwin


app = FastAPI(
    title="BipolarTwin AI",
    description=(
        "Research prototype for longitudinal health-state "
        "monitoring and exploratory mood-related forecasting."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the models once when the API starts.
mood_model = train_mood_model()
forecast_model = train_forecast_model()


@app.get("/")
def home():
    return {
        "project": "BipolarTwin AI",
        "status": "running",
        "purpose": "Research prototype",
    }


@app.get("/api/patient/{patient_id}")
def get_patient_state(patient_id: str):
    try:
        twin = DigitalTwin(patient_id)

        latest = twin.history.iloc[[-1]]

        mood_probabilities = mood_model.predict_proba(
            latest[
                __import__("mood_classifier").FEATURES
            ]
        )[0]

        mood_classes = mood_model.named_steps[
            "classifier"
        ].classes_

        mood_output = {
            label: round(float(probability), 4)
            for label, probability in zip(
                mood_classes,
                mood_probabilities,
            )
        }

        forecast_features = __import__(
            "mood_forecaster"
        ).FEATURES

        relapse_probability = forecast_model.predict_proba(
            latest[forecast_features]
        )[0, 1]

        return {
            "patient_id": patient_id,
            "digital_twin": twin.update_state(),
            "personal_baseline": twin.calculate_baseline(),
            "baseline_deviations": twin.calculate_deviations(),
            "current_mood": {
                "predicted_class": max(
                    mood_output,
                    key=mood_output.get,
                ),
                "probabilities": mood_output,
            },
            "future_outcome": {
                "target": "relapse_7d",
                "estimated_probability": round(
                    float(relapse_probability), 4
                ),
                "validation_status": "experimental",
            },
            "limitations": [
                "Dataset contains one patient.",
                "Future-label provenance is unverified.",
                "Clinical validity has not been established.",
            ],
        }

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error
