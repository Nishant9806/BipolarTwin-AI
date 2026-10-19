# BipolarTwin AI
### Digital Twin Challenge 2026 — Happiest Health

**BipolarTwin AI** is a research prototype that explores how a digital twin can represent an individual's evolving health state using longitudinal observations, personal baselines, machine learning, and experimental mood forecasting.

The project combines a Python-based digital twin engine, machine-learning models, a FastAPI backend, and a React dashboard.

> **Research disclaimer:** BipolarTwin AI is an experimental proof of concept. It is not a clinically validated diagnostic, monitoring, or treatment system. Predictions must not be used for medical decisions.

---

## 1. Problem Statement

Bipolar disorder involves changes in mood over time. Understanding how daily observations relate to an individual's historical patterns may help researchers investigate longitudinal changes in mood and behavior.

BipolarTwin AI explores a digital-twin approach that combines historical information with daily observations to:

- Maintain a representation of an individual's recorded health state.
- Calculate personal baselines from recent observations.
- Identify deviations from those baselines.
- Estimate the current recorded mood category using a machine-learning model.
- Experiment with forecasting mood states seven calendar days ahead.
- Present measurements and experimental model outputs through a research dashboard.

The project is a proof of concept, not evidence that digital-twin predictions can reliably identify or prevent mood episodes.

---

## 2. System Architecture

```text
        PATIENT PROFILE
              |
      HISTORICAL RECORDS
              |
       DAILY OBSERVATIONS
              |
              v
      DIGITAL TWIN ENGINE
              |
       +------+------+
       |             |
       v             v
 PERSONAL BASELINE  DATA QUALITY
       |
       v
 CURRENT MOOD CLASSIFIER
   Logistic Regression
       |
       v
 SEVEN-DAY MOOD FORECAST
 Experimental model
       |
       v
    FASTAPI BACKEND
       |
       v
    REACT DASHBOARD
```

The current implementation contains a Digital Twin engine, a current mood classifier, an experimental forecasting script, a FastAPI backend, and a React frontend.

**Implementation note:** The standalone seven-day, three-class forecasting experiment is not yet integrated into the API. The API currently returns an experimental binary `relapse_7d` estimate. These outputs must not be confused.

---

## 3. Key Features

### Digital Twin Engine

Implemented in `digital_twin.py`.

- Loads patient profile information.
- Loads historical daily observations.
- Loads medication and laboratory records.
- Tracks the latest recorded observation.
- Calculates rolling personal baselines.
- Calculates deviations from personal baselines.
- Reports missing measurements.
- Supports in-memory ingestion of new observations.

Newly ingested observations are not persisted to a database or external storage.

### Current Mood Classification

Implemented in `mood_classifier.py`.

The model uses Logistic Regression to classify recorded mood states:

- Stable
- Depression
- Mania

Features include sleep duration, sleep timing, steps, resting heart rate, heart-rate variability, activity timing, and medication indicators.

The preprocessing pipeline includes numerical imputation, scaling, categorical imputation, and one-hot encoding.

### Experimental Mood Forecasting

Implemented in `mood_forecaster.py`.

The experimental model predicts the recorded mood state exactly seven calendar days after an eligible observation.

The target is constructed by matching patient identifiers and dates, rather than shifting rows blindly.

The experiment uses a chronological train/test split and compares Logistic Regression with a persistence baseline.

### Research Dashboard

The React frontend displays:

- Latest recorded measurements.
- Personal baseline values.
- Deviations from personal baselines.
- Current mood-classification probabilities.
- An experimental future-outcome estimate.
- Research limitations.

The dashboard is a single-page overview. Its visual navigation items do not represent completed separate pages.

---

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn |
| Backend | FastAPI |
| ASGI server | Uvicorn |
| Frontend | React, TypeScript, Vite |
| Charts | Recharts |
| Icons | Lucide React |
| Version control | Git and GitHub |

---

## 5. Dataset

The current prototype uses a local dataset containing four principal CSV files:

- `patients.csv`
- `daily_data.csv`
- `ehr_meds.csv`
- `ehr_labs.csv`

A patient overview image is also present in the local dataset directory.

The daily dataset contains 730 observations for one patient, identified as `P001`, spanning September 2024 through August 2026.

The recorded mood-state distribution is:

| State | Observations |
|---|---:|
| Stable | 521 |
| Depression | 153 |
| Mania | 56 |
| Total | 730 |

### Data provenance

The provenance of the current dataset has not been independently established. It must not be represented as verified clinical patient data.

The raw dataset is excluded from this repository until its source, license, redistribution permissions, and privacy implications have been verified.

Consequently, the application cannot run successfully from the repository alone without obtaining the dataset through an authorized source and placing the required files in the expected local directory.

### Data limitations

- Only one patient is represented.
- Some wearable measurements are missing.
- Laboratory observations are sparse.
- Some medication indicators have no variation.
- The original mood-label definitions have not been independently verified.
- The original definition of `relapse_7d` remains uncertain.

Repeated observations from one patient do not establish generalization to other patients.

---

## 6. Experimental Model Evaluation

### Binary `relapse_7d` model

The initial binary forecasting experiment produced:

| Metric | Result |
|---|---:|
| Accuracy | 0.80 |
| ROC-AUC | 0.5147 |
| Positive-class precision | 0.14 |
| Positive-class recall | 0.21 |
| Positive-class F1-score | 0.17 |

The ROC-AUC is close to random ranking, and positive-class performance is poor.

The apparently high accuracy should not be interpreted as evidence of reliable relapse prediction.

### Seven-day mood-state forecasting experiment

The three-class Logistic Regression experiment produced:

| Metric | Logistic Regression | Persistence baseline |
|---|---:|---:|
| Accuracy | 73.79% | 75.86% |
| Macro F1-score | 0.67 | 0.66 |
| Stable F1-score | 0.78 | 0.80 |
| Depression F1-score | 0.74 | 0.77 |
| Mania F1-score | 0.50 | 0.42 |

The persistence baseline predicts that the current recorded mood state continues for seven days.

Logistic Regression achieves a slightly higher macro F1-score, while the persistence baseline achieves higher overall accuracy.

The test set contains only 12 mania observations, making the mania metrics uncertain.

These results do not establish that Logistic Regression is superior to the baseline or that the model generalizes to other patients.

### Current mood classifier

The current mood classifier uses a chronological train/test split in its standalone evaluation script.

Its evaluation is limited to the same single-patient dataset and does not establish independent clinical validity.

---

## 7. Project Structure

```text
BipolarTwin-AI/
├── .gitignore
├── README.md
├── api.py
├── check_forecast_labels.py
├── dataset_audit.py
├── digital_twin.py
├── inspect_labels.py
├── mood_classifier.py
├── mood_forecaster.py
├── run_bipolartwin.py
└── frontend/
    ├── package.json
    ├── package-lock.json
    ├── index.html
    └── src/
        ├── App.tsx
        ├── App.css
        ├── index.css
        └── main.tsx
```

The local dataset directory and Python virtual environment are excluded from version control.

---

## 8. Installation and Setup

### Prerequisites

- Python 3.12 or a compatible Python version.
- Node.js and npm.
- Git.

### Step 1: Clone the repository

```bash
git clone https://github.com/Nishant9806/BipolarTwin-AI.git
cd BipolarTwin-AI
```

### Step 2: Create a Python virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Python dependencies

Install the packages used by the backend and models:

```powershell
python -m pip install pandas numpy scikit-learn fastapi "uvicorn[standard]"
```

### Step 4: Obtain the dataset

Obtain the dataset through an authorized source.

After verifying its permissions, place the required CSV files in:

```text
dataset/
├── patients.csv
├── daily_data.csv
├── ehr_meds.csv
└── ehr_labs.csv
```

The directory name must be lowercase because the Python files currently reference `dataset`.

Do not publish restricted or identifiable health data.

### Step 5: Start the backend

From the project root:

```powershell
python -m uvicorn api:app --reload
```

The API is expected to be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### Step 6: Start the frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite. The default is usually:

```text
http://localhost:5173
```

Keep both the backend and frontend running while testing the dashboard.

---

## 9. API

### Health endpoint

```http
GET /
```

Returns basic application status information.

### Patient endpoint

```http
GET /api/patient/P001
```

Returns the current digital-twin representation, personal baselines, baseline deviations, current mood-classification probabilities, and the experimental binary future-outcome estimate.

The endpoint requires the locally available dataset and model dependencies.

---

## 10. Limitations and Future Work

Current limitations include:

1. The existing dataset contains only one patient.
2. Dataset provenance and target definitions remain unverified.
3. The binary forecasting model has weak discrimination.
4. The seven-day mood classifier does not outperform the persistence baseline in overall accuracy.
5. Mania evaluation is based on a small test sample.
6. Cross-patient generalization has not been established.
7. The current API does not expose the separate three-class forecasting experiment.
8. New observations are stored only in memory.
9. Real wearable-device integration is not implemented.
10. Clinical validity has not been established.

Potential future work includes:

- Identifying suitable, appropriately licensed longitudinal datasets.
- Evaluating models on independent participants.
- Improving target definitions and temporal evaluation.
- Integrating the selected forecasting model into the API.
- Adding longitudinal trend visualizations.
- Implementing persistent observation storage.
- Investigating real wearable integrations.
- Conducting appropriate clinical validation before considering clinical use.

These are proposed improvements, not completed capabilities.

---

## 11. Intended Use and Safety

BipolarTwin AI is intended for research, educational demonstrations, and exploration of digital-twin architectures.

It must not be used to diagnose bipolar disorder, predict an individual's clinical course with assumed certainty, recommend medication changes, or replace professional medical care.

Recorded observations, simulated observations, and machine-learning predictions must remain clearly distinguishable.

The project has not undergone clinical validation.

---

## 12. Competition

**Digital Twin Challenge 2026 — Happiest Health**

Project: BipolarTwin AI

Repository: https://github.com/Nishant9806/BipolarTwin-AI

This repository documents a research prototype and its current limitations. Competition participation does not imply endorsement or clinical validation.

---

## 13. License

The repository license will be specified after the project owner selects an appropriate open-source license and verifies the licensing status of the project's code and dependencies.