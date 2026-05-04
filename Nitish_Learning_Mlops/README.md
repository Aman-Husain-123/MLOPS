# Nitish Learning MLOps — DVC Pipeline Project

This project demonstrates an end-to-end **MLOps pipeline** using **DVC (Data Version Control)**, structured into modular stages for: data ingestion → preprocessing → feature engineering → model building → model evaluation.

## Project Structure

```
Nitish_Learning_Mlops/
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── metrics.json
├── model.pkl
├── data/
│   ├── raw/                # train.csv, test.csv (from ingestion)
│   ├── processed/          # train_processed.csv, test_processed.csv (from preprocessing)
│   └── features/           # train_bow.csv, test_bow.csv (from feature_engineering)
└── src/
    ├── data_ingestion.py
    ├── data_preprocessing.py
    ├── feature_engineering.py
    ├── model_building.py
    └── model_evaluation.py
```

## DVC Pipeline DAG

Below is the current pipeline DAG output:

![DVC DAG Output](./assets/dvc_dag_output.png)

## Pipeline Stages

### 1. `data_ingestion`
- **Script**: `src/data_ingestion.py`
- **Output**: `data/raw/`

### 2. `data_preprocessing`
- **Script**: `src/data_preprocessing.py`
- **Dependencies**: `data/raw/`
- **Output**: `data/processed/`

### 3. `feature_engineering`
- **Script**: `src/feature_engineering.py`
- **Dependencies**: `data/processed/`
- **Output**: `data/features/`

### 4. `model_building`
- **Script**: `src/model_building.py`
- **Dependencies**: `data/features/`
- **Output**: `model.pkl`

### 5. `model_evaluation`
- **Script**: `src/model_evaluation.py`
- **Dependencies**: `model.pkl`
- **Metrics Output**: `metrics.json`

## How to Run

### 1) Reproduce the whole pipeline

```bash
dvc repro
```

### 2) Show the pipeline DAG

```bash
dvc dag
```

### 3) Track changes in Git

```bash
git add dvc.yaml dvc.lock data/.gitignore
```

## Notes

- Make sure your environment has required Python dependencies installed.
- DVC will manage outputs cache and pipeline tracking.

---

If you want, you can add more stages such as model registration, deployment, and monitoring.
