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

---

## Beginner Guide: Create a DVC Pipeline (Step-by-step)

All commands below are **PowerShell** friendly. Run them from the project folder:

```powershell
cd "./Nitish_Learning_Mlops"
```

### 0) One-time project setup (DVC + Git)

If you are starting from scratch in a new repo:

```powershell
# Initialize git (only once)
git init

# Initialize DVC (only once)
dvc init

# First commit (recommended)
git add .
git commit -m "Initialize DVC"
```

> If `dvc init` creates `.dvc/`, make sure it is committed.

---

## Pipeline Stages (Code → DVC stage → Repro)

The pattern for every stage is:
1. Write/update the Python script in `src/`
2. Add (or update) the stage in `dvc.yaml`
3. Run `dvc repro`
4. Track changes in Git

---

### 1) `data_ingestion`

**What the code does:** Loads raw data and writes it to `data/raw/`.

**Files:**
- Code: `src/data_ingestion.py`
- Output folder: `data/raw/`

**Command to add/update stage:**

```powershell
dvc stage add -n data_ingestion `
  -d src/data_ingestion.py `
  -o data/raw `
  python src/data_ingestion.py
```

**Run this stage (or the whole pipeline):**

```powershell
dvc repro data_ingestion
# or run all stages
dvc repro
```

**Track in Git:**

```powershell
git add dvc.yaml dvc.lock data/.gitignore
```

---

### 2) `data_preprocessing`

**What the code does:** Cleans/normalizes text and saves processed CSVs into `data/processed/`.

**Files:**
- Code: `src/data_preprocessing.py`
- Depends on: `data/raw/`
- Output folder: `data/processed/`

**Command to add/update stage:**

```powershell
dvc stage add -n data_preprocessing `
  -d src/data_preprocessing.py `
  -d data/raw `
  -o data/processed `
  python src/data_preprocessing.py
```

**Reproduce:**

```powershell
dvc repro data_preprocessing
```

**Track in Git:**

```powershell
git add dvc.yaml dvc.lock data/.gitignore
```

---

### 3) `feature_engineering`

**What the code does:** Converts processed text into features (example: Bag-of-Words) and writes to `data/features/`.

**Files:**
- Code: `src/feature_engineering.py`
- Depends on: `data/processed/`
- Output folder: `data/features/`

**Command to add/update stage:**

```powershell
dvc stage add -n feature_engineering `
  -d src/feature_engineering.py `
  -d data/processed `
  -o data/features `
  python src/feature_engineering.py
```

**Reproduce:**

```powershell
dvc repro feature_engineering
```

**Track in Git:**

```powershell
git add dvc.yaml dvc.lock data/.gitignore
```

---

### 4) `model_building`

**What the code does:** Trains a model using features and saves the trained artifact as `model.pkl`.

**Files:**
- Code: `src/model_building.py`
- Depends on: `data/features/`
- Output file: `model.pkl`

**Command to add/update stage:**

```powershell
dvc stage add -n model_building `
  -d src/model_building.py `
  -d data/features `
  -o model.pkl `
  python src/model_building.py
```

**Reproduce:**

```powershell
dvc repro model_building
```

**Track in Git:**

```powershell
git add dvc.yaml dvc.lock data/.gitignore
```

---

### 5) `model_evaluation`

**What the code does:** Evaluates the model and writes metrics to `metrics.json`.

**Files:**
- Code: `src/model_evaluation.py`
- Depends on: `model.pkl`
- Metrics: `metrics.json`

**Command to add/update stage:**

```powershell
dvc stage add -n model_evaluation `
  -d src/model_evaluation.py `
  -d model.pkl `
  -M metrics.json `
  python src/model_evaluation.py
```

**Reproduce:**

```powershell
dvc repro model_evaluation
```

**Track in Git:**

```powershell
git add dvc.yaml dvc.lock metrics.json
```

---

## Useful DVC commands

```powershell
# Show the pipeline DAG
dvc dag

# List pipeline stages
dvc stage list

# See what changed / what will run
dvc status

# Run all stages
dvc repro
```

---

## Commit your pipeline changes (Git)

After you add/update any stage:

```powershell
git add dvc.yaml dvc.lock data/.gitignore metrics.json params.yaml

git commit -m "Update DVC pipeline"
```

## Notes

- Make sure your environment has required Python dependencies installed.
- DVC will manage outputs cache and pipeline tracking.
- Place your DAG screenshot at `assets/dvc_dag_output.png` so it renders in this README.
