import logging
import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(os.path.join("logs", "errors.log"))
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug("Loaded model from %s", file_path)
        return model
    except Exception as e:
        logger.error("Failed to load model from %s: %s", file_path, e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug("Loaded data from %s", file_path)
        return df
    except Exception as e:
        logger.error("Failed to load data from %s: %s", file_path, e)
        raise

def evaluate_model(model, test_data: pd.DataFrame) -> dict:
    try:
        X_test = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'auc': float(auc)
        }

        logger.debug("Model evaluated successfully. Metrics: %s", metrics_dict)
        return metrics_dict
    except Exception as e:
        logger.error("Failed to evaluate model: %s", e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug("Metrics saved to %s", file_path)
    except Exception as e:
        logger.error("Failed to save metrics to %s: %s", file_path, e)
        raise

def main() -> None:
    try:
        model = load_model('model.pkl')
        test_data = load_data('./data/features/test_bow.csv')

        metrics = evaluate_model(model, test_data)

        save_metrics(metrics, 'metrics.json')

        logger.info("Model evaluation completed successfully.")
    except Exception as e:
        logger.error("Failed to complete the model evaluation process: %s", e)
        raise

if __name__ == "__main__":
    main()
