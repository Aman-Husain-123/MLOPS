import logging
import os
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

logger = logging.getLogger("model_building")
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

def load_params(params_path: str = "params.yaml") -> dict:
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)
        model_params = params['model_building']
        logger.debug("Model parameters retrieved: %s", model_params)
        return model_params
    except Exception as e:
        logger.error("Failed to load params from %s: %s", params_path, e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug("Loaded data from %s", file_path)
        return df
    except Exception as e:
        logger.error("Failed to load data from %s: %s", file_path, e)
        raise

def train_model(train_data: pd.DataFrame, params: dict) -> GradientBoostingClassifier:
    try:
        X_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values

        clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
        clf.fit(X_train, y_train)
        
        logger.debug("Model trained successfully.")
        return clf
    except Exception as e:
        logger.error("Failed to train model: %s", e)
        raise

def save_model(model: GradientBoostingClassifier, file_path: str) -> None:
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug("Model saved to %s", file_path)
    except Exception as e:
        logger.error("Failed to save model to %s: %s", file_path, e)
        raise

def main() -> None:
    try:
        params = load_params('params.yaml')

        train_data = load_data('./data/features/train_bow.csv')

        model = train_model(train_data, params)

        save_model(model, 'model.pkl')

        logger.info("Model building completed successfully.")
    except Exception as e:
        logger.error("Failed to complete the model building process: %s", e)
        raise

if __name__ == "__main__":
    main()
