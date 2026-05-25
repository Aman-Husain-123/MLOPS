import logging
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger("feature_engineering")
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

def load_params(params_path: str = "params.yaml") -> int:
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)
        max_features = int(params['feature_engineering']['max_features'])
        logger.debug("max_features retrieved: %s", max_features)
        return max_features
    except Exception as e:
        logger.error("Failed to load params from %s: %s", params_path, e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug("Loaded data from %s", file_path)
        return df
    except Exception as e:
        logger.error("Failed to load data from %s: %s", file_path, e)
        raise

def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        vectorizer = CountVectorizer(max_features=max_features)

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug("Applied Bag of Words successfully.")
        return train_df, test_df
    except Exception as e:
        logger.error("Failed to apply Bag of Words: %s", e)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug("Saved data to %s", file_path)
    except Exception as e:
        logger.error("Failed to save data to %s: %s", file_path, e)
        raise

def main() -> None:
    try:
        max_features = load_params('params.yaml')

        train_data = load_data('./data/processed/train_processed.csv')
        test_data = load_data('./data/processed/test_processed.csv')

        train_df, test_df = apply_bow(train_data, test_data, max_features)

        save_data(train_df, os.path.join("data", "features", "train_bow.csv"))
        save_data(test_df, os.path.join("data", "features", "test_bow.csv"))

        logger.info("Feature engineering completed successfully.")
    except Exception as e:
        logger.error("Failed to complete the feature engineering process: %s", e)
        raise

if __name__ == "__main__":
    main()
