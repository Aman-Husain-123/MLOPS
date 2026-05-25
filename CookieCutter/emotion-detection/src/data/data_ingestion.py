import logging
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# logging configure
logger = logging.getLogger("data_ingestion")
logger.setLevel('DEBUG')

# Avoid adding handlers multiple times (common in notebooks/re-runs)
if not logger.handlers:
    # Console handler for all logs (DEBUG and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Ensure logs folder exists
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(os.path.join("logs", "errors.log"))
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_params(params_path: str = "params.yaml") -> float:
    """Load test_size from params.yaml (DVC-friendly)."""
    try:
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"'{params_path}' not found in repo root.")

        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)

        if not isinstance(params, dict):
            raise ValueError(f"'{params_path}' is empty/invalid YAML.")

        test_size = float(params["data_ingestion"]["test_size"])
        logger.debug("test_size retrieved: %s", test_size)
        return test_size

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    except (KeyError, TypeError, ValueError) as e:
        logger.error("Invalid params.yaml structure/value: %s", e)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML parse error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error in load_params: %s", e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug("Loaded data from URL. Shape=%s", df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file from %s. Error=%s", data_url, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading data: %s", e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to happiness/sadness and map labels to 1/0."""
    try:
        df = df.copy()
        df.drop(columns=["tweet_id"], inplace=True)

        final_df = df[df["sentiment"].isin(["happiness", "sadness"])].copy()
        # Avoid SettingWithCopyWarning
        final_df.loc[:, "sentiment"] = final_df["sentiment"].replace(
            {"happiness": 1, "sadness": 0}
        )

        logger.debug("Preprocessed data. Shape=%s", final_df.shape)
        return final_df
    except KeyError as e:
        logger.error("Missing expected column: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_dir: str = "data") -> None:
    try:
        raw_path = os.path.join(data_dir, "raw")
        os.makedirs(raw_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_path, "test.csv"), index=False)

        logger.debug("Saved train/test to %s", raw_path)
    except Exception as e:
        logger.error("Unexpected error while saving data: %s", e)
        raise


def main() -> None:
    data_url = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"

    try:
        # FIX: was params1.yaml; should be params.yaml
        test_size = load_params("params.yaml")

        df = load_data(data_url=data_url)
        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=42
        )
        save_data(train_data, test_data, data_dir="data")

        logger.info("Data ingestion completed successfully.")
    except Exception as e:
        logger.error("Failed to complete the data ingestion process: %s", e)
        raise  # let DVC see the failure properly


if __name__ == "__main__":
    main()