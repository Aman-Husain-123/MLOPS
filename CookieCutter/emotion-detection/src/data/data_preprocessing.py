import logging
import os
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# logging configure
logger = logging.getLogger("data_preprocessing")
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


def setup_nltk() -> None:
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.debug("NLTK data downloaded successfully.")
    except Exception as e:
        logger.error("Failed to download NLTK data: %s", e)
        raise

def lemmatization(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        text_list = str(text).split()
        text_list = [lemmatizer.lemmatize(y) for y in text_list]
        return " ".join(text_list)
    except Exception as e:
        logger.error("Error in lemmatization: %s", e)
        raise

def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        text_list = [i for i in str(text).split() if i not in stop_words]
        return " ".join(text_list)
    except Exception as e:
        logger.error("Error in remove_stop_words: %s", e)
        raise

def removing_numbers(text: str) -> str:
    try:
        text = ''.join([i for i in str(text) if not i.isdigit()])
        return text
    except Exception as e:
        logger.error("Error in removing_numbers: %s", e)
        raise

def lower_case(text: str) -> str:
    try:
        text_list = str(text).split()
        text_list = [y.lower() for y in text_list]
        return " ".join(text_list)
    except Exception as e:
        logger.error("Error in lower_case: %s", e)
        raise

def removing_punctuations(text: str) -> str:
    try:
        text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', str(text))
        text = text.replace('؛', "")
        text = re.sub(r'\s+', ' ', text)
        return " ".join(text.split()).strip()
    except Exception as e:
        logger.error("Error in removing_punctuations: %s", e)
        raise

def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', str(text))
    except Exception as e:
        logger.error("Error in removing_urls: %s", e)
        raise

def remove_small_sentences(df: pd.DataFrame, column: str = 'content') -> pd.DataFrame:
    try:
        df = df.dropna(subset=[column])
        df = df[df[column].apply(lambda x: len(str(x).split()) >= 3)]
        return df
    except Exception as e:
        logger.error("Error in remove_small_sentences: %s", e)
        raise

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        df = remove_small_sentences(df)
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logger.debug("Text normalization completed.")
        return df
    except Exception as e:
        logger.error("Error in normalize_text: %s", e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug("Loaded data from %s", file_path)
        return df
    except Exception as e:
        logger.error("Failed to load data from %s: %s", file_path, e)
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
        setup_nltk()

        train_data = load_data('./data/raw/train.csv')
        test_data = load_data('./data/raw/test.csv')

        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        save_data(train_processed_data, os.path.join("data", "processed", "train_processed.csv"))
        save_data(test_processed_data, os.path.join("data", "processed", "test_processed.csv"))

        logger.info("Data preprocessing completed successfully.")
    except Exception as e:
        logger.error("Failed to complete the data preprocessing process: %s", e)
        raise

if __name__ == "__main__":
    main()