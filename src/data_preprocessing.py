import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    """
    Load dataset from CSV file.
    """
    data = pd.read_csv(path)
    return data


def preprocess_data(data):
    """
    dropping na values
    """
    data = data.dropna()

    X = data.drop("quality", axis=1) # response variable dropped
    y = data["quality"]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
