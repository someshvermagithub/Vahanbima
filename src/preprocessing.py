import pandas as pd
from src.utils import clean_column_names

def preprocess(train, test, target):
    # Combine for consistent encoding
    test[target] = 0
    df = pd.concat([train, test], axis=0)

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Clean column names
    df = clean_column_names(df)

    # Split back
    train = df[df[target] != 0]
    test = df[df[target] == 0]

    test = test.drop(target, axis=1)

    return train, test