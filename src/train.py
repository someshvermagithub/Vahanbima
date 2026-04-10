from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

from src.model import get_models

def train_models(train, target, id_col):
    X = train.drop([target, id_col], axis=1)
    y = np.log1p(train[target])  # log transform

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_models()
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        score = r2_score(y_val, pred)
        results[name] = score

        print(f"{name} R2: {score:.4f}")

    # Best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    print(f"\nBest Model: {best_model_name}")

    return best_model, results