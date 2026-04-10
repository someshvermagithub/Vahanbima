import pandas as pd
import numpy as np

from src.config import TRAIN_PATH, TEST_PATH, TARGET, ID_COL
from src.feature_engineering import create_features
from src.preprocessing import preprocess
from src.train import train_models

# Load data
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

test_ids = test[ID_COL]

# Feature Engineering
train = create_features(train)
test = create_features(test)

# Preprocessing
train, test = preprocess(train, test, TARGET)

# Train models
best_model, results = train_models(train, TARGET, ID_COL)

# Prepare test
X_test = test.drop(ID_COL, axis=1)

# Predict
pred = best_model.predict(X_test)

# Reverse log transform
pred = np.expm1(pred)

# Submission
submission = pd.DataFrame({
    ID_COL: test_ids,
    TARGET: pred
})

submission.to_csv("submission.csv", index=False)

print("✅ Submission file created!")