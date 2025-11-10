#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# src/train/train_model.py
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def load_data(path="src/data/insurance.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def preprocess(df):
    df = df.copy()

    # Encode categorical features
    if 'sex' in df.columns:
        df['sex'] = (df['sex'] == 'male').astype(int)
    if 'smoker' in df.columns:
        df['smoker'] = (df['smoker'] == 'yes').astype(int)
    if 'region' in df.columns:
        df = pd.get_dummies(df, columns=['region'], prefix='region', drop_first=False)

    # Fill missing numeric values (if any)
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def train(save_model_path="models/model_v1.pkl", save_metrics_path="models/metrics.json"):
    os.makedirs("models", exist_ok=True)

    df = load_data()
    df = preprocess(df)

    target = 'charges'
    assert target in df.columns, f"Target column '{target}' not found in dataset"

    X = df.drop(columns=[target])
    y = df[target]

    # Save feature columns for predictor
    with open("models/feature_columns.json", "w") as f:
        json.dump({"feature_columns": X.columns.tolist()}, f, indent=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0])
    }

    joblib.dump(model, save_model_path)
    with open(save_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Model trained and saved!")
    print("📁 Model path:", save_model_path)
    print("📊 Metrics:", metrics)
    print("📜 Feature columns saved to models/feature_columns.json")


if __name__ == "__main__":
    train()

