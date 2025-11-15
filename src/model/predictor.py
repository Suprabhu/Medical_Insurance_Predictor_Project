#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# src/model/predictor.py
import os, json, joblib, numpy as np
from typing import Dict, Any

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_v1.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "models/feature_columns.json")

class Predictor:
    def __init__(self, model_path: str = MODEL_PATH, features_path: str = FEATURES_PATH):
        self.model_path = model_path
        self.features_path = features_path
        self.model = None
        self.feature_order = None
        self._load_feature_order()
        self._load_model()

    def _load_feature_order(self):
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"features file not found: {self.features_path}")
        with open(self.features_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "feature_columns" in data:
            self.feature_order = data["feature_columns"]
        elif isinstance(data, list):
            self.feature_order = data
        else:
            raise ValueError("Unexpected format in feature_columns.json")

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        self.model = joblib.load(self.model_path)

    def _preprocess(self, features: Dict[str, Any]) -> np.ndarray:
        f = {k.lower(): v for k, v in (features or {}).items()}

        # basic conversions and defaults
        age = int(f.get("age", 0))
        sex = str(f.get("sex", "")).lower()
        bmi = float(f.get("bmi", 0.0))
        children = int(f.get("children", 0))
        smoker = str(f.get("smoker", "")).lower()
        region = str(f.get("region", "")).lower()

        base = {}
        # numeric fields
        if "age" in self.feature_order:
            base["age"] = age
        if "bmi" in self.feature_order:
            base["bmi"] = bmi
        if "children" in self.feature_order:
            base["children"] = children

        # sex encoding
        if "sex" in self.feature_order:
            base["sex"] = 1 if sex == "male" else 0
        else:
            if any(col.startswith("sex_") for col in self.feature_order):
                base["sex_male"] = 1 if sex == "male" else 0
                base["sex_female"] = 0 if sex == "male" else 1

        # smoker encoding
        if "smoker" in self.feature_order:
            base["smoker"] = 1 if smoker == "yes" else 0
        else:
            if any(col.startswith("smoker_") for col in self.feature_order):
                base["smoker_yes"] = 1 if smoker == "yes" else 0
                base["smoker_no"] = 0 if smoker == "yes" else 1

        # region one-hot
        region_keys = [c for c in self.feature_order if c.startswith("region")]
        for rk in region_keys:
            suffix = rk.split("region_", 1)[-1]
            base[rk] = 1.0 if suffix == region else 0.0

        vec = [float(base.get(col, 0.0)) for col in self.feature_order]
        return np.array(vec).reshape(1, -1)

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            self._load_model()
        X = self._preprocess(features)
        pred = float(self.model.predict(X)[0])
        confidence = None
        try:
            if hasattr(self.model, "estimators_"):
                preds = [est.predict(X)[0] for est in self.model.estimators_]
                confidence = float(np.std(preds))
        except Exception:
            confidence = None
        return {"premium": pred, "model_version": os.path.basename(self.model_path), "confidence": confidence}

