#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# src/model/factory.py
from typing import Dict
from .predictor import Predictor

def get_predictor(config: Dict = None) -> Predictor:
    config = config or {}
    return Predictor(model_path=config.get("model_path", "models/model_v1.pkl"),
                     features_path=config.get("features_path", "models/feature_columns.json"))


