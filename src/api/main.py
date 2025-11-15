#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# src/api/main.py
from fastapi import FastAPI
from ..model.factory import get_predictor
from .schema import PredictRequest

app = FastAPI(title="Insurance Premium Predictor API")

predictor = get_predictor()

@app.get("/")
def home():
    return {"message": "Insurance Premium Predictor API is running"}

@app.post("/predict")
def predict(req: PredictRequest):
    data = req.dict()
    result = predictor.predict(data)
    return {"input": data, "prediction": result}




