#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# src/api/schema.py
from pydantic import BaseModel

class PredictRequest(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str



