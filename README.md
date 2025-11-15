# Medical_Insurance_Predictor_Project
Medical Insurance Premium Predictor

A production-ready machine-learning system built with FastAPI, Scikit-Learn, and a layered architecture.
This project predicts health-insurance premiums using customer attributes such as age, BMI, smoking habits, region, etc.

Below is a **clean, polished, copy-pastable README.md** for your GitHub repository.
It is structured exactly the way industry projects write it — clear, professional, and complete.

---

# 🚑 Medical Insurance Premium Predictor

**A production-ready machine-learning system built with FastAPI, Scikit-Learn, and a layered architecture.**
This project predicts health-insurance premiums using customer attributes such as age, BMI, smoking habits, region, etc.

---

## 📌 Table of Contents

* [Overview](#overview)
* [Architecture](#architecture)
* [Quality Attributes Achieved](#quality-attributes-achieved)
* [Project Structure](#project-structure)
* [Tech Stack](#tech-stack)
* [Setup Instructions](#setup-instructions)
* [Training the Model](#training-the-model)
* [Running the API](#running-the-api)
* [API Usage](#api-usage)
* [Testing & Coverage](#testing--coverage)
* [Performance Benchmark](#performance-benchmark)
* [Future Enhancements](#future-enhancements)

---

## 📘 Overview

This project implements a complete end-to-end ML workflow:

✔️ Data ingestion
✔️ Preprocessing & feature engineering
✔️ Model training using Random Forest
✔️ Model persistence using joblib
✔️ Modular predictor API
✔️ Layered, maintainable architecture
✔️ Automated testing with pytest
✔️ Code coverage measurement

The system outputs not only the premium but also the model version and a confidence value.

---

## 🏗 Architecture

This system follows a **four-layered architecture**:

### **1️⃣ Presentation Layer**

* Swagger UI (FastAPI auto-generated)
* Receives user inputs
* Sends results back as JSON

### **2️⃣ API Layer (FastAPI)**

* `/predict` endpoint
* Validates request body using Pydantic
* Converts request → internal model call

### **3️⃣ Business Logic Layer**

* `Predictor` class
* Loads model + feature order
* Handles preprocessing (encoding, ordering)
* Computes prediction + confidence

### **4️⃣ Model & Data Layer**

* Trained `model_v1.pkl`
* `metrics.json` (MAE, R², sample size)
* `feature_columns.json`

### 📌 Architecture Diagram

(Place the generated PNG: `architecture_diagram.png` here in your repo.)

---

## ⭐ Quality Attributes Achieved

### **✔ Maintainability**

* Code structured into layers (`api/`, `model/`, `train/`)
* Predictor logic is isolated → easy to upgrade model versions

### **✔ Reliability**

* Automated tests with pytest
* 81% code coverage
* Stable predictions through feature-order enforcement

### **✔ Performance**

* API tested with 50 consecutive requests
* Average latency ~0.6 seconds per 50 requests
* Suitable for real-time inference workloads

### **✔ Reusability**

* Model can be swapped without changing API or UI
* Modular training script

### **✔ Scalability**

* Stateless API
* Can be containerized & deployed with load balancers
* Supports horizontal scaling

---

## 📁 Project Structure

```
Medical_Insurance_Predictor_Project/
│── .venv/
│── models/
│   ├── model_v1.pkl
│   ├── metrics.json
│   └── feature_columns.json
│
│── src/
│   ├── api/
│   │   └── main.py
│   ├── model/
│   │   ├── predictor.py
│   │   └── factory.py
│   ├── train/
│   │   └── train_model.py
│   └── data/
│       └── insurance.csv
│
│── tests/
│   └── test_predictor.py
│
│── requirements.txt
│── README.md
```

---

## ⚙ Tech Stack

* **Python 3.9**
* **FastAPI**
* **Scikit-Learn**
* **NumPy / Pandas**
* **Joblib**
* **Pytest**
* **Uvicorn**

---

## 🔧 Setup Instructions

### **1. Clone the repo**

```bash
git clone https://github.com/<your-username>/Medical_Insurance_Predictor_Project.git
cd Medical_Insurance_Predictor_Project
```

### **2. Create & activate virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🧠 Training the Model

Run:

```bash
python src/train/train_model.py
```

Outputs:

* `models/model_v1.pkl`
* `models/metrics.json`
* `models/feature_columns.json`

---

## 🚀 Running the API

Start the FastAPI server:

```bash
uvicorn src.api.main:app --reload
```

Open Swagger UI in browser:
📍 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🎯 API Usage

### **POST /predict**

**Request JSON**

```json
{
  "age": 45,
  "sex": "male",
  "bmi": 28,
  "children": 2,
  "smoker": "no",
  "region": "southeast"
}
```

**Response JSON**

```json
{
  "input": {...},
  "prediction": {
    "premium": 8885.76,
    "model_version": "model_v1.pkl",
    "confidence": 2664.42
  }
}
```

---

## 🧪 Testing & Coverage

Run tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=src --cov-report=term
```

Typical output:

```
Name                 Stmts   Miss  Cover
src/model/factory.py   5      0    100%
src/model/predictor.py 70     14   80%
TOTAL                  75     14   81%
```

---

## ⚡ Performance Benchmark

50 sequential prediction requests:

```
Total time: ~0.68 seconds
Average latency per request: ~13 ms
```

---

## 🔮 Future Enhancements

* Deploy using Docker & Kubernetes
* Add Elastic APM for monitoring
* Build React / Streamlit UI
* Introduce model versioning (v2, v3…)
* Add CI/CD (GitHub Actions)

---




