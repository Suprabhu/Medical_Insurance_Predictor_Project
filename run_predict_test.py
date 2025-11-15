from src.model.factory import get_predictor

p = get_predictor()

print("Loaded features count:", len(p.feature_order))
print("Feature sample:", p.feature_order[:10])

print("Prediction example:", p.predict({
    "age": 40,
    "sex": "male",
    "bmi": 27,
    "children": 1,
    "smoker": "no",
    "region": "southeast"
}))
