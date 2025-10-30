from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import os

app = FastAPI(title="Iris Prediction API")

MODEL_PATH = "model.joblib"

# Step 1: Train model if not found
def train_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump((model, iris.target_names), MODEL_PATH)
    print("✅ Model trained and saved to", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    train_model()

# Step 2: Load the model
model, target_names = joblib.load(MODEL_PATH)

# Step 3: Define input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Step 4: Endpoints
@app.get("/")
def home():
    return {"message": "Welcome to the Iris Prediction API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(iris: IrisInput):
    data = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])
    preds = model.predict(data)
    class_name = target_names[preds[0]]
    return {"prediction": class_name}

