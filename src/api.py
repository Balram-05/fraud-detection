import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException, Query as FastAPI_Query
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(
    title="Production-Ready Fraud Detection Service",
    description="High-performance REST API for real-time credit card fraud inference.",
    version="1.0.0"
)

model = None

@app.on_event("startup")
def load_production_model():
    global model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, "models", "best_model.pkl")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Successfully loaded production model artifact.")

class TransactionData(BaseModel):
    features: List[float]

@app.post("/predict")
def predict_fraud(transaction: TransactionData):
    if len(transaction.features) != 29:
        raise HTTPException(status_code=400, detail="Expected 29 features.")
        
    input_array = np.array(transaction.features).reshape(1, -1)
    
    # DEBUG LOGGING: Let's see what the model thinks!
    prediction = int(model.predict(input_array)[0])
    raw_probas = model.predict_proba(input_array)[0]
    
    print("\n========= BACKEND INFERENCE DEBUG =========")
    print(f"Received Feature Vector Shape: {input_array.shape}")
    print(f"V14 (Index 13) Value: {transaction.features[13]:.4f}")
    print(f"V17 (Index 16) Value: {transaction.features[16]:.4f}")
    print(f"Raw Output Probabilities: [Class 0: {raw_probas[0]:.4f}, Class 1: {raw_probas[1]:.4f}]")
    print(f"Final Model Decision: {prediction}")
    print("===========================================\n")
    
    return {
        "fraud_prediction": prediction,
        "label": "Fraudulent" if prediction == 1 else "Non-Fraudulent"
    }

@app.get("/score")
def score_transaction(features: List[float] = FastAPI_Query(..., alias="f")):
    input_array = np.array(features).reshape(1, -1)
    probability = float(model.predict_proba(input_array)[0][1])
    return {"fraud_probability": round(probability, 4)}