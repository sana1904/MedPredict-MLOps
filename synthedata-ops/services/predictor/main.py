import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Extra

app = FastAPI(title="SyntheData-Ops API", version="1.0")

# Note: We will generate these files inside Docker later!
MODEL_PATH = "artifacts/lightgbm_model.pkl"
META_PATH = "artifacts/pipeline_meta.pkl"

class PatientPayload(BaseModel, extra=Extra.allow):
    Age: float

@app.on_event("startup")
def load_artifacts():
    global model, meta
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Warning: Model not found. This is normal if you haven't run train.py yet!")
        return
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    print("🚀 Inference system operational.")

@app.post("/predict")
def predict_prognosis(payload: PatientPayload):
    try:
        # 1. Convert user input into a DataFrame
        input_data = pd.DataFrame([payload.dict()])

        # 2. Replicate the One-Hot Encoding from training
        input_encoded = pd.get_dummies(input_data, columns=meta.get("categorical_cols", []), drop_first=True)

        # 3. Align the new data perfectly with the model's expected features
        final_features = pd.DataFrame(columns=meta.get("feature_columns", []))
        final_features = pd.concat([final_features, input_encoded], axis=0).fillna(0)
        final_features = final_features[meta.get("feature_columns", [])]

        # 4. Make the prediction
        prediction = model.predict(final_features)[0]
        probabilities = model.predict_proba(final_features)[0]
        max_prob_idx = probabilities.argmax()

        return {
            "predicted_prognosis": str(prediction),
            "confidence_score": float(probabilities[max_prob_idx]),
            "diagnostic_distribution": dict(zip(meta.get("target_labels", []), probabilities.tolist()))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
