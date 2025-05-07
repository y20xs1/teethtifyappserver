from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "simple_model")
clf = joblib.load(os.path.join(MODEL_DIR, 'model.joblib'))
vectorizer = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.joblib'))
LABELS = joblib.load(os.path.join(MODEL_DIR, 'labels.joblib'))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input_data: TextInput):
    X_test = vectorizer.transform([input_data.text])
    proba = clf.predict_proba(X_test)[0]
    pred = proba.argmax()
    confidence = float(proba[pred])
    label = LABELS[pred]
    # إذا الثقة أقل من 0.5 اعتبرها Other
    if confidence < 0.5:
        label = "Other"
    return {"prediction": label, "confidence": confidence}
