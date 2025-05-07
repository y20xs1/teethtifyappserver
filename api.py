from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

# تحميل النموذج والفيكتورايزر والليبلات
MODEL_DIR = os.path.join(os.path.dirname(__file__), "simple_model")
clf = joblib.load(os.path.join(MODEL_DIR, 'model.joblib'))
vectorizer = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.joblib'))
LABELS = joblib.load(os.path.join(MODEL_DIR, 'labels.joblib'))

app = FastAPI()

# Add CORS middleware
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
    pred = clf.predict(X_test)[0]
    return {"prediction": LABELS[pred]}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

# This is for local development only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
