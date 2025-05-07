import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

LABELS = ["Extraction", "Filling", "Root Canal", "Cleaning", "Initial Checkup", "Other"]

train_texts = [
    # Extraction (10)
    "My tooth is loose and needs to be pulled.",
    "I need to have my tooth removed.",
    "My wisdom tooth is causing problems and should be extracted.",
    "My tooth is broken and cannot be saved.",
    "My tooth is fractured below the gum line.",
    "extraction",
    "remove tooth",
    "pull my tooth",
    "tooth extraction",
    "tooth pulled",
    # Filling (10)
    "I have a cavity in my tooth.",
    "My tooth hurts when I eat sweets.",
    "I need a filling for my tooth.",
    "There is a small hole in my tooth.",
    "filling",
    "tooth filling",
    "fill my tooth",
    "cavity filling",
    "I need a dental filling.",
    "tooth has a cavity",
    # Root Canal (10)
    "My tooth pain is deep and throbbing, especially at night.",
    "I have pain radiating from my tooth to my ear.",
    "root canal",
    "canal treatment",
    "I need a root canal.",
    "tooth root canal",
    "root canal therapy",
    "canal",
    "root",
    "nerve treatment",
    # Cleaning (10)
    "I want a dental cleaning.",
    "I need my teeth cleaned.",
    "cleaning",
    "teeth cleaning",
    "professional cleaning",
    "scaling",
    "polishing",
    "remove tartar",
    "I want to whiten my teeth.",
    "book a cleaning appointment",
    # Initial Checkup (10)
    "I want a dental checkup.",
    "I need a routine dental exam.",
    "checkup",
    "dental checkup",
    "teeth checkup",
    "I want to get my teeth checked.",
    "consultation",
    "I want to see a dentist.",
    "book a checkup",
    "schedule a dental exam",
    # Other (10)
    "hello",
    "weather is nice",
    "I like pizza",
    "random text",
    "football",
    "I am going to the park",
    "What is the time?",
    "This is a random sentence.",
    "I love programming.",
    "The sky is blue."
]
train_labels = (
    [0]*10 +  # Extraction
    [1]*10 +  # Filling
    [2]*10 +  # Root Canal
    [3]*10 +  # Cleaning
    [4]*10 +  # Initial Checkup
    [5]*10    # Other
)

vectorizer = TfidfVectorizer(lowercase=True)
X = vectorizer.fit_transform(train_texts)
clf = LogisticRegression(max_iter=1000)
clf.fit(X, train_labels)

os.makedirs('simple_model', exist_ok=True)
joblib.dump(clf, 'simple_model/model.joblib')
joblib.dump(vectorizer, 'simple_model/vectorizer.joblib')
joblib.dump(LABELS, 'simple_model/labels.joblib')

def predict(text):
    X_test = vectorizer.transform([text])
    proba = clf.predict_proba(X_test)[0]
    pred = np.argmax(proba)
    confidence = proba[pred]
    # إذا الثقة أقل من 0.5 اعتبرها Other
    if confidence < 0.5:
        return "Other", float(confidence)
    return LABELS[pred], float(confidence)

# اختبر النموذج
if __name__ == '__main__':
    for test in [
        "I want a dental cleaning",
        "My tooth is loose and needs to be pulled.",
        "hello",
        "I have a cavity in my tooth.",
        "My tooth pain is deep and throbbing, especially at night.",
        "I want a dental checkup.",
        "I love programming.",
        "root",
        "canal",
        "pizzza",
        "nonsense"
    ]:
        print(f"{test} => {predict(test)}")
