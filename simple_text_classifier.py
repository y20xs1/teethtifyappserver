import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

LABELS = ["Extraction", "Filling", "Root Canal", "Cleaning", "Initial Checkup", "Other"]

train_texts = [
    # Extraction
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
    # Filling
    "I have a cavity in my tooth.",
    "My tooth hurts when I eat sweets.",
    "I need a filling for my tooth.",
    "There is a small hole in my tooth.",
    "filling",
    "tooth filling",
    "fill my tooth",
    "cavity filling",
    "I need a dental filling.",
    # Root Canal
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
    # Cleaning
    "I want a dental cleaning.",
    "I need my teeth cleaned.",
    "cleaning",
    "teeth cleaning",
    "professional cleaning",
    "scaling",
    "polishing",
    "remove tartar",
    "I want to whiten my teeth.",
    # Initial Checkup
    "I want a dental checkup.",
    "I need a routine dental exam.",
    "checkup",
    "dental checkup",
    "teeth checkup",
    "I want to get my teeth checked.",
    "consultation",
    "I want to see a dentist.",
    # Other
    "hello",
    "weather is nice",
    "I like pizza",
    "random text",
    "football",
    "I am going to the park",
    "What is the time?",
    "This is a random sentence.",
    "I love programming.",
    "The sky is blue.",
    "pizzza",
    "rootz",
    "canalization",
    "unrelated",
    "nonsense",
    "asdfghjkl",
    "qwerty",
    "blabla",
    "test",
    "music",
    "movie",
    "shopping",
    "I want to travel.",
    "I am hungry.",
    "I want to sleep.",
    "dog",
    "cat"
]
train_labels = (
    [0]*10 +  # Extraction
    [1]*9 +   # Filling
    [2]*10 +  # Root Canal
    [3]*9 +   # Cleaning
    [4]*8 +   # Initial Checkup
    [5]*20    # Other
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
