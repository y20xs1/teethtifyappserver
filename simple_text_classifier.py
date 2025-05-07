import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

LABELS = ["Extraction", "Filling", "Root Canal", "Cleaning", "Initial Checkup", "Other"]

# أمثلة كثيرة وواضحة لكل فئة
train_texts = [
    # Extraction
    "My tooth is loose and needs to be pulled.",
    "I need to have my tooth removed.",
    "My wisdom tooth is causing problems and should be extracted.",
    "My tooth is broken and cannot be saved.",
    "I have a tooth that must be taken out.",
    "My dentist said I need a tooth extraction.",
    "My tooth is fractured below the gum line.",
    "My tooth is beyond repair and needs removal.",
    "My tooth is severely decayed and must be pulled.",
    "My tooth is making my jaw swell and needs to be removed.",
    "My tooth is causing my cheek to swell and needs extraction.",
    "My tooth is infected and the dentist recommended extraction.",
    "My tooth is hurting and wobbly, needs to be pulled.",
    "My tooth is broken to the gum line and can't be fixed.",
    "My tooth is loose after an accident.",
    "My tooth is causing pain when I bite and needs to be removed.",
    "My tooth is black and painful, dentist said to extract it.",
    "My tooth is making it hard to eat and needs extraction.",
    "My tooth is causing headaches and must be pulled.",
    "My tooth is decayed to the root and needs to be removed.",
    # Filling
    "I have a cavity in my tooth.",
    "My tooth hurts when I eat sweets.",
    "I need a filling for my tooth.",
    "There is a small hole in my tooth.",
    "My tooth is sensitive to cold drinks.",
    "I have a decayed tooth that needs a filling.",
    "My tooth is chipped and needs a filling.",
    "My tooth hurts when I eat chocolate.",
    "I have a tooth with a cavity that needs to be filled.",
    "My tooth is hurting and has a small hole.",
    # Root Canal
    "My tooth pain is deep and throbbing, especially at night.",
    "I have pain radiating from my tooth to my ear.",
    "My tooth is sensitive to hot and cold and the pain is severe.",
    "I have a severe toothache that keeps me awake.",
    "My tooth hurts badly and the pain is deep.",
    "My tooth is throbbing and keeps me awake.",
    "I have a tooth that hurts when I bite down and the pain lingers.",
    "My tooth is aching and the pain is constant.",
    "I have a tooth that is sensitive to pressure and the pain is sharp.",
    "My tooth is hurting and the pain is intense and deep.",
    # Cleaning
    "I want a dental cleaning.",
    "I need my teeth cleaned.",
    "I want to remove tartar from my teeth.",
    "I want a professional cleaning.",
    "I want to whiten my teeth.",
    "cleaning",
    "I want to polish my teeth.",
    "I want to get rid of stains on my teeth.",
    "I want a scaling and polishing.",
    "I want to book a cleaning appointment.",
    # Initial Checkup
    "I want a dental checkup.",
    "I need a routine dental exam.",
    "I want to see a dentist for a checkup.",
    "I want to make sure my teeth are healthy.",
    "I want a regular dental checkup.",
    "I want to schedule a dental exam.",
    "I want to get my teeth checked.",
    "I want to have a dental checkup.",
    "I want to book a dental checkup.",
    "I want to get a dental exam.",
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
    "The sky is blue."
]
train_labels = (
    [0]*20 +  # Extraction
    [1]*10 +  # Filling
    [2]*10 +  # Root Canal
    [3]*10 +  # Cleaning
    [4]*10 +  # Initial Checkup
    [5]*10    # Other
)

# 1. تحويل النصوص إلى ميزات رقمية
vectorizer = TfidfVectorizer(lowercase=True)
X = vectorizer.fit_transform(train_texts)

# 2. تدريب النموذج
clf = LogisticRegression(max_iter=1000)
clf.fit(X, train_labels)

# 3. حفظ النموذج والفيكتورايزر
os.makedirs('simple_model', exist_ok=True)
joblib.dump(clf, 'simple_model/model.joblib')
joblib.dump(vectorizer, 'simple_model/vectorizer.joblib')
joblib.dump(LABELS, 'simple_model/labels.joblib')

# 4. دالة التنبؤ
def predict(text):
    X_test = vectorizer.transform([text])
    pred = clf.predict(X_test)[0]
    return LABELS[pred]

# أمثلة اختبار
print(predict("I want a dental cleaning"))         # Cleaning
print(predict("My tooth is loose and needs to be pulled."))  # Extraction
print(predict("hello"))                            # Other
print(predict("I have a cavity in my tooth."))     # Filling
print(predict("My tooth pain is deep and throbbing, especially at night.")) # Root Canal
print(predict("I want a dental checkup."))         # Initial Checkup
print(predict("I love programming."))              # Other 