import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

import pandas as pd
import numpy as np
train_df = pd.read_csv(
    r"dataset\train.csv"
)

test_df = pd.read_csv(
    r"dataset\test.csv"
)

print("Training Data Shape:", train_df.shape)
print("Testing Data Shape:", test_df.shape)

print("\nColumns:")
print(train_df.columns)

print("\nFirst 5 Training Records:")
print(train_df.head())

print("\nMissing Values in Training Data:")
print(train_df.isnull().sum())

print("\nMissing Values in Testing Data:")
print(test_df.isnull().sum())

print("\nCategory Distribution:")
print(train_df["Class Index"].value_counts())
category_names = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

print("\nCategory Names:")
for class_index, count in train_df["Class Index"].value_counts().sort_index().items():
    print(class_index, "->", category_names[class_index], ":", count)


train_df["Text"] = train_df["Title"] + " " + train_df["Description"]
test_df["Text"] = test_df["Title"] + " " + test_df["Description"]

print("\nCombined Text:")
print(train_df[["Title", "Description", "Text"]].head())


import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]
    return " ".join(words)
print("\nApplying preprocessing to training data...")

train_df["Clean_Text"] = train_df["Text"].apply(preprocess_text)

print("Training preprocessing completed.")

print("\nApplying preprocessing to testing data...")

test_df["Clean_Text"] = test_df["Text"].apply(preprocess_text)

print("Testing preprocessing completed.")



print("\nProcessed Training Data:")
print(train_df[["Class Index", "Clean_Text"]].head())

X_train = train_df["Clean_Text"]
y_train = train_df["Class Index"]

X_test = test_df["Clean_Text"]
y_test = test_df["Class Index"]

print("\nTraining Text Shape:", X_train.shape)
print("Training Target Shape:", y_train.shape)

print("Testing Text Shape:", X_test.shape)
print("Testing Target Shape:", y_test.shape)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

print("\nCreating TF-IDF features...")

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF Training Shape:", X_train_tfidf.shape)
print("TF-IDF Testing Shape:", X_test_tfidf.shape)

from sklearn.linear_model import LogisticRegression

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

joblib.dump(model, BASE_DIR / "model.pkl")
joblib.dump(tfidf, BASE_DIR / "tfidf.pkl")

print("Model and TF-IDF saved successfully!")
print("Model training completed.")

print("\nMaking predictions...")

y_pred = model.predict(X_test_tfidf)

print("Prediction completed.")

print("\nFirst 20 Predictions:")
print(y_pred[:20])

print("\nActual Labels:")
print(y_test.iloc[:20].values)



from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "World",
            "Sports",
            "Business",
            "Sci/Tech"
        ]
    )
)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

class_labels = ["World", "Sports", "Business", "Sci/Tech"]

cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_labels
)

disp.plot(cmap="Blues", values_format="d")
plt.title("News Category Classification - Confusion Matrix")
plt.tight_layout()
plt.show()


def predict_news(title, description):
    text = title + " " + description

    cleaned_text = preprocess_text(text)

    text_vector = tfidf.transform([cleaned_text])

    prediction = model.predict(text_vector)[0]

    return category_names[prediction]


title = input("Enter news title: ")
description = input("Enter news description: ")

predicted_category = predict_news(title, description)

print("\nPredicted News Category:", predicted_category)

import joblib

joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")

print("Model and TF-IDF saved successfully!")