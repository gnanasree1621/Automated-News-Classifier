import sys
import db
import joblib
from pathlib import Path
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def test_all():
    print("--- 1. Testing MongoDB Connection Diagnostic ---")
    is_conn, msg = db.check_connection()
    print(f"Connection result: {is_conn}, message: {msg}")
    assert is_conn, f"MongoDB not reachable: {msg}"

    print("--- 2. Testing clearing existing predictions ---")
    clear_ok, clear_cnt = db.clear_predictions()
    print(f"Clear ok: {clear_ok}, deleted count: {clear_cnt}")

    print("--- 3. Testing KPI metrics when empty ---")
    kpis = db.get_kpi_metrics()
    print(f"KPIs empty: {kpis}")
    assert kpis["total_predictions"] == 0
    assert kpis["latest_category"] == "—"
    assert kpis["avg_confidence"] == 0.0

    print("--- 4. Testing ML Model Loading and Inference ---")
    base_dir = Path(__file__).resolve().parent
    model = joblib.load(base_dir / "model.pkl")
    tfidf = joblib.load(base_dir / "tfidf.pkl")

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        words = text.split()
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)

    categories = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
    
    sample_title = "Tech company launches futuristic AI processor chip"
    sample_desc = "A major semiconductor vendor announced its next generation artificial intelligence computing chip designed for data centers."
    
    cleaned = preprocess_text(f"{sample_title} {sample_desc}")
    vec = tfidf.transform([cleaned])
    pred = int(model.predict(vec)[0])
    category = categories.get(pred, "Unknown")
    probs = model.predict_proba(vec)[0]
    prob_map = {categories.get(int(cid), str(cid)): float(p) for cid, p in zip(model.classes_, probs)}
    confidence = round(prob_map.get(category, 0.0) * 100, 2)

    print(f"Predicted category: {category}, Confidence: {confidence}%, Probabilities: {prob_map}")

    print("--- 5. Testing saving prediction to MongoDB ---")
    saved_ok, doc = db.save_prediction(
        headline=sample_title,
        description=sample_desc,
        category=category,
        confidence=confidence,
        probabilities=prob_map
    )
    print(f"Save ok: {saved_ok}, Doc ID: {doc.get('_id')}")
    assert saved_ok, f"Save failed: {doc}"

    print("--- 6. Testing retrieving predictions from MongoDB ---")
    get_ok, docs = db.get_predictions()
    print(f"Get ok: {get_ok}, Total retrieved: {len(docs)}")
    assert get_ok
    assert len(docs) == 1
    assert docs[0]["headline"] == sample_title
    assert docs[0]["category"] == category

    print("--- 7. Testing KPI metrics with data ---")
    kpis = db.get_kpi_metrics()
    print(f"KPIs with data: {kpis}")
    assert kpis["total_predictions"] == 1
    assert kpis["latest_category"] == category
    assert abs(kpis["avg_confidence"] - confidence) < 0.2

    print("--- 8. Testing Clear Prediction History ---")
    clear_ok, del_cnt = db.clear_predictions()
    print(f"Clear ok: {clear_ok}, Deleted count: {del_cnt}")
    assert clear_ok
    assert del_cnt == 1

    get_ok, docs_after = db.get_predictions()
    assert len(docs_after) == 0

    print("\n[SUCCESS] ALL WORKFLOW TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_all()
