import numpy as np

def predict_text(text: str, model, vectorizer) -> dict:
    features = vectorizer.transform([text])
    pred_class = model.predict(features)[0]
    proba = np.max(model.predict_proba(features))
    return {
        "class": pred_class,
        "probability": round(float(proba), 4)
    }
