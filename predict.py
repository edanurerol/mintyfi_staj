import pickle
import sys

def load_model():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model, vectorizer

def predict_message(message):
    model, vectorizer = load_model()
    X = vectorizer.transform([message])
    return model.predict(X)[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python predict.py \"mesaj metni\"")
        sys.exit(1)
    message = sys.argv[1]
    prediction = predict_message(message)
    print(f"Tahmin: {prediction}")
