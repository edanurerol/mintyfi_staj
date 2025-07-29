# test_main.py
from fastapi.testclient import TestClient
from main import app, load_model

client = TestClient(app)
load_model()  # MODELLERİ BURADA YÜKLÜYORUZ

def test_predict_valid_text():
    response = client.post("/predict", json={"text": "Kazandınız! Hemen tıklayın."})
    print(">>>", response.text)
    assert response.status_code == 200
    assert "class" in response.json()
    assert "probability" in response.json()

def test_predict_empty_text():
    response = client.post("/predict", json={"text": "   "})
    print(">>>", response.text)
    assert response.status_code == 400
    assert response.json()["detail"] == "Boş metin gönderilemez."

def test_predict_invalid_payload():
    response = client.post("/predict", json={})
    print(">>>", response.text)
    assert response.status_code == 422  # Validation Error
