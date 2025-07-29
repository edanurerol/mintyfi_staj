import requests

url = "http://127.0.0.1:8000/predict"

texts = [
    "Tebrikler kazandınız!",
    "Merhaba, nasılsınız?",
    ""
]

for text in texts:
    response = requests.post(url, json={"text": text})
    print("Girdi:", text)
    print("Yanıt:", response.json())
    print("-" * 30)
