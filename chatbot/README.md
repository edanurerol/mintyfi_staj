# 📖 Nutuk RAG Chatbot

Nutuk RAG Chatbot, **Retrieval-Augmented Generation (RAG)** yaklaşımını kullanarak **Nutuk’tan alıntılarla Türkçe soru-cevap** yapabilen bir uygulamadır.  
Backend tarafında **FastAPI**, frontend tarafında ise **Gradio** arayüzü kullanılmıştır.

---

## 🚀 Özellikler
- 📑 **Belge yükleme ve parçalama**: Nutuk metni chunk’lara ayrılır ve vektör veritabanına kaydedilir.  
- 🔍 **Arama + Geri Getirme**: Kullanıcının sorusuna uygun parçalar vektör benzerliğiyle bulunur.  
- 🧠 **LLM özetleme**: Bulunan parçalar OpenAI GPT modeliyle özetlenip kullanıcıya cevap döner.  
- 💬 **Web arayüzü**: Gradio tabanlı basit chatbot arayüzü.  
- ⚡ **REST API**: FastAPI ile `/ingest` ve `/chat` endpoint’leri.  

---

## 📂 Proje Yapısı
```
chatbot/
│
├── rag_app.py        # FastAPI backend
├── ui_app.py         # Gradio frontend
├── storage/          # Embedding veritabanı dosyaları
├── docs/             # Kaynak belgeler (ör. Nutuk.pdf)
├── .env              # Ortam değişkenleri (API key vs.)
├── requirements.txt  # Bağımlılıklar
└── README.md
```

---

## 🔧 Kurulum

1. **Repo’yu klonla** (veya proje dosyalarını indir):
   ```bash
   git clone https://github.com/kullanici/nutuk-rag-chatbot.git
   cd nutuk-rag-chatbot
   ```

2. **Sanal ortam oluştur**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   .\venv\Scripts\activate    # Windows PowerShell
   ```

3. **Gereksinimleri yükle**:
   ```bash
   pip install -r requirements.txt
   ```

4. **.env dosyası oluştur**:
   ```
   OPENAI_API_KEY=sk-XXXX
   CHUNK_MAX_CHARS=800
   CHUNK_OVERLAP=150
   EMBED_MODEL=intfloat/multilingual-e5-base
   ```

---

## ▶️ Çalıştırma

### 1) Belgeleri ingest et
```bash
curl -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d "{\"dir_path\":\"docs\"}"
```
Bu adım, `docs/` klasöründeki belgeleri parçalayarak `storage/` içine embedding olarak kaydeder.

### 2) Backend’i çalıştır
```bash
uvicorn rag_app:app --reload --port 8000
```
- API dokümantasyonu için: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3) Gradio arayüzünü çalıştır
```bash
python ui_app.py
```
- Tarayıcıda açılır: [http://127.0.0.1:7860](http://127.0.0.1:7860)

---

## 💡 Kullanım
- Sorularını **chatbox**’a yazabilirsin:
  - `Nutuk hangi yılları kapsar?`
  - `Nutuk'un yazarı kimdir?`
  - `Samsun’a çıkış hangi tarihte olmuştur?`

- API üzerinden test etmek için:
```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"query\":\"Nutuk hangi yılları kapsar?\"}"
```

---

## 📌 Notlar
- Daha doğru cevaplar için `intfloat/multilingual-e5-large` modeli önerilir.  
- `storage/` klasörünü silip tekrar `/ingest` çalıştırarak embedding veritabanını yenileyebilirsin.  
- Yanlış cevaplar gelirse `.env` içinde **OPENAI_API_KEY** tanımlı olduğuna emin ol.  

---

## 📜 Lisans
Bu proje eğitim amaçlıdır.  
