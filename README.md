# Transformers Pipeline Deneyi

Bu projede Hugging Face `transformers` kütüphanesi kullanılarak bir **text classification (metin sınıflandırma)** işlemi gerçekleştirilmiştir.  
Model, `pipeline("text-classification")` arayüzüyle çağrılmış ve varsayılan olarak  
[`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) modeli kullanılmıştır.

---

##  Dosyalar

- `hf_results.md` → 3 metin örneği üzerinde sınıflandırma sonuçları ve çalışma süreleri  
- `requirements.txt` → Gerekli Python kütüphaneleri (uyumlu sürümlerle birlikte)  
- `README.md` → Bu dokümantasyon

---

##  Kullanılan Girişler

| Metin                                                   | Tahmin Edilen Etiket |
|----------------------------------------------------------|-----------------------|
| Bu ürün harikaydı, çok memnun kaldım.                    | POSITIVE              |
| Film gerçekten zaman kaybıydı.                           | NEGATIVE              |
| Yemekler vasattı ama ortam güzeldi.                      | POSITIVE              |

---

## ⚙Sistem Bilgisi

- Cihaz: `CUDA` (GPU destekli)
- PyTorch: `2.5.1+cu121`
- Transformers: `4.54.1`

---

##  Çalıştırmak için

Aynı ortamı kurmak için:

```bash
pip install -r requirements.txt
