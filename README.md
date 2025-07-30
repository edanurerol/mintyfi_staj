# Iris Baseline Model

Bu proje, Iris veri seti üzerinde basit bir makine öğrenmesi modeli geliştirmek için oluşturulmuştur. Amaç, temel bir sınıflandırma modeli ile başlangıç seviyesinde bir tahmin sistemi kurmaktır.

##  Proje Amacı

- Iris veri setini kullanarak çiçek türlerini sınıflandırmak
- Veri temizleme ve ön işleme adımlarını uygulamak
- Basit bir model (örneğin: Logistic Regression) ile doğruluk ölçümü yapmak

## Dosya Açıklamaları

| Dosya Adı            | Açıklama                                      |
|----------------------|-----------------------------------------------|
| `baseline_model.py`  | Model eğitimi ve test işlemleri               |
| `README.md`          | Proje açıklamaları ve kullanım talimatları    |

##  Kullanılan Kütüphaneler

- `pandas`
- `numpy`
- `scikit-learn`

##  İlk Sonuçlar

| Model               | Doğruluk (%) |
|---------------------|--------------|
| Logistic Regression | 96.7         |

> Not: Bu sonuç, eğitim/test bölünmesi sonrası elde edilen ilk sonuçtur. İleride iyileştirmeler yapılacaktır.

##  Nasıl Çalıştırılır?

```bash
pip install -r requirements.txt
python baseline_model.py
