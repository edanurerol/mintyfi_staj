#  Gün 3 - Spam Sınıflandırma Sonuçları

##  Kullanılan Yöntemler
- Train/validation split: Stratified, 80/20 oranında
- Vektörleştirme: TF-IDF
- Modeller: Logistic Regression, Multinomial Naive Bayes, Random Forest

---

##  Performans Karşılaştırma Tablosu

| Model                    | Accuracy | Precision | Recall | F1 Score |
|--------------------------|----------|-----------|--------|----------|
| Logistic Regression      | 0.9731   | 1.000     | 0.7987 | 0.8881   |
| Multinomial Naive Bayes | 0.9605   | 1.000     | 0.7047 | 0.8268   |
| Random Forest            | 0.9722   | 1.000     | 0.7919 | 0.8839   |

---

##  Kısa Yorum

- **Tüm modellerde precision 1.0** çıktı → Spam olarak işaretlenen mesajlar gerçekten spam.
- **Recall değerleri** ise farklılık gösteriyor. En az hatayla çalışan model Logistic Regression oldu.
- MultinomialNB, çok hızlı çalışmasına rağmen spam’lerin bazılarını kaçırdı.
- Random Forest da güçlü ama daha yavaş ve LogisticRegression ile neredeyse eşdeğer.

---

##  En İyi Model

> **Logistic Regression**  
> Parametreler: `max_iter=1000`  
> Nedeni: En yüksek F1 skoru ve en dengeli başarı.

---

##  Not

Veri seti: [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  
Veri sayısı: 5574 mesaj  
Etiket dağılımı: ~4825 ham / ~747 spam

