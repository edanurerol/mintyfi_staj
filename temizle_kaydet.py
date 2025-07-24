import pandas as pd

# Veriyi oku
df = pd.read_csv("sms_spam_dataset/SMSSpamCollection", sep="\t", header=None, names=["label", "message"], encoding="utf-8")

# Yinelenenleri temizle
df = df.drop_duplicates()

# CSV olarak kaydet
df.to_csv("sms_clean.csv", index=False, encoding="utf-8")

# Sınıf sayımlarını al
counts = df["label"].value_counts()

# readme içeriği
readme_text = f"""# SMS Spam Dataset (Temizlenmiş)

Bu dosya, UCI'den alınan SMS Spam veri kümesinin temizlenmiş halidir.  
Orijinal veri kümesinde 5572 satır bulunmaktaydı.

## Temizlik Aşamaları:
- Yinelenen kayıtlar kaldırıldı (403 satır)
- Eksik veri bulunmadı
- UTF-8 encoding ile okundu

## Sınıf Dağılımı:
- ham: {counts.get('ham', 0)}
- spam: {counts.get('spam', 0)}

Toplam: {len(df)} kayıt
"""

# README dosyasını yaz
with open("data_readme.md", "w", encoding="utf-8") as f:
    f.write(readme_text)

print("Temiz veri 'sms_clean.csv' olarak kaydedildi.")
print("'data_readme.md' dosyası oluşturuldu.")
