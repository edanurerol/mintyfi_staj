# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target

# Eğitim ve test setine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Basit bir model eğit
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Başarıyı yazdır
acc = accuracy_score(y_test, y_pred)
print(f"Baseline model doğruluk skoru: {acc:.2f}")
