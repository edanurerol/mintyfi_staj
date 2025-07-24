import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 🔹 Veri yükleme
df = pd.read_csv("sms_clean.csv")
X = df["message"]
y = df["label"]

# 🔹 Eğitim/test bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Denenecek hiperparametreler
param_configs = [
    {"ngram_range": (1, 1), "max_features": 3000, "C": 0.5},
    {"ngram_range": (1, 2), "max_features": 5000, "C": 1.0},
    {"ngram_range": (1, 3), "max_features": 8000, "C": 2.0},
]

# 🔹 MLflow deney başlat
mlflow.set_experiment("SMS Spam Sınıflandırma")

best_acc = 0
best_run_id = ""

for config in param_configs:
    with mlflow.start_run() as run:
        # 🔸 Model pipeline'ı oluştur
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=config["ngram_range"],
                                      max_features=config["max_features"])),
            ("clf", LogisticRegression(C=config["C"], max_iter=1000))
        ])

        # 🔸 Eğit ve tahmin et
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # 🔸 MLflow logları
        mlflow.log_param("ngram_range", str(config["ngram_range"]))
        mlflow.log_param("max_features", config["max_features"])
        mlflow.log_param("C", config["C"])
        mlflow.log_metric("accuracy", acc)

        # 🔸 Modeli kaydet
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # 🔸 En iyi koşuyu güncelle
        if acc > best_acc:
            best_acc = acc
            best_run_id = run.info.run_id

# 🔸 En iyi run_id’yi dosyaya yaz
with open("best_run.txt", "w") as f:
    f.write(best_run_id)

print(f" En iyi run_id yazıldı: {best_run_id}")
