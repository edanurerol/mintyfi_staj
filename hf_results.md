# Hugging Face Transformers Test Sonuçları

## pipeline("text-classification")

| Girdi                                                    | Etiket    | Süre (sn) |
|----------------------------------------------------------|-----------|-----------|
| Bu ürün harikaydı, çok memnun kaldım.                    | POSITIVE  | 0.421     |
| Film gerçekten zaman kaybıydı.                           | NEGATIVE  | 0.436     |
| Yemekler vasattı ama ortam güzeldi.                      | POSITIVE  | 0.401     |

Kullanılan model: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`  
Cihaz: CUDA (`cuda:0`)
