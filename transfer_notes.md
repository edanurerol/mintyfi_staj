# Transfer Learning Notları (04.08.2025)

##  Görüntü Modeli: ResNet18
- Model: `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`
- Önceden ImageNet üzerinde eğitilmiş ağırlıklar kullanıldı.
- Görsel: `picasso.jpg`, boyutlandırma: `224x224`
- Embedding çıkarımı için `model.children()[:-1]` ile `fc` katmanı hariç tutuldu.
- Elde edilen embedding boyutu: **[1, 512, 1, 1]**

##  Gözlemler
- ResNet18'in son `avgpool` katmanından elde edilen özellik temsili 512 boyutlu.
- Bu çıktı sınıflandırma yerine başka görevlerde (örneğin benzerlik, transfer öğrenme) kullanılabilir.
- Görüntüden çıkarılan bu embedding sabit uzunluklu vektör olarak kullanılabilir.

##  Sonraki Adımlar
- Metin tarafında benzer bir embedding çıkarımı için `distilbert-base-uncased` kullanılacak.
