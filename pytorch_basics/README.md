# PyTorch Temelleri - torch_basics.py

Bu proje, PyTorch kütüphanesini kullanarak tensörlerle yapılan temel işlemleri içeren bir örnek çalışmadır.

##  İçerik

`torch_basics.py` dosyasında aşağıdaki işlemler gösterilmektedir:

-  Tensör oluşturma (`torch.randn`)
-  Cihaz seçimi (CPU / GPU desteği)
-  Tensör toplama
-  Matris çarpımı (`torch.matmul`)
-  Element-wise aktivasyon fonksiyonu (ReLU)
-  Tensör şekil değiştirme (`view`)
-  Otomatik türev hesaplama (`autograd` ile backward)

##  Kullanım

```bash
python torch_basics.py

##   Gereksinimler

-  Python 3.7+
-  PyTorch

Kurulum:

bash
pip install torch

##   Örnek Çıktı

Kullanılan cihaz: cuda
Tensör a:
 tensor([[ 1.24, -0.87, 0.34],
         [ 0.12,  0.45, -1.28]], requires_grad=True)
Tensör b:
 tensor([[ 0.33,  0.78, -0.11],
         [ 1.02, -0.14,  0.65]])