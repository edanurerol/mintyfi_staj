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
 tensor([[ 0.2535,  0.1042,  0.6050],
        [-1.0452, -0.3174, -1.5578],
        [ 0.2089, -0.1379, -1.0009]], device='cuda:0')
Tensör b:
 tensor([[ 0.3910, -0.4702, -0.1320],
        [-1.7294,  0.4639, -0.6649],
        [-1.2048, -1.8568, -0.0343]], device='cuda:0')
Toplam:
 tensor([[ 0.6445, -0.3660,  0.4730],
        [-2.7746,  0.1465, -2.2226],
        [-0.9959, -1.9947, -1.0352]], device='cuda:0')
Matris çarpımı:
 tensor([[-0.8100, -1.1942, -0.1235],
        [ 2.0170,  3.2366,  0.4025],
        [ 1.5261,  1.6963,  0.0985]], device='cuda:0')
ReLU (element-wise):
 tensor([[0.2535, 0.1042, 0.6050],
        [0.0000, 0.0000, 0.0000],
        [0.2089, 0.0000, 0.0000]], device='cuda:0')
Yeniden şekillendirilmiş (1D):
 tensor([ 0.2535,  0.1042,  0.6050, -1.0452, -0.3174, -1.5578,  0.2089, -0.1379,
        -1.0009], device='cuda:0')
x.grad:
 tensor([[0.2222, 0.2222, 0.2222],
        [0.2222, 0.2222, 0.2222],
        [0.2222, 0.2222, 0.2222]], device='cuda:0')
