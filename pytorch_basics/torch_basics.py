import torch

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", device)

# 2B rastgele tensör oluşturma
a = torch.randn(3, 3, device=device)
b = torch.randn(3, 3, device=device)
print("Tensör a:\n", a)
print("Tensör b:\n", b)

# Toplama
sum_tensor = a + b
print("Toplam:\n", sum_tensor)

# Matris çarpımı
matmul_tensor = a @ b    # veya torch.matmul(a, b)
print("Matris çarpımı:\n", matmul_tensor)

# Element-wise fonksiyon (örn. ReLU)
relu_tensor = torch.relu(a)
print("ReLU (element-wise):\n", relu_tensor)

# Şekil değiştirme (reshape)
reshaped = a.view(9)
print("Yeniden şekillendirilmiş (1D):\n", reshaped)

# Basit ileri-geri hesaplama (otomatik türev)
x = torch.randn(3, 3, requires_grad=True, device=device)
y = x * 2 + 1
z = y.mean()
z.backward()
print("x.grad:\n", x.grad)
