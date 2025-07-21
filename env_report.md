# Ortam Raporu (env_report.md)

## ✅ Sistem Bilgileri

- **Python sürümü:** 3.10.0
- **PyTorch sürümü:** 2.5.1+cu121
- **CUDA kullanılabilir mi?:** True
- **CUDA sürümü:** 12.1
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU
- **NumPy sürümü:** 2.2.6
- **OpenCV sürümü:** 4.12.0
- **İşletim Sistemi:** Windows 10 (10.0.19045)

---

## ⚠️ Karşılaşılan Sorunlar ve Çözümler

### 1. `python --version` komutu çalışmıyordu
➡️ **Çözüm:** Python yüklendi ve PATH’e eklendi

### 2. `nvcc --version` komutu tanınmadı
➡️ **Çözüm:** CUDA Toolkit kuruldu ve `bin` klasörü PATH'e eklendi

### 3. PyTorch, GPU'yu tanımadı (CPU sürümü kuruldu)
➡️ **Çözüm:** Python 3.10 ile yeni bir `venv` oluşturuldu ve `torch+cu121` kuruldu

### 4. `git push` sırasında authentication hatası yaşandı
➡️ **Çözüm:** GitHub için kişisel erişim token (PAT) oluşturuldu ve push başarıyla yapıldı

---

Bu ortam raporu, `versions.py` çıktısı ve kurulum sürecini özetlemektedir.
