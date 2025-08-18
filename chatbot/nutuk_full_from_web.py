import os, io, re, sys, time
from typing import List
import requests
from pypdf import PdfReader

# --- Kaynak PDF URL adayları (bazıları cilt cilt, bazıları tek cilt) ---
URLS = [
    # Türk Tarih Kurumu – Nutuk (Cilt I) (resmi)
    "https://www.ttk.gov.tr/karekod/nutuk.pdf",  # I. Cilt (TTK)  

    # TBMM Açık Erişim – Osmanlıca (metin çıkarma zayıf olabilir)
    "https://acikerisim.tbmm.gov.tr/server/api/core/bitstreams/4b1f70ec-7b70-4e73-897c-e69c3366e8ec/content",  # 197603162.pdf  

    # Bazı okul sitelerinden alternatif cilt PDF'leri (erişilebilirlik değişebilir)
    "https://www.tedcorlu.k12.tr/wp-content/uploads/2021/11/Nutuk-Cilt-1-1919-1920_compressed-1.pdf",  # Cilt 1  
    "https://www.tedcorlu.k12.tr/wp-content/uploads/2021/11/Nutuk-Cilt-3_compressed.pdf",             # Cilt 3  

    # Çalışmazsa kullanıcı KTBY e-Kitap sayfasından manuel indirebilir (cilt seçerek)
    # https://ekitap.ktb.gov.tr/TR-273376/nutuk.html  
]

PDF_DIR = "docs_full_pdfs"
TXT_DIR = "docs_full_txt"

def safe_name(url: str) -> str:
    base = url.strip().split("/")[-1] or "nutuk.pdf"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return base

def download_pdf(url: str, timeout=60) -> str | None:
    os.makedirs(PDF_DIR, exist_ok=True)
    fn = os.path.join(PDF_DIR, safe_name(url))
    try:
        print(f"[DL] {url}")
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200 or "pdf" not in resp.headers.get("Content-Type","").lower():
            print(f"  -> Uyarı: PDF değil ya da erişilemedi (status={resp.status_code}). Atlaniyor.")
            return None
        with open(fn, "wb") as f:
            f.write(resp.content)
        size_kb = len(resp.content) // 1024
        print(f"  -> Kaydedildi: {fn} ({size_kb} KB)")
        return fn
    except Exception as e:
        print(f"  -> Hata: {e}. Atlaniyor.")
        return None

def extract_text(pdf_path: str) -> List[str]:
    """Basit metin çıkarıcı: her sayfayı ayrı txt’ye döker."""
    outfiles = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            n = len(reader.pages)
            if n == 0:
                print(f"  -> Uyarı: {pdf_path} boş ya da okunamadı.")
                return outfiles
            os.makedirs(TXT_DIR, exist_ok=True)
            for i in range(n):
                try:
                    text = reader.pages[i].extract_text() or ""
                    text = text.strip()
                    if not text:
                        # Tarama/Osmanlıca olabilir; OCR gerekir (bu script OCR yapmaz)
                        continue
                    out_fn = os.path.join(TXT_DIR, f"{os.path.basename(pdf_path)}_p{i+1:04d}.txt")
                    with open(out_fn, "w", encoding="utf-8") as out:
                        out.write(text)
                    outfiles.append(out_fn)
                except Exception as e:
                    print(f"    -> Sayfa {i+1} metin çikarma hatasi: {e}")
    except Exception as e:
        print(f"  -> PDF okuma hatasi: {pdf_path} | {e}")
    print(f"  -> {pdf_path} için {len(outfiles)} sayfa metne dönüştü.")
    return outfiles

def main():
    print("Nutuk tam metin PDF indiriliyor & metin çıkarılıyor...")
    downloaded = []
    for url in URLS:
        fn = download_pdf(url)
        if fn:
            downloaded.append(fn)

    if not downloaded:
        print("Hiç PDF indirilemedi. Gerekirse ekitap sayfasından ciltleri indirip 'docs_full_pdfs' klasörüne koyun:\n  https://ekitap.ktb.gov.tr/TR-273376/nutuk.html")
        sys.exit(1)

    total_txt = 0
    for pdf in downloaded:
        outfiles = extract_text(pdf)
        total_txt += len(outfiles)

    print(f"[OK] Toplam {len(downloaded)} PDF işlendi, {total_txt} TXT üretildi: {TXT_DIR}")

if __name__ == "__main__":
    main()

