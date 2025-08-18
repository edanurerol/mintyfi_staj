# ask.py — RAG botuna komut satırından soru soran basit istemci
import argparse, os, sys, requests

def ingest(dir_path: str, base_url: str):
    r = requests.post(f"{base_url}/ingest", json={"dir_path": dir_path}, timeout=60)
    r.raise_for_status()
    print("Ingest:", r.json())

def chat(query: str, top_k: int, base_url: str, temperature: float | None):
    payload = {"query": query, "top_k": top_k}
    if temperature is not None:
        payload["temperature"] = temperature
    r = requests.post(f"{base_url}/chat", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    print("\nYanıt:\n" + data.get("answer", ""))
    cits = data.get("citations", [])
    if cits:
        print("\nKaynaklar:")
        for c in cits:
            try:
                print(f"- {c['source']} [parça {c['chunk_id']}] (skor {float(c['score']):.3f})")
            except Exception:
                print(f"- {c}")
    return data

def main():
    ap = argparse.ArgumentParser(description="RAG botuna soru sor")
    ap.add_argument("query", nargs="?", help="Sorunuz (tırnak içinde)")
    ap.add_argument("--top-k", type=int, default=4, help="Getirilecek parça sayısı")
    ap.add_argument("--host", default=os.getenv("RAG_HOST", "127.0.0.1"))
    ap.add_argument("--port", default=os.getenv("RAG_PORT", "8000"))
    ap.add_argument("--ingest", help="Soru sormadan önce şu klasörü indeksle (örn. docs)")
    ap.add_argument("--temp", type=float, default=None, help="LLM sıcaklığı (örn. 0.2)")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Sağlık kontrolü
    try:
        requests.get(f"{base_url}/", timeout=5).raise_for_status()
    except Exception:
        print(f"Sunucuya ulaşılamıyor: {base_url}\n"
              f"Önce çalıştırın:\n  python -m uvicorn rag_app:app --reload --port {args.port}")
        sys.exit(1)

    # Opsiyonel ingest
    if args.ingest:
        try:
            ingest(args.ingest, base_url)
        except requests.HTTPError as he:
            print("Ingest hatası:", he.response.text); sys.exit(2)

    if not args.query:
        print('Soru metni vermediniz. Örnek:\n  python ask.py "İade politikası nedir?"')
        sys.exit(3)

    try:
        chat(args.query, args.top_k, base_url, args.temp)
    except requests.HTTPError as he:
        print("HTTP hata:", he.response.text); sys.exit(4)
    except Exception as e:
        print("Hata:", e); sys.exit(5)

if __name__ == "__main__":
    main()
