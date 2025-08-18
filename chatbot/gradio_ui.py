# ui.py — Nutuk RAG Chatbot (Gradio arayüzü)
import requests
import gradio as gr

API_BASE = "http://127.0.0.1:8000"

def ask_api(question: str):
    if not question.strip():
        return "Bir soru yazın."
    try:
        r = requests.post(f"{API_BASE}/chat", json={"query": question, "top_k": 6}, timeout=90)
        r.raise_for_status()
        data = r.json()
        return data.get("answer", "Cevap alınamadı.")
    except Exception as e:
        return f"Hata: {e}"

with gr.Blocks(title="Nutuk RAG Chatbot") as demo:
    gr.Markdown("## Nutuk RAG Chatbot\nNutuk'tan alıntılarla Türkçe soru-cevap.")
    chatbot = gr.Chatbot(height=420)
    msg = gr.Textbox(placeholder="Bir soru yazın ve Enter'a basın…", label=None)
    clear = gr.Button("Temizle")

    def respond(history, message):
        answer = ask_api(message)
        history = (history or []) + [(message, answer)]
        return "", history

    msg.submit(respond, [chatbot, msg], [msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    # Gradio'yu 7860'da aç
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)
