# Dil Modeli Fine-Tuning Rehberi (SFT, LoRA, QLoRA)

Bu rehber, bir dil modelini (örn. LLaMA, Mistral) **fine-tuning** yaparak özelleştirmek isteyenler için uçtan uca bir yol haritası sunar.  
Anlatımda **Supervised Fine-Tuning (SFT)**, **LoRA**, **QLoRA**, veri hazırlığı, eğitim, değerlendirme ve yayınlama adımları detaylı olarak yer alır.

---

## 1. Fine-Tuning Türleri

| Yöntem           | Açıklama                                                                 | Kullanım Durumu |
|------------------|---------------------------------------------------------------------------|-----------------|
| **SFT**          | Talimat → Yanıt formatında denetimli ince ayar.                          | Genel amaçlı, ilk adım |
| **LoRA**         | Büyük modeli dondurup düşük rütbeli adaptörleri eğitir.                  | Düşük VRAM, hızlı eğitim |
| **QLoRA**        | LoRA + 4-bit quantizasyon.                                               | Tek GPU (12–24GB) için |
| **DPO**          | Tercih öğrenimi (SFT sonrası).                                           | Üslup/format iyileştirme |
| **Devam Eğitimi**| Ham metin üzerinde alan dili uyumu.                                      | Domain adaptasyonu |

---

## 2. Donanım Gereksinimleri

- **7–8B model (QLoRA)**: 16–24GB VRAM
- **Tam fine-tune (FP16)**: Çok daha fazla VRAM ve çok GPU (gereksizse kullanma)
- **Windows**: WSL2 + Ubuntu önerilir (bitsandbytes uyumluluğu için)

---

## 3. Veri Hazırlığı

### 3.1. Formatlar

**Instruction Tuning Formatı**
```json
{"instruction":"Kısa bir özgeçmiş yaz.", "input":"3. sınıf bilgisayar müh.", "output":"Özgeçmiş metni..."}
```

**Prompt/Completion Formatı**
```json
{"prompt":"### Talimat:\nMetni özetle.\n### Yanıt:\n", "completion":"Özet..."}
```

> Eğitim ve inference sırasında **aynı prompt şablonunu** kullanın.

### 3.2. Temizlik

- Yinelenen, çok kısa veya çok uzun örnekleri çıkarın.
- Toksik veya istenmeyen içerikleri filtreleyin.
- Kod görevlerinde formatı net belirtin.

### 3.3. Bölme

- Train/Valid oranı: %90 / %10
- Veri sızıntısını önlemek için benzer örnekleri farklı setlere koymayın.

---

## 4. QLoRA + SFT ile Eğitim

### 4.1. Kurulum

```bash
pip install -U "transformers>=4.41" "datasets" "accelerate"   "bitsandbytes" "peft" "trl" "evaluate" "mlflow"
```

### 4.2. Eğitim Betiği

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

BASE = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA = "data/train.jsonl"
OUT  = "runs/qlora-sft"

raw = load_dataset("json", data_files=DATA, split="train")
split = raw.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split["train"], split["test"]

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, torch_dtype="bfloat16", device_map="auto")

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

def to_text(ex):
    instr, inp, out = ex.get("instruction","").strip(), ex.get("input","").strip(), ex.get("output","").strip()
    if instr and inp:
        prompt = f"<s>### Talimat:\n{instr}\n\n### Girdi:\n{inp}\n\n### Yanıt:\n"
    elif instr:
        prompt = f"<s>### Talimat:\n{instr}\n\n### Yanıt:\n"
    else:
        prompt = ex["prompt"]
    return {"text": prompt + out + tok.eos_token}

train_ds = train_ds.map(to_text, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(to_text, remove_columns=eval_ds.column_names)

cfg = SFTConfig(
    output_dir=OUT, num_train_epochs=3,
    per_device_train_batch_size=1, gradient_accumulation_steps=16,
    learning_rate=2e-4, logging_steps=20,
    eval_strategy="steps", eval_steps=200,
    save_steps=200, save_total_limit=2,
    warmup_ratio=0.03, lr_scheduler_type="cosine",
    max_seq_length=2048, packing=True,
    gradient_checkpointing=True, bf16=True
)

trainer = SFTTrainer(
    model=model, args=cfg, peft_config=peft_cfg,
    train_dataset=train_ds, eval_dataset=eval_ds,
    tokenizer=tok
)
trainer.train()
trainer.save_model(f"{OUT}/adapter")
tok.save_pretrained(OUT)
```

---

## 5. İnference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

BASE = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER = "runs/qlora-sft/adapter"

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", load_in_4bit=True)
model = PeftModel.from_pretrained(base, ADAPTER)

pipe = pipeline("text-generation", model=model, tokenizer=tok, device_map="auto")
prompt = "### Talimat:\nFastAPI’de /predict endpoint’i yaz.\n\n### Yanıt:\n"
print(pipe(prompt, max_new_tokens=300, temperature=0.4, do_sample=True)[0]["generated_text"])
```

---

## 6. Değerlendirme

- **Otomatik metrikler**: Exact Match, ROUGE, BLEU
- **LLM-as-Judge**: Talimata uygunluk, doğruluk, zararsızlık
- **Manual review**: Türkçe akıcılık, halüsinasyon kontrolü

---

## 7. Yayınlama

- **LoRA adaptörü**: Hafif, kolay paylaşım
- **Merge edilmiş model**: Tek dosya, büyük boyut
- **Sunum seçenekleri**:
  - [vLLM](https://github.com/vllm-project/vllm)
  - [TGI](https://github.com/huggingface/text-generation-inference)
  - FastAPI wrapper

---

## 8. Sık Hatalar

- Eğitim ve inference prompt formatının farklı olması
- `pad_token` tanımlamamak
- Çok yüksek learning rate
- Veride sızıntı (train/test aynı örnek)

---

## 9. Lisans ve Güvenlik

- Kullanılan taban modelin lisansına uyun.
- Yayın öncesi zararlı içerik testleri yapın.
- Gerekiyorsa çıktı sonrası moderasyon ekleyin.

---
