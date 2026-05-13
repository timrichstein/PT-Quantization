# LiLT INT8 Post-Training Quantization mit PyTorch

> **Ziel:** Das Document-Understanding-Modell **LiLT** (`SCUT-DLVCLab/lilt-roberta-en-base`)
> mit symmetrischer linearer INT8-Quantisierung verkleinern – ohne Neutraining.
> Danach wird das quantisierte Modell mit **W&B** auf verschiedenen Datasets gebenchmarkt.

---

## Inhaltsverzeichnis

1. [Konzept: Was ist INT8-PTQ?](#1-konzept-was-ist-int8-ptq)
2. [Voraussetzungen & Installation](#2-voraussetzungen--installation)
3. [Projektstruktur](#3-projektstruktur)
4. [Schritt-für-Schritt-Code](#4-schritt-für-schritt-code)
   - [4.1 Modell laden](#41-modell-laden)
   - [4.2 Quantisierung anwenden](#42-quantisierung-anwenden)
   - [4.3 Modell speichern & laden](#43-modell-speichern--laden)
   - [4.4 Benchmark mit W&B](#44-benchmark-mit-wb)
5. [Vollständiges Skript](#5-vollständiges-skript)
6. [Erwartete Ergebnisse & Troubleshooting](#6-erwartete-ergebnisse--troubleshooting)
7. [Nächste Schritte](#7-nächste-schritte)

---

## 1. Konzept: Was ist INT8-PTQ?

### Warum quantisieren?

Ein Transformer-Modell wie LiLT speichert seine Gewichte standardmäßig als **float32**
(32 Bit pro Zahl). Quantisierung ersetzt diese durch **int8** (8 Bit) – das spart:

| Metrik         | float32  | int8     | Ersparnis |
|----------------|----------|----------|-----------|
| Speicher       | ~440 MB  | ~110 MB  | ~75 %     |
| Inferenzzeit   | Baseline | ~1,5–2×  | schneller |
| Genauigkeit    | Baseline | minimal ↓| ~1–2 %    |

### Symmetrische vs. asymmetrische Quantisierung

Bei der **symmetrischen** Variante wird der Wertebereich eines Gewichts auf
`[-127, 127]` abgebildet – mit einem einzigen Skalierungsfaktor `s`:

```
x_quant = round(x / s)    wobei    s = max(|x|) / 127
```

Das ist einfacher zu implementieren als asymmetrische Quantisierung (die zusätzlich
einen Zero-Point benötigt) und funktioniert bei den meisten Transformer-Gewichten sehr gut.

### Post-Training (PTQ) vs. Quantization-Aware Training (QAT)

- **PTQ** (unser Ansatz): Modell ist bereits trainiert → wir quantisieren danach.
  Kein erneutes Training nötig. Einfach, schnell, guter Einstiegspunkt.
- **QAT**: Quantisierung wird *während* des Trainings simuliert → bessere Genauigkeit,
  aber deutlich aufwändiger.

---

## 2. Voraussetzungen & Installation

### Systemvoraussetzungen

- Python 3.9+
- VS Code mit der Python-Extension
- Git + GitHub-Account

### Virtuelle Umgebung erstellen (empfohlen)

```bash
# Im Projektordner:
python -m venv venv

# Aktivieren (Windows):
venv\Scripts\activate

# Aktivieren (macOS/Linux):
source venv/bin/activate
```

### Pakete installieren

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets wandb
```

> **Hinweis:** Für GPU-Support `--index-url https://download.pytorch.org/whl/cu121`
> statt dem CPU-Link verwenden (CUDA 12.1).

### `.gitignore` für das Repo

```
venv/
__pycache__/
*.pt
*.bin
wandb/
.env
```

---

## 3. Projektstruktur

```
lilt-quantization/
│
├── quantize.py            ← Hauptskript: Laden, Quantisieren, Speichern
├── benchmark.py           ← W&B Benchmark
├── requirements.txt       ← Paketliste
├── README.md              ← Diese Anleitung (oder Link dazu)
│
├── models/
│   ├── lilt_fp32/         ← Original float32 (optional lokal speichern)
│   └── lilt_int8/         ← Quantisiertes Modell
│
└── results/
    └── benchmark_results.json
```

`requirements.txt` generieren:

```bash
pip freeze > requirements.txt
```

---

## 4. Schritt-für-Schritt-Code

### 4.1 Modell laden

```python
# quantize.py  –  Teil 1: Modell & Tokenizer laden
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "SCUT-DLVCLab/lilt-roberta-en-base"

print("Lade Tokenizer und Modell von Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Modell in den Eval-Modus setzen (deaktiviert Dropout etc.)
model.eval()

print(f"Modell geladen. Parameter: {sum(p.numel() for p in model.parameters()):,}")
```

**Was passiert hier?**
- `AutoTokenizer` lädt den passenden Tokenizer für das Modell.
- `AutoModel` lädt die Gewichte von Hugging Face (beim ersten Mal ~440 MB Download).
- `model.eval()` ist **wichtig** – ohne es könnte Dropout die Inferenz beeinflussen.

---

### 4.2 Quantisierung anwenden

PyTorch bringt `torch.quantization` direkt mit – kein extra Paket nötig.

```python
# quantize.py  –  Teil 2: INT8-Quantisierung (dynamisch, symmetrisch)
import torch

def quantize_model_int8(model):
    """
    Wendet dynamische INT8-Quantisierung auf alle Linear-Layer an.
    
    Dynamisch bedeutet: Die Aktivierungen werden zur Laufzeit quantisiert,
    die Gewichte werden einmalig vorab quantisiert (symmetrisch).
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},   # Diese Layer-Typen werden quantisiert
        dtype=torch.qint8    # Ziel-Datentyp: 8-Bit Integer
    )
    return quantized_model

# Größe vor der Quantisierung messen
def get_model_size_mb(model):
    """Gibt die Modellgröße in MB zurück."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        torch.save(model.state_dict(), f.name)
        size_mb = os.path.getsize(f.name) / (1024 * 1024)
    os.unlink(f.name)
    return size_mb

size_before = get_model_size_mb(model)
print(f"Größe vor Quantisierung: {size_before:.1f} MB")

# Quantisierung durchführen
print("Quantisiere Modell...")
quantized_model = quantize_model_int8(model)

size_after = get_model_size_mb(quantized_model)
print(f"Größe nach Quantisierung: {size_after:.1f} MB")
print(f"Reduktion: {(1 - size_after/size_before)*100:.1f}%")
```

**Was ist `quantize_dynamic`?**

`torch.quantization.quantize_dynamic` ist die einfachste Quantisierungs-API in PyTorch:
- Die **Gewichte** aller `nn.Linear`-Layer werden sofort in int8 umgewandelt (symmetrisch).
- Die **Aktivierungen** (Zwischenergebnisse) werden dynamisch zur Inferenzzeit quantisiert.
- Kein Kalibrierungsdaten-Set nötig – daher "Post-Training" im einfachsten Sinne.

---

### 4.3 Modell speichern & laden

```python
# quantize.py  –  Teil 3: Speichern und wieder laden
import os

SAVE_PATH = "models/lilt_int8"
os.makedirs(SAVE_PATH, exist_ok=True)

# Quantisiertes Modell speichern
torch.save(quantized_model.state_dict(), f"{SAVE_PATH}/model_int8.pt")
tokenizer.save_pretrained(SAVE_PATH)
print(f"Modell gespeichert unter: {SAVE_PATH}/")

# --- Später wieder laden ---
def load_quantized_model(save_path, original_model_name):
    """
    Lädt ein zuvor quantisiertes Modell.
    Das Original-Modell wird als Skelett benötigt, dann laden wir die INT8-Gewichte.
    """
    from transformers import AutoModel, AutoTokenizer
    
    # Erst das Original-Skelett laden
    base_model = AutoModel.from_pretrained(original_model_name)
    base_model.eval()
    
    # Dann das Skelett quantisieren (gleiche Struktur wie beim Speichern)
    quantized_skeleton = torch.quantization.quantize_dynamic(
        base_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # INT8-Gewichte einfügen
    quantized_skeleton.load_state_dict(
        torch.load(f"{save_path}/model_int8.pt", map_location="cpu")
    )
    
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    return quantized_skeleton, tokenizer
```

---

### 4.4 Benchmark mit W&B

```python
# benchmark.py  –  Vollständiges Benchmark-Skript mit W&B
import torch
import time
import wandb
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# ── W&B initialisieren ──────────────────────────────────────────────────────
wandb.init(
    project="lilt-quantization-benchmark",
    name="int8-ptq-vs-fp32",
    config={
        "model": "SCUT-DLVCLab/lilt-roberta-en-base",
        "quantization": "dynamic-int8",
        "method": "torch.quantization.quantize_dynamic",
    }
)

# ── Modelle laden ───────────────────────────────────────────────────────────
MODEL_NAME = "SCUT-DLVCLab/lilt-roberta-en-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

fp32_model = AutoModel.from_pretrained(MODEL_NAME)
fp32_model.eval()

int8_model = torch.quantization.quantize_dynamic(
    AutoModel.from_pretrained(MODEL_NAME),
    {torch.nn.Linear},
    dtype=torch.qint8
)
int8_model.eval()

# ── Dataset laden ───────────────────────────────────────────────────────────
# FUNSD: Form Understanding in Noisy Scanned Documents
dataset = load_dataset("nielsr/funsd-layoutlmv3", split="test")

# ── Benchmark-Funktion ──────────────────────────────────────────────────────
def run_benchmark(model, dataset, tokenizer, n_samples=50, label="model"):
    """
    Führt Inferenz auf n_samples durch und misst Latenz.
    Gibt durchschnittliche Latenz und Ergebnisse zurück.
    """
    latencies = []
    
    for i, example in enumerate(dataset.select(range(n_samples))):
        words = example["tokens"]
        boxes = example["bboxes"]
        
        # Tokenisierung
        encoding = tokenizer(
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        
        # Inferenz messen
        with torch.no_grad():
            start = time.perf_counter()
            outputs = model(**encoding)
            end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        
        # Live-Logging in W&B
        wandb.log({
            f"{label}/latency_ms": latency_ms,
            f"{label}/sample_idx": i
        })
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"[{label}] Durchschn. Latenz: {avg_latency:.2f} ms über {n_samples} Samples")
    return avg_latency, latencies

# ── Benchmarks ausführen ────────────────────────────────────────────────────
print("Starte FP32-Benchmark...")
fp32_avg, fp32_latencies = run_benchmark(fp32_model, dataset, tokenizer, label="fp32")

print("Starte INT8-Benchmark...")
int8_avg, int8_latencies = run_benchmark(int8_model, dataset, tokenizer, label="int8")

# ── Größenvergleich ─────────────────────────────────────────────────────────
import tempfile, os

def get_size_mb(model):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        torch.save(model.state_dict(), f.name)
        size = os.path.getsize(f.name) / (1024 * 1024)
    os.unlink(f.name)
    return size

fp32_size = get_size_mb(fp32_model)
int8_size = get_size_mb(int8_model)

# ── Zusammenfassung in W&B loggen ───────────────────────────────────────────
summary = {
    "fp32_avg_latency_ms":  fp32_avg,
    "int8_avg_latency_ms":  int8_avg,
    "speedup_factor":       fp32_avg / int8_avg,
    "fp32_size_mb":         fp32_size,
    "int8_size_mb":         int8_size,
    "size_reduction_pct":   (1 - int8_size / fp32_size) * 100,
}
wandb.log(summary)

print("\n── Zusammenfassung ──────────────────────────────")
for k, v in summary.items():
    print(f"  {k}: {v:.2f}")

wandb.finish()
print("\nBenchmark abgeschlossen. Ergebnisse in W&B Dashboard.")
```

---

## 5. Vollständiges Skript

Hier ist `quantize.py` als vollständige, lauffähige Datei:

```python
# quantize.py
"""
LiLT INT8 Post-Training Quantisierung
Symmetrisch, dynamisch – mit PyTorch torch.quantization
"""

import os
import torch
from transformers import AutoTokenizer, AutoModel

# ── Konfiguration ────────────────────────────────────────────────────────────
MODEL_NAME = "SCUT-DLVCLab/lilt-roberta-en-base"
SAVE_PATH  = "models/lilt_int8"


# ── Hilfsfunktion: Modellgröße messen ───────────────────────────────────────
def get_model_size_mb(model):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        torch.save(model.state_dict(), f.name)
        size_mb = os.path.getsize(f.name) / (1024 * 1024)
    os.unlink(f.name)
    return size_mb


# ── 1. Modell laden ──────────────────────────────────────────────────────────
print("=" * 50)
print("Schritt 1: Modell laden")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

fp32_size = get_model_size_mb(model)
print(f"FP32 Modellgröße: {fp32_size:.1f} MB")
print(f"Anzahl Parameter: {sum(p.numel() for p in model.parameters()):,}")


# ── 2. Quantisierung ─────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Schritt 2: INT8-Quantisierung")
print("=" * 50)

quantized_model = torch.quantization.quantize_dynamic(
    model,                   # das zu quantisierende Modell
    {torch.nn.Linear},       # welche Layer-Typen quantisiert werden
    dtype=torch.qint8        # Ziel-Datentyp
)

int8_size = get_model_size_mb(quantized_model)
print(f"INT8 Modellgröße: {int8_size:.1f} MB")
print(f"Größenreduktion:  {(1 - int8_size/fp32_size)*100:.1f}%")


# ── 3. Kurzer Funktionstest ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Schritt 3: Funktionstest")
print("=" * 50)

# Minimales Beispiel: ein Satz + Dummy-Bounding-Boxes
test_words  = ["Hallo", "Welt", "das", "ist", "ein", "Test"]
test_boxes  = [[0, 0, 100, 50]] * len(test_words)  # Dummy-Boxen

encoding = tokenizer(
    test_words,
    boxes=test_boxes,
    return_tensors="pt",
    truncation=True,
    max_length=512,
    padding="max_length"
)

with torch.no_grad():
    outputs = quantized_model(**encoding)

print(f"Output-Shape (last_hidden_state): {outputs.last_hidden_state.shape}")
print("Funktionstest erfolgreich!")


# ── 4. Modell speichern ───────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Schritt 4: Modell speichern")
print("=" * 50)

os.makedirs(SAVE_PATH, exist_ok=True)
torch.save(quantized_model.state_dict(), f"{SAVE_PATH}/model_int8.pt")
tokenizer.save_pretrained(SAVE_PATH)

print(f"Gespeichert unter: {SAVE_PATH}/")
print("Fertig!")
```

---

## 6. Erwartete Ergebnisse & Troubleshooting

### Erwartete Ausgabe

```
Schritt 1: Modell laden
FP32 Modellgröße: ~440 MB
Anzahl Parameter: 125,537,544

Schritt 2: INT8-Quantisierung
INT8 Modellgröße: ~115 MB
Größenreduktion:  ~74%

Schritt 3: Funktionstest
Output-Shape (last_hidden_state): torch.Size([1, 512, 768])
Funktionstest erfolgreich!
```

### Häufige Fehler

| Fehler | Ursache | Lösung |
|--------|---------|--------|
| `ModuleNotFoundError: transformers` | Paket fehlt | `pip install transformers` |
| `RuntimeError: CUDA out of memory` | GPU zu klein | `model.to('cpu')` vor Quantisierung |
| `KeyError: bbox` | Falsches Dataset-Format | Bounding Box Format prüfen: `[x0, y0, x1, y1]` |
| `ValueError: Token indices out of range` | Sequenz zu lang | `truncation=True, max_length=512` setzen |
| W&B: `wandb: ERROR ...` | Nicht eingeloggt | `wandb login` im Terminal ausführen |

---

## 7. Nächste Schritte

Nachdem die INT8-PTQ funktioniert, sind das die nächsten Quantisierungsmethoden
in aufsteigender Komplexität:

1. **Static INT8 PTQ** – Kalibrierungsdaten werden genutzt, um Aktivierungsstatistiken
   zu berechnen. Bessere Genauigkeit als dynamisch, etwas mehr Aufwand.

2. **INT4 Quantisierung** – Weitere Halbierung des Speicherbedarfs.
   Kann mit `bitsandbytes` (`load_in_4bit=True`) umgesetzt werden.

3. **GPTQ** – Post-Training-Methode mit Gewichts-Optimierung für präzisere INT4/INT8.

4. **Quantization-Aware Training (QAT)** – Quantisierung wird ins Training integriert.
   Beste Genauigkeit, aber höchster Aufwand.

---

*Erstellt für das LiLT Quantization Research Projekt.*
*Modell: `SCUT-DLVCLab/lilt-roberta-en-base` | Framework: PyTorch + HuggingFace Transformers*
