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