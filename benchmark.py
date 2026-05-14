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