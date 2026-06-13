"""
Scripts/ablation.py

Ablationsstudie: isolierter Einfluss der einzelnen Quantisierungsteile.
Modell: LayoutLMv3, Datensatz: FUNSD (SER, weighted F1).

Fünf Konfigurationen (alle im selben Lauf, identische Umgebung):
  1 fp32                 – Baseline, nichts quantisiert
  2 embedding_only       – nur word_embeddings (weight-only INT8)
  3 linear_weight_only   – nur Encoder-Linear, Gewichte INT8 (Fake-Quant:
                           INT8 -> dequant -> FP32-Rechnung). Isoliert den
                           GENAUIGKEITS-Effekt der Gewichtsquantisierung.
                           Größe/Latenz bleiben FP32 (size_is_real_int8=False).
  4 linear_dynamic       – nur Encoder-Linear, dynamisch INT8 (Gewicht+Aktivierung).
                           Der Unterschied zu (3) ist der Aktivierungsbeitrag.
  5 full                 – Embedding + Linear dynamic (= produktives Setup).

Schreibt eine Zeile je Konfiguration nach results/ablation.csv.

Aufruf:
    python -u Scripts/ablation.py
"""

import copy
import os
import platform
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.ao.nn.quantized.dynamic import Linear as DynQLinear
from torch.ao.nn.quantized import Embedding as QEmbedding
from torch.ao.quantization import (
    per_channel_dynamic_qconfig,
    float_qparams_weight_only_qconfig,
)

REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIMDOC_PATH = "/data/stud/2026-richtstein-ba/slimdoc-main"
sys.path.insert(0, SLIMDOC_PATH)
sys.path.insert(0, REPO_ROOT)

from slimdoc import DATASET_CONF, DUModel
from slimdoc.model import get_model  # noqa: F401 (indirekt via load-Funktion)

from layoutlmv3_ptq import load_layoutlmv3_teacher
from eval.evaluate import evaluate_quantized_model, benchmark_latency
from utils.model_utils import get_model_size_mb

# ── Konfiguration ─────────────────────────────────────────────────────────────
DATASET     = "FUNSD"
MODEL_TYPE  = DUModel.LayoutLMv3_TextAndImage
MODEL_LABEL = "layoutlmv3_teacher"
ENCODER_PREFIX = "du_model.encoder.layer."
EMB_NAME       = "du_model.embeddings.word_embeddings"
THREADS     = 8
RESULTS     = os.path.join(REPO_ROOT, "results")

CONFIGS = ["fp32", "embedding_only", "linear_weight_only", "linear_dynamic", "full"]


# ── Hilfen ────────────────────────────────────────────────────────────────────
def _set_module(root, dotted_name, new_module):
    *parents, leaf = dotted_name.split(".")
    obj = root
    for p in parents:
        obj = getattr(obj, p)
    setattr(obj, leaf, new_module)


def _quantize_embedding(model):
    emb = dict(model.named_modules())[EMB_NAME]
    emb.qconfig = float_qparams_weight_only_qconfig
    _set_module(model, EMB_NAME, QEmbedding.from_float(emb))


def _quantize_linears_dynamic(model):
    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and n.startswith(ENCODER_PREFIX)]
    for name, module in targets:
        module.qconfig = per_channel_dynamic_qconfig
        _set_module(model, name, DynQLinear.from_float(module))
    return len(targets)


def _fakequant_linears_weight_only(model):
    """
    Fake-Quant der Encoder-Linear-GEWICHTE: per-channel symmetrisch INT8,
    sofort zurück nach FP32. Isoliert den Genauigkeitseffekt der
    Gewichtsquantisierung; Rechnung bleibt FP32 (keine echte INT8-Größe/Latenz).
    """
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name.startswith(ENCODER_PREFIX):
            w = m.weight.detach()
            # per-channel symmetrisch (eine Skala pro Ausgabezeile), qint8
            s = w.abs().amax(dim=1) / 127.5            # [out_features]
            s = s.clamp(min=1e-12)
            q = torch.clamp(torch.round(w / s[:, None]), -128, 127)
            w_fakequant = q * s[:, None]               # dequantisiert -> FP32
            with torch.no_grad():
                m.weight.copy_(w_fakequant)
            n += 1
    return n


def build_model(config: str):
    """Lädt ein frisches FP32-Modell und wendet die jeweilige Konfiguration an."""
    torch.backends.quantized.engine = "fbgemm"
    model = load_layoutlmv3_teacher(DATASET)

    real_int8 = True
    if config == "fp32":
        real_int8 = False
    elif config == "embedding_only":
        _quantize_embedding(model)
    elif config == "linear_weight_only":
        n = _fakequant_linears_weight_only(model)
        print(f"    Fake-Quant (weight-only) auf {n} Encoder-Linear")
        real_int8 = False  # Gewichte sind FP32 gespeichert -> Größe nicht echt INT8
    elif config == "linear_dynamic":
        n = _quantize_linears_dynamic(model)
        print(f"    Dynamisch quantisiert: {n} Encoder-Linear")
    elif config == "full":
        _quantize_embedding(model)
        n = _quantize_linears_dynamic(model)
        print(f"    Voll: word_embeddings + {n} Encoder-Linear")
    else:
        raise ValueError(config)

    return model.cpu().eval(), real_int8


# ── Hauptablauf ───────────────────────────────────────────────────────────────
def _cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


if __name__ == "__main__":
    ds_conf = DATASET_CONF[DATASET]
    os.makedirs(RESULTS, exist_ok=True)
    rows = []

    for cfg in CONFIGS:
        print("\n" + "=" * 60)
        print(f"  Ablation | config={cfg}")
        print("=" * 60)

        model, real_int8 = build_model(cfg)
        size_mb = get_model_size_mb(model)

        res = evaluate_quantized_model(
            model=model, dataset_name=DATASET, task=ds_conf.task,
            model_type=MODEL_TYPE, split="test", batch_size=16,
        )
        lat = benchmark_latency(model, MODEL_TYPE, num_threads=THREADS)

        print(f"    Score: {res['score']:.4f} | Größe: {size_mb:.1f} MB "
              f"(echt INT8: {real_int8}) | Latenz: {lat['batch_ms_mean']:.1f} ms/Batch")

        rows.append({
            "model":            MODEL_LABEL,
            "dataset":          DATASET,
            "config":           cfg,
            "score":            round(res["score"], 4),
            "size_mb":          round(size_mb, 1),
            "size_is_real_int8": real_int8,
            "batch_ms_mean":    round(lat["batch_ms_mean"], 1),
            "batch_ms_std":     round(lat["batch_ms_std"], 1),
            "latency_is_real_int8": cfg in ("linear_dynamic", "full"),
            "num_threads":      lat["num_threads"],
            "cpu":              _cpu_name(),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS, "ablation.csv")
    df.to_csv(path, index=False)
    print(f"\nFertig. -> {path}")
    print(df.to_string(index=False))