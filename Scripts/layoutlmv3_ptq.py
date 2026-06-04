"""
scripts/layoutlmv3_ptq.py

Post-Training Quantisierung (PTQ) für das LayoutLMv3 Teacher-Modell.

Dieses Skript lädt das fine-getunte LayoutLMv3 Teacher-Modell für ein
gegebenes Dataset, quantisiert es und evaluiert das quantisierte Modell.

Die Quantisierung wird granular auf Layer-Ebene durchgeführt:
  - nn.Linear Gewichte:      [WIRD NOCH DEFINIERT]
  - nn.Linear Aktivierungen: [WIRD NOCH DEFINIERT]
  - nn.Embedding Gewichte:   [WIRD NOCH DEFINIERT]
  - Bias:                    bleibt float32 – zu klein für sinnvolle Ersparnis
  - nn.LayerNorm:            bleibt float32 – sehr empfindlich auf Quantisierung

Aufruf:
    python scripts/layoutlmv3_ptq.py --dataset FUNSD
    python scripts/layoutlmv3_ptq.py --dataset SROIE
    python scripts/layoutlmv3_ptq.py --all
"""
import torch
import torch.nn as nn
# Pfade gelten für aktuelle torch-Versionen (torch.ao.*).
# Bei sehr altem torch hieße es torch.nn.quantized.* statt torch.ao.nn.quantized.*
from torch.ao.nn.quantized.dynamic import Linear as DynQLinear
from torch.ao.nn.quantized import Embedding as QEmbedding
from torch.ao.quantization import (
    per_channel_dynamic_qconfig,       # Gewichte: symmetrisch, per-channel
    float_qparams_weight_only_qconfig, # Embeddings: weight-only
)

import argparse
import os
import sys

# ── Pfade einrichten ──────────────────────────────────────────────────────────
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIMDOC_PATH = "/data/stud/2026-richtstein-ba/slimdoc-main"
sys.path.insert(0, SLIMDOC_PATH)
sys.path.insert(0, REPO_ROOT)

from slimdoc import DATASET_CONF, ENV, DUModel, TASKS
from slimdoc.model import get_model


# ── Konfiguration ─────────────────────────────────────────────────────────────
# Mapping: Dataset → Run-Name des fine-getunten LayoutLMv3 Teacher-Modells
TEACHER_RUN_NAMES = {
    "FUNSD":              "LayoutLMv3-TextAndImage_ft-teacher_funsd_50epochs",
    "SROIE":              "LayoutLMv3-TextAndImage_ft-teacher_sroie_50epochs",
    "DocVQA":             "LayoutLMv3-TextAndImage_ft-teacher_docvqa_30epochs",
    "InfographicsVQA":    "LayoutLMv3-TextAndImage_ft-teacher_infographicsvqa_30epochs",
    "WikiTableQuestions": "LayoutLMv3-TextAndImage_ft-teacher_wikitablequestions_30epochs",
}

SUPPORTED_DATASETS = list(TEACHER_RUN_NAMES.keys())


# ── Modell laden ──────────────────────────────────────────────────────────────
def load_layoutlmv3_teacher(dataset_name: str) -> torch.nn.Module:
    """
    Lädt das fine-getunte LayoutLMv3 Teacher-Modell für ein gegebenes Dataset.

    1. get_model() baut das Modell-Skelett auf (microsoft/layoutlmv3-base
       von HuggingFace) – Architektur steht, Gewichte sind vortrainiert.
    2. load_state_dict() ersetzt alle Gewichte mit den fine-getunten Werten
       aus dem lokalen Checkpoint (best.pth).

    Args:
        dataset_name: Name des Datasets (z.B. "FUNSD")

    Returns:
        model: Fine-getuntes LayoutLMv3-Modell im eval()-Modus auf CPU
    """
    run_name = TEACHER_RUN_NAMES[dataset_name]
    ds_conf  = DATASET_CONF[dataset_name]
    device   = torch.device("cpu")

    print(f"  Lade Checkpoint: {run_name}")

    # Checkpoint laden
    chk_path   = ENV.MODELS_DIR / run_name / "best.pth"
    checkpoint = torch.load(chk_path, map_location=device)

    # Schritt 1: Modell-Skelett aufbauen
    model = get_model(
        model_type=DUModel.LayoutLMv3_TextAndImage,
        task=ds_conf.task,
        is_student=False,
        num_labels=ds_conf.num_labels,
        vocab_map=None,
        device=device,
        teacher_run_name=None,
        student_layer_map=None,
    )

    # Schritt 2: Fine-getunete Gewichte laden
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  Modell geladen. Task: {ds_conf.task}")

    return model


# ── Quantisierung ─────────────────────────────────────────────────────────────
def quantize_layoutlmv3(model: torch.nn.Module) -> torch.nn.Module:
    """
    Manuelle, explizite dynamische INT8-PTQ.

    Quantisiert:
      - alle 72 Encoder-Linear (du_model.encoder.layer.*): dynamisch INT8,
        Gewichte symmetrisch/per-channel, Aktivierungen affin/per-tensor
        (zur Laufzeit pro Batch).
      - word_embeddings: weight-only INT8.

    Bleibt FP32:
      rel_pos_*_bias, classifier, alle LayerNorm, patch_embed.proj (Conv2d),
      alle Biases, sowie die kleinen Layout-/Positions-Embeddings.
    """
    torch.backends.quantized.engine = "fbgemm"  # x86-Server-Backend
    model = model.cpu().eval()

    def _set_module(root, dotted_name, new_module):
        *parents, leaf = dotted_name.split(".")
        obj = root
        for p in parents:           # funktioniert auch durch ModuleList-Indizes ("0", "1", ...)
            obj = getattr(obj, p)
        setattr(obj, leaf, new_module)

    # 1) Encoder-Linear -> dynamisch quantisiert
    linear_targets = [
        (n, m) for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n.startswith("du_model.encoder.layer.")
    ]
    for name, module in linear_targets:
        module.qconfig = per_channel_dynamic_qconfig
        _set_module(model, name, DynQLinear.from_float(module))

    # 2) word_embeddings -> weight-only quantisiert
    emb_name = "du_model.embeddings.word_embeddings"
    emb = dict(model.named_modules())[emb_name]
    emb.qconfig = float_qparams_weight_only_qconfig
    _set_module(model, emb_name, QEmbedding.from_float(emb))

    print(f"  Quantisiert: {len(linear_targets)} Encoder-Linear + word_embeddings "
          f"(erwartet 72 Linear)")
    return model


# ── Hauptfunktion ─────────────────────────────────────────────────────────────
def run(dataset_name: str):
    """
    Kompletter PTQ-Ablauf: 1. Laden  2. Quantisieren  3. Evaluieren  4. Ausgeben
    """
    from eval.evaluate import evaluate_quantized_model
    from utils.model_utils import get_model_size_mb

    print("\n" + "=" * 60)
    print(f"  LayoutLMv3 PTQ | Dataset: {dataset_name}")
    print("=" * 60)

    ds_conf = DATASET_CONF[dataset_name]

    # [1] Laden
    print("\n[1] Lade fine-getuntes LayoutLMv3 Teacher-Modell...")
    model = load_layoutlmv3_teacher(dataset_name)
    size_fp32 = get_model_size_mb(model)
    print(f"  ✓ Modell geladen | FP32-Größe: {size_fp32:.1f} MB")

    # [2] Quantisieren
    print("\n[2] Quantisiere Modell (dynamisch INT8)...")
    model_q = quantize_layoutlmv3(model)
    size_int8 = get_model_size_mb(model_q)
    print(f"  ✓ Quantisiert | INT8-Größe: {size_int8:.1f} MB "
          f"({size_fp32 / size_int8:.2f}x kleiner)")

    # [3] Evaluieren
    print("\n[3] Evaluiere quantisiertes Modell...")
    results = evaluate_quantized_model(
        model=model_q,
        dataset_name=dataset_name,
        task=ds_conf.task,
        model_type=DUModel.LayoutLMv3_TextAndImage,
        split="test",
        batch_size=16,
    )

    # [4] Ausgeben
    print("\n[4] Ergebnisse:")
    print(f"  Score:      {results['score']:.4f}")
    print(f"  Latenz:     {results['avg_forward_pass_per_sample_ms']:.2f} ms/Sample")
    print(f"  Throughput: {results['throughput_samples_s']:.1f} Samples/s")
    print(f"  Samples:    {results['total_samples']}")
    print(f"  FP32: {size_fp32:.1f} MB → INT8: {size_int8:.1f} MB")


# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-Training Quantisierung für LayoutLMv3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASETS,
        help="Dataset für Quantisierung und Evaluierung.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Alle 5 Datasets nacheinander.",
    )
    return parser.parse_args()


# ── Einstiegspunkt ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    if not args.all and not args.dataset:
        print("Fehler: Bitte --dataset DATASETNAME oder --all angeben.")
        sys.exit(1)

    datasets = SUPPORTED_DATASETS if args.all else [args.dataset]

    for dataset in datasets:
        run(dataset)
