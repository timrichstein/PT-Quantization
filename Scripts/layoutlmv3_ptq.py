"""
Scripts/layoutlmv3_ptq.py

Post-Training Quantisierung (PTQ) für LayoutLMv3 (Teacher und distillierter Student).

Lädt das fine-getunte LayoutLMv3 Teacher- ODER das distillierte 4-Layer-Student-
Modell für ein Dataset, quantisiert es (oder lässt es als FP32-Baseline),
evaluiert Score + Größe, vermisst die Latenz nach SlimDoc-Protokoll und schreibt
die Ergebnisse modellübergreifend nach results/.

Quantisierung (granular auf Layer-Ebene):
  - Encoder-nn.Linear (du_model.encoder.layer.*): dynamisch INT8
        Gewichte symmetrisch/per-channel, Aktivierungen affin/per-tensor
        (Teacher: 72 Linear / Student: 24 Linear — der startswith-Filter trifft
         automatisch nur die vorhandenen Layer)
  - word_embeddings:                              weight-only INT8
  - rel_pos_*_bias, classifier, LayerNorm, Conv2d, Biases,
    kleine Layout-/Positions-Embeddings:          bleiben FP32

Aufruf (vier Varianten erzeugen alle Pareto-Punkte):
    python -u Scripts/layoutlmv3_ptq.py --all --threads 8                            # teacher int8
    python -u Scripts/layoutlmv3_ptq.py --all --threads 8 --no-quantize              # teacher fp32
    python -u Scripts/layoutlmv3_ptq.py --all --threads 8 --model student            # student int8
    python -u Scripts/layoutlmv3_ptq.py --all --threads 8 --model student --no-quantize  # student fp32
"""

import argparse
import csv
import os
import platform
import sys

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

# ── Pfade einrichten ──────────────────────────────────────────────────────────
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIMDOC_PATH = "/data/stud/2026-richtstein-ba/slimdoc-main"
sys.path.insert(0, SLIMDOC_PATH)
sys.path.insert(0, REPO_ROOT)

from slimdoc import DATASET_CONF, ENV, DUModel, TASKS
from slimdoc.model import get_model

from eval.evaluate import evaluate_quantized_model, benchmark_latency
from utils.model_utils import get_model_size_mb


# ── Konfiguration ─────────────────────────────────────────────────────────────
# Mapping: Dataset → Run-Name des fine-getunten LayoutLMv3 Teacher-Modells
TEACHER_RUN_NAMES = {
    "FUNSD":              "LayoutLMv3-TextAndImage_ft-teacher_funsd_50epochs",
    "SROIE":              "LayoutLMv3-TextAndImage_ft-teacher_sroie_50epochs",
    "DocVQA":             "LayoutLMv3-TextAndImage_ft-teacher_docvqa_30epochs",
    "InfographicsVQA":    "LayoutLMv3-TextAndImage_ft-teacher_infographicsvqa_30epochs",
    "WikiTableQuestions": "LayoutLMv3-TextAndImage_ft-teacher_wikitablequestions_30epochs",
}

# Mapping: Dataset → Run-Name des distillierten 4-Layer-Studenten (volles Vokabular)
STUDENT_RUN_NAMES = {
    "FUNSD":              "LayoutLMv3-TextAndImage_dt-student_funsd_layers-0-3-6-9_100epochs_alpha=1-beta=1-gamma=100-delta=0.01",
    "SROIE":              "LayoutLMv3-TextAndImage_dt-student_sroie_layers-0-3-6-9_100epochs_alpha=1-beta=1-gamma=100-delta=0.01",
    "DocVQA":             "LayoutLMv3-TextAndImage_dt-student_docvqa_layers-0-3-6-9_60epochs_alpha=1-beta=1-gamma=100-delta=0.01",
    "InfographicsVQA":    "LayoutLMv3-TextAndImage_dt-student_infographicsvqa_layers-0-3-6-9_60epochs_alpha=1-beta=1-gamma=100-delta=0.01",
    "WikiTableQuestions": "LayoutLMv3-TextAndImage_dt-student_wikitablequestions_layers-0-3-6-9_60epochs_alpha=1-beta=1-gamma=100-delta=0.01",
}

SUPPORTED_DATASETS = list(TEACHER_RUN_NAMES.keys())

# Modell-Label fürs CSV
MODEL_LABELS = {
    "teacher": "layoutlmv3_teacher",
    "student": "layoutlmv3_student_4L",
}


# ── Ergebnis-Logging ──────────────────────────────────────────────────────────
def append_row(filename: str, row: dict, base: str = None) -> str:
    """Hängt eine Zeile an eine CSV in results/ an (Header beim ersten Mal)."""
    base = base or os.path.join(REPO_ROOT, "results")
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, filename)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    return path


def _cpu_name() -> str:
    """Liest den CPU-Modellnamen (für den Methodenteil der Arbeit)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


# ── Modelle laden ─────────────────────────────────────────────────────────────
def load_layoutlmv3_teacher(dataset_name: str) -> torch.nn.Module:
    """
    Lädt das fine-getunte LayoutLMv3 Teacher-Modell für ein gegebenes Dataset.

    1. get_model() baut das Modell-Skelett auf (microsoft/layoutlmv3-base).
    2. load_state_dict() ersetzt alle Gewichte mit den fine-getunten Werten.

    Returns:
        Fine-getuntes LayoutLMv3-Modell im eval()-Modus auf CPU.
    """
    run_name = TEACHER_RUN_NAMES[dataset_name]
    ds_conf  = DATASET_CONF[dataset_name]
    device   = torch.device("cpu")

    print(f"  Lade Teacher-Checkpoint: {run_name}")

    chk_path   = ENV.MODELS_DIR / run_name / "best.pth"
    checkpoint = torch.load(chk_path, map_location=device)

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

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  Teacher geladen. Task: {ds_conf.task} | "
          f"Layer: {model.config.num_hidden_layers}")
    return model


def load_layoutlmv3_student(dataset_name: str) -> torch.nn.Module:
    """
    Lädt das distillierte 4-Layer-LayoutLMv3-Studentenmodell (volles Vokabular).

    Spiegelt SlimDocs evaluate_model-Lademuster: Die Metadaten (is_student,
    student_layer_map, vocab_name) kommen aus dem Checkpoint selbst, nicht aus
    dem Run-Namen. Bei vocab_name=None bleibt vocab_map=None -> kein
    Vokabular-Lookup nötig, die Standard-Eval funktioniert unverändert.

    Returns:
        4-Layer-Studentenmodell im eval()-Modus auf CPU.
    """
    run_name = STUDENT_RUN_NAMES[dataset_name]
    ds_conf  = DATASET_CONF[dataset_name]
    device   = torch.device("cpu")

    print(f"  Lade Student-Checkpoint: {run_name}")

    chk_path   = ENV.MODELS_DIR / run_name / "best.pth"
    checkpoint = torch.load(chk_path, map_location=device)

    # Metadaten aus dem Checkpoint (wie in SlimDocs evaluate_model)
    is_student        = checkpoint["is_student"]
    student_layer_map = checkpoint["student_layer_map"]
    vocab_name        = checkpoint["vocab_name"]
    assert is_student, "Checkpoint ist kein Studentenmodell."
    assert vocab_name is None, (
        f"vocab_name={vocab_name!r}: reduziertes Vokabular wird hier nicht "
        f"unterstützt (Vokabular-Lookup fehlt in der Eval)."
    )

    # Skelett bauen: get_model konstruiert via create_student das 4-Layer-Modell.
    # Die kopierten Teacher-Gewichte werden direkt danach durch load_state_dict
    # mit den distillierten Werten überschrieben (folgt exakt dem SlimDoc-Pfad).
    model = get_model(
        model_type=DUModel.LayoutLMv3_TextAndImage,
        task=ds_conf.task,
        is_student=is_student,
        num_labels=ds_conf.num_labels,
        vocab_map=None,                       # volles Vokabular
        device=device,
        teacher_run_name=None,                # Gewichte kommen aus dem Checkpoint
        student_layer_map=student_layer_map,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  Student geladen. Task: {ds_conf.task} | "
          f"Layer: {model.config.num_hidden_layers} | layer_map: {student_layer_map}")
    return model


def _load_model(model_kind: str, dataset_name: str) -> torch.nn.Module:
    if model_kind == "teacher":
        return load_layoutlmv3_teacher(dataset_name)
    elif model_kind == "student":
        return load_layoutlmv3_student(dataset_name)
    raise ValueError(f"Unbekannte model_kind: {model_kind!r}")


# ── Quantisierung ─────────────────────────────────────────────────────────────
def quantize_layoutlmv3(model: torch.nn.Module) -> torch.nn.Module:
    """
    Manuelle, explizite dynamische INT8-PTQ.

    Quantisiert:
      - alle Encoder-Linear (du_model.encoder.layer.*): dynamisch INT8,
        Gewichte symmetrisch/per-channel, Aktivierungen affin/per-tensor.
        Teacher -> 72 Linear (12 Layer), Student -> 24 Linear (4 Layer).
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
        for p in parents:           # funktioniert auch durch ModuleList-Indizes
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

    print(f"  Quantisiert: {len(linear_targets)} Encoder-Linear + word_embeddings")
    return model


# ── Score + Größe (pro Dataset) ───────────────────────────────────────────────
def run(dataset_name: str, model_kind: str = "teacher", quantize: bool = True):
    """
    Lädt Modell (Teacher oder Student), (quantisiert), evaluiert Score + Größe
    und schreibt eine Zeile nach results/accuracy_size.csv.
    KEINE Latenzmessung hier (die läuft genau einmal in run_latency).
    """
    model_label = MODEL_LABELS[model_kind]
    print("\n" + "=" * 60)
    print(f"  LayoutLMv3 PTQ | {model_label} | {dataset_name} | quantize={quantize}")
    print("=" * 60)

    ds_conf = DATASET_CONF[dataset_name]

    # [1] Laden
    print(f"\n[1] Lade {model_kind}-Modell...")
    model = _load_model(model_kind, dataset_name)
    size_base = get_model_size_mb(model)
    print(f"  ✓ Modell geladen | FP32-Größe: {size_base:.1f} MB")

    # [2] (Quantisieren)
    if quantize:
        print("\n[2] Quantisiere Modell (dynamisch INT8)...")
        model_eval = quantize_layoutlmv3(model)
        precision = "int8"
    else:
        print("\n[2] Überspringe Quantisierung (FP32-Baseline)...")
        model_eval = model
        precision = "fp32"
    size_eval = get_model_size_mb(model_eval)
    print(f"  Größe ({precision}): {size_eval:.1f} MB "
          f"({size_base / size_eval:.2f}x ggü. eigener FP32-Größe)")

    # [3] Evaluieren (Score)
    print("\n[3] Evaluiere Modell (Score)...")
    results = evaluate_quantized_model(
        model=model_eval,
        dataset_name=dataset_name,
        task=ds_conf.task,
        model_type=DUModel.LayoutLMv3_TextAndImage,
        split="test",
        batch_size=16,
    )

    # [4] Ausgeben + speichern
    print("\n[4] Ergebnisse:")
    print(f"  Score:   {results['score']:.4f}")
    print(f"  Samples: {results['total_samples']}")
    print(f"  Größe:   {size_eval:.1f} MB")

    append_row("accuracy_size.csv", {
        "model":     model_label,
        "precision": precision,
        "dataset":   dataset_name,
        "task":      str(ds_conf.task),
        "score":     round(results["score"], 4),
        "size_mb":   round(size_eval, 1),
        "n_samples": results["total_samples"],
    })


# ── Latenz (genau einmal) ─────────────────────────────────────────────────────
def run_latency(dataset_name: str, model_kind: str = "teacher",
                quantize: bool = True, num_threads: int = 8):
    """
    Vermisst die Latenz EINMAL (architektur-, nicht datensatzabhängig) nach
    SlimDoc-Protokoll und schreibt eine Zeile nach results/latency.csv.
    """
    model_label = MODEL_LABELS[model_kind]
    print("\n" + "=" * 60)
    print(f"  Latenz-Benchmark | {model_label} | quantize={quantize} | "
          f"threads={num_threads}")
    print("=" * 60)

    model = _load_model(model_kind, dataset_name)
    if quantize:
        model = quantize_layoutlmv3(model)
        precision = "int8"
    else:
        precision = "fp32"

    lat = benchmark_latency(
        model, DUModel.LayoutLMv3_TextAndImage, num_threads=num_threads,
    )
    print(f"  {lat['batch_ms_mean']:.1f} ± {lat['batch_ms_std']:.1f} ms/Batch | "
          f"{lat['sample_ms_mean']:.2f} ms/Sample | "
          f"{lat['throughput_samples_s']:.1f} Samples/s | "
          f"{lat['num_threads']} Threads")

    append_row("latency.csv", {
        "model":     model_label,
        "precision": precision,
        "cpu":       _cpu_name(),
        **lat,
    })


# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-Training Quantisierung für LayoutLMv3 (Teacher/Student)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, choices=SUPPORTED_DATASETS,
                        help="Einzelnes Dataset.")
    parser.add_argument("--all", action="store_true",
                        help="Alle 5 Datasets nacheinander.")
    parser.add_argument("--model", type=str, choices=["teacher", "student"],
                        default="teacher", help="Welches Modell evaluieren.")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Quantisierung überspringen (FP32-Baseline).")
    parser.add_argument("--threads", type=int, default=8,
                        help="Feste CPU-Threadzahl für den Latenz-Benchmark.")
    return parser.parse_args()


# ── Einstiegspunkt ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    if not args.all and not args.dataset:
        print("Fehler: Bitte --dataset DATASETNAME oder --all angeben.")
        sys.exit(1)

    datasets = SUPPORTED_DATASETS if args.all else [args.dataset]
    quantize = not args.no_quantize

    # Score + Größe pro Dataset
    for dataset in datasets:
        run(dataset, model_kind=args.model, quantize=quantize)

    # Latenz genau einmal (Architektur für alle Datasets identisch)
    run_latency(datasets[0], model_kind=args.model, quantize=quantize,
                num_threads=args.threads)