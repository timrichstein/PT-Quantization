"""
Scripts/lilt_ptq.py

Post-Training Quantisierung (PTQ) für LiLT (Teacher und distillierter Student).

Lädt das fine-getunte LiLT Teacher- ODER das distillierte 4-Layer-Student-Modell
(LiLT-TextFlow) für ein Dataset, quantisiert es (oder lässt es als FP32-Baseline),
evaluiert Score + Größe, vermisst die Latenz nach SlimDoc-Protokoll und schreibt
die Ergebnisse modellübergreifend nach results/ (gemeinsame CSVs mit LayoutLMv3).

Architektur-Hinweis (unterscheidet LiLT von LayoutLMv3):
  LiLT hat ZWEI parallele Flows pro Layer — einen breiten Text-Flow (768) und
  einen schmalen Layout-Flow (192) — gekoppelt über BiACM. Pro Layer also 12
  Linear (6 Text + 6 Layout). Es gibt KEINEN Bildzweig (kein Conv2d).

Quantisierung (granular auf Layer-Ebene):
  - Encoder-nn.Linear beider Flows (du_model.encoder.layer.*): dynamisch INT8
        Gewichte symmetrisch/per-channel, Aktivierungen affin/per-tensor
        Teacher -> 144 Linear (12 Layer x 12), Student -> 48 Linear (4 Layer x 12)
  - word_embeddings:                              weight-only INT8
  Bleibt FP32 (automatisch durch den startswith-Filter ausgeschlossen):
  - box_linear_embeddings (Layout-Koordinaten-Projektion, traegt das raeumliche Signal)
  - pooler.dense (fuer SER/VQA ungenutzt), classifier (Head)
  - alle LayerNorm, alle Biases, BiACM (keine Linear-Schicht)

Aufruf (vier Varianten erzeugen alle Pareto-Punkte):
    python -u Scripts/lilt_ptq.py --all --threads 8                            # teacher int8
    python -u Scripts/lilt_ptq.py --all --threads 8 --no-quantize              # teacher fp32
    python -u Scripts/lilt_ptq.py --all --threads 8 --model student            # student int8
    python -u Scripts/lilt_ptq.py --all --threads 8 --model student --no-quantize  # student fp32
"""

import argparse
import os
import platform
import sys

import pandas as pd
import torch
import torch.nn as nn
# Pfade gelten für aktuelle torch-Versionen (torch.ao.*).
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
MODEL_TYPE = DUModel.LiLT_TextFlow

# Mapping: Dataset → Run-Name des fine-getunten LiLT Teacher-Modells
TEACHER_RUN_NAMES = {
    "FUNSD":              "LiLT-TextFlow_ft-teacher_funsd_50epochs",
    "SROIE":              "LiLT-TextFlow_ft-teacher_sroie_50epochs",
    "DocVQA":             "LiLT-TextFlow_ft-teacher_docvqa_30epochs",
    "InfographicsVQA":    "LiLT-TextFlow_ft-teacher_infographicsvqa_30epochs",
    "WikiTableQuestions": "LiLT-TextFlow_ft-teacher_wikitablequestions_30epochs",
}

# Mapping: Dataset → Run-Name des distillierten 4-Layer-Studenten (volles Vokabular)
STUDENT_RUN_NAMES = {
    "FUNSD":              "LiLT-TextFlow_dt-student_funsd_layers-0-3-6-9_100epochs_alpha=1-beta=1-gamma=100-delta=0.01",
    "SROIE":              "LiLT-TextFlow_dt-student_sroie_layers-0-3-6-9_100epochs_alpha=1-beta=1-gamma=100-delta=0.01",
    "DocVQA":             "LiLT-TextFlow_dt-student_docvqa_layers-0-3-6-9_60epochs_alpha=1-beta=1-gamma=100-delta=0.01",
    "InfographicsVQA":    "LiLT-TextFlow_dt-student_infographicsvqa_layers-0-3-6-9_60epochs_alpha=1-beta=1-gamma=100-delta=0.01",
    "WikiTableQuestions": "LiLT-TextFlow_dt-student_wikitablequestions_layers-0-3-6-9_60epochs_alpha=1-beta=1-gamma=100-delta=0.01",
}

SUPPORTED_DATASETS = list(TEACHER_RUN_NAMES.keys())

MODEL_LABELS = {
    "teacher": "lilt_teacher",
    "student": "lilt_student_4L",
}


# ── Ergebnis-Logging ──────────────────────────────────────────────────────────
def upsert_row(filename: str, row: dict, key_cols: list, base: str = None) -> str:
    """
    Schreibt eine Zeile nach results/<filename> und ersetzt eine eventuell
    vorhandene Zeile mit gleichem Schlüssel (key_cols). Wiederholte Läufe sind
    dadurch gefahrlos – kein Duplikat, kein manuelles Aufräumen.
    """
    base = base or os.path.join(REPO_ROOT, "results")
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, filename)
    new = pd.DataFrame([row])

    if os.path.exists(path):
        df = pd.read_csv(path)
        for col in new.columns:
            if col not in df.columns:
                df[col] = pd.NA
        for col in df.columns:
            if col not in new.columns:
                new[col] = pd.NA
        new = new[df.columns]
        key_series = pd.Series({k: row[k] for k in key_cols})
        mask = (df[key_cols].astype(str) == key_series.astype(str)).all(axis=1)
        df = df[~mask]
        df = pd.concat([df, new], ignore_index=True)
    else:
        df = new

    df.to_csv(path, index=False)
    return path


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


# ── Modelle laden ─────────────────────────────────────────────────────────────
def load_lilt_teacher(dataset_name: str) -> torch.nn.Module:
    """Lädt das fine-getunte LiLT Teacher-Modell (LiLT-TextFlow)."""
    run_name = TEACHER_RUN_NAMES[dataset_name]
    ds_conf  = DATASET_CONF[dataset_name]
    device   = torch.device("cpu")

    print(f"  Lade Teacher-Checkpoint: {run_name}")
    chk_path   = ENV.MODELS_DIR / run_name / "best.pth"
    checkpoint = torch.load(chk_path, map_location=device)

    model = get_model(
        model_type=MODEL_TYPE,
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


def load_lilt_student(dataset_name: str) -> torch.nn.Module:
    """
    Lädt das distillierte 4-Layer-LiLT-Studentenmodell (volles Vokabular).
    Metadaten (is_student, student_layer_map, vocab_name) kommen aus dem
    Checkpoint, wie in SlimDocs evaluate_model. vocab_name=None -> kein
    Vokabular-Lookup nötig.
    """
    run_name = STUDENT_RUN_NAMES[dataset_name]
    ds_conf  = DATASET_CONF[dataset_name]
    device   = torch.device("cpu")

    print(f"  Lade Student-Checkpoint: {run_name}")
    chk_path   = ENV.MODELS_DIR / run_name / "best.pth"
    checkpoint = torch.load(chk_path, map_location=device)

    is_student        = checkpoint["is_student"]
    student_layer_map = checkpoint["student_layer_map"]
    vocab_name        = checkpoint["vocab_name"]
    assert is_student, "Checkpoint ist kein Studentenmodell."
    assert vocab_name is None, (
        f"vocab_name={vocab_name!r}: reduziertes Vokabular wird hier nicht "
        f"unterstützt (Vokabular-Lookup fehlt in der Eval)."
    )

    model = get_model(
        model_type=MODEL_TYPE,
        task=ds_conf.task,
        is_student=is_student,
        num_labels=ds_conf.num_labels,
        vocab_map=None,
        device=device,
        teacher_run_name=None,
        student_layer_map=student_layer_map,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Student geladen. Task: {ds_conf.task} | "
          f"Layer: {model.config.num_hidden_layers} | layer_map: {student_layer_map}")
    return model


def _load_model(model_kind: str, dataset_name: str) -> torch.nn.Module:
    if model_kind == "teacher":
        return load_lilt_teacher(dataset_name)
    elif model_kind == "student":
        return load_lilt_student(dataset_name)
    raise ValueError(f"Unbekannte model_kind: {model_kind!r}")


# ── Quantisierung ─────────────────────────────────────────────────────────────
def quantize_lilt(model: torch.nn.Module) -> torch.nn.Module:
    """
    Manuelle, explizite dynamische INT8-PTQ für LiLT.

    Quantisiert:
      - alle Encoder-Linear BEIDER Flows (du_model.encoder.layer.*): dynamisch
        INT8, Gewichte symmetrisch/per-channel, Aktivierungen affin/per-tensor.
        Erfasst Text-Flow (768) UND Layout-Flow (192). Teacher -> 144, Student -> 48.
      - word_embeddings: weight-only INT8.

    Bleibt FP32 (automatisch durch den startswith-Filter ausgeschlossen):
      box_linear_embeddings, pooler.dense, classifier, alle LayerNorm, alle
      Biases, sowie die Layout-/Positions-Embeddings. BiACM ist keine Linear-
      Schicht und bleibt ohnehin FP32.
    """
    torch.backends.quantized.engine = "fbgemm"  # x86-Server-Backend
    model = model.cpu().eval()

    def _set_module(root, dotted_name, new_module):
        *parents, leaf = dotted_name.split(".")
        obj = root
        for p in parents:
            obj = getattr(obj, p)
        setattr(obj, leaf, new_module)

    # 1) Encoder-Linear beider Flows -> dynamisch quantisiert
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

    print(f"  Quantisiert: {len(linear_targets)} Encoder-Linear (Text+Layout) "
          f"+ word_embeddings")
    return model


# ── Score + Größe (pro Dataset) ───────────────────────────────────────────────
def run(dataset_name: str, model_kind: str = "teacher", quantize: bool = True):
    model_label = MODEL_LABELS[model_kind]
    print("\n" + "=" * 60)
    print(f"  LiLT PTQ | {model_label} | {dataset_name} | quantize={quantize}")
    print("=" * 60)

    ds_conf = DATASET_CONF[dataset_name]

    print(f"\n[1] Lade {model_kind}-Modell...")
    model = _load_model(model_kind, dataset_name)
    size_base = get_model_size_mb(model)
    print(f"  ✓ Modell geladen | FP32-Größe: {size_base:.1f} MB")

    if quantize:
        print("\n[2] Quantisiere Modell (dynamisch INT8)...")
        model_eval = quantize_lilt(model)
        precision = "int8"
    else:
        print("\n[2] Überspringe Quantisierung (FP32-Baseline)...")
        model_eval = model
        precision = "fp32"
    size_eval = get_model_size_mb(model_eval)
    print(f"  Größe ({precision}): {size_eval:.1f} MB "
          f"({size_base / size_eval:.2f}x ggü. eigener FP32-Größe)")

    print("\n[3] Evaluiere Modell (Score)...")
    results = evaluate_quantized_model(
        model=model_eval,
        dataset_name=dataset_name,
        task=ds_conf.task,
        model_type=MODEL_TYPE,
        split="test",
        batch_size=16,
    )

    print("\n[4] Ergebnisse:")
    print(f"  Score:   {results['score']:.4f}")
    print(f"  Samples: {results['total_samples']}")
    print(f"  Größe:   {size_eval:.1f} MB")

    upsert_row("accuracy_size.csv", {
        "model":     model_label,
        "precision": precision,
        "dataset":   dataset_name,
        "task":      str(ds_conf.task),
        "score":     round(results["score"], 4),
        "size_mb":   round(size_eval, 1),
        "n_samples": results["total_samples"],
    }, key_cols=["model", "precision", "dataset"])


# ── Latenz (genau einmal) ─────────────────────────────────────────────────────
def run_latency(dataset_name: str, model_kind: str = "teacher",
                quantize: bool = True, num_threads: int = 8):
    model_label = MODEL_LABELS[model_kind]
    print("\n" + "=" * 60)
    print(f"  Latenz-Benchmark | {model_label} | quantize={quantize} | "
          f"threads={num_threads}")
    print("=" * 60)

    model = _load_model(model_kind, dataset_name)
    if quantize:
        model = quantize_lilt(model)
        precision = "int8"
    else:
        precision = "fp32"

    lat = benchmark_latency(model, MODEL_TYPE, num_threads=num_threads)
    print(f"  {lat['batch_ms_mean']:.1f} ± {lat['batch_ms_std']:.1f} ms/Batch | "
          f"{lat['sample_ms_mean']:.2f} ms/Sample | "
          f"{lat['throughput_samples_s']:.1f} Samples/s | "
          f"{lat['num_threads']} Threads")

    upsert_row("latency.csv", {
        "model":     model_label,
        "precision": precision,
        "cpu":       _cpu_name(),
        **lat,
    }, key_cols=["model", "precision"])


# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-Training Quantisierung für LiLT (Teacher/Student)",
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

    for dataset in datasets:
        run(dataset, model_kind=args.model, quantize=quantize)

    run_latency(datasets[0], model_kind=args.model, quantize=quantize,
                num_threads=args.threads)