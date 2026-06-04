"""
scripts/lilt_ptq.py

Post-Training Quantisierung (PTQ) für das LiLT Teacher-Modell.

Dieses Skript lädt das fine-getunte LiLT Teacher-Modell für ein gegebenes
Dataset, quantisiert es und evaluiert das quantisierte Modell.

Die Quantisierung wird granular auf Layer-Ebene durchgeführt:
  - nn.Linear Gewichte:      [WIRD NOCH DEFINIERT]
  - nn.Linear Aktivierungen: [WIRD NOCH DEFINIERT]
  - nn.Embedding Gewichte:   [WIRD NOCH DEFINIERT]
  - Bias:                    bleibt float32 – zu klein für sinnvolle Ersparnis
  - nn.LayerNorm:            bleibt float32 – sehr empfindlich auf Quantisierung

Aufruf:
    python scripts/lilt_ptq.py --dataset FUNSD
    python scripts/lilt_ptq.py --dataset SROIE
    python scripts/lilt_ptq.py --all
"""

import argparse
import os
import sys
import torch

# ── Pfade einrichten ──────────────────────────────────────────────────────────
# REPO_ROOT: Wurzelverzeichnis des PT-Quantization Repos
# SLIMDOC_PATH: Pfad zum SlimDoc-Framework für Modell- und Datenladen
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIMDOC_PATH = "/data/stud/2026-richtstein-ba/slimdoc-main"
sys.path.insert(0, SLIMDOC_PATH)
sys.path.insert(0, REPO_ROOT)

from slimdoc import DATASET_CONF, ENV, DUModel, TASKS
from slimdoc.model import get_model


# ── Konfiguration ─────────────────────────────────────────────────────────────
# Mapping: Dataset → Run-Name des fine-getunten LiLT Teacher-Modells
# Die Run-Namen entsprechen den Ordnernamen unter ENV.MODELS_DIR
TEACHER_RUN_NAMES = {
    "FUNSD":              "LiLT-TextFlow_ft-teacher_funsd_50epochs",
    "SROIE":              "LiLT-TextFlow_ft-teacher_sroie_50epochs",
    "DocVQA":             "LiLT-TextFlow_ft-teacher_docvqa_30epochs",
    "InfographicsVQA":    "LiLT-TextFlow_ft-teacher_infographicsvqa_30epochs",
    "WikiTableQuestions": "LiLT-TextFlow_ft-teacher_wikitablequestions_30epochs",
}

SUPPORTED_DATASETS = list(TEACHER_RUN_NAMES.keys())


# ── Modell laden ──────────────────────────────────────────────────────────────
def load_lilt_teacher(dataset_name: str) -> torch.nn.Module:
    """
    Lädt das fine-getunte LiLT Teacher-Modell für ein gegebenes Dataset.

    Der Ladevorgang besteht aus zwei Schritten:
    1. get_model() baut das Modell-Skelett auf – die Architektur wird von
       HuggingFace geladen (SCUT-DLVCLab/lilt-roberta-en-base), aber die
       Gewichte sind noch zufällig initialisiert bzw. vortrainiert.
    2. load_state_dict() ersetzt alle Gewichte mit den fine-getunten Werten
       aus dem lokalen Checkpoint (best.pth).

    Args:
        dataset_name: Name des Datasets (z.B. "FUNSD")

    Returns:
        model: Fine-getuntes LiLT-Modell im eval()-Modus auf CPU
    """
    run_name = TEACHER_RUN_NAMES[dataset_name]
    ds_conf  = DATASET_CONF[dataset_name]
    device   = torch.device("cpu")

    print(f"  Lade Checkpoint: {run_name}")

    # Checkpoint laden
    chk_path   = ENV.MODELS_DIR / run_name / "best.pth"
    checkpoint = torch.load(chk_path, map_location=device)

    # Schritt 1: Modell-Skelett aufbauen
    # teacher_run_name=None bedeutet: Gewichte von HuggingFace laden
    model = get_model(
        model_type=DUModel.LiLT_TextFlow,
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

    # eval()-Modus: deaktiviert Dropout und BatchNorm-Training-Verhalten
    model.eval()

    print(f"  Modell geladen. Task: {ds_conf.task}")
    return model


# ── Quantisierung ─────────────────────────────────────────────────────────────
def quantize_lilt(model: torch.nn.Module) -> torch.nn.Module:
    """
    Quantisiert das LiLT-Modell.

    [WIRD IM NÄCHSTEN SCHRITT IMPLEMENTIERT]

    Args:
        model: Fine-getuntes LiLT float32-Modell

    Returns:
        Quantisiertes Modell
    """
    raise NotImplementedError("Quantisierung wird im nächsten Schritt implementiert.")


# ── Hauptfunktion ─────────────────────────────────────────────────────────────
def run(dataset_name: str):
    """
    Führt den kompletten PTQ-Ablauf für ein Dataset durch:
    1. Modell laden
    2. Quantisieren
    3. Evaluieren
    4. Ergebnisse speichern

    Args:
        dataset_name: Name des Datasets
    """
    print("\n" + "=" * 60)
    print(f"  LiLT PTQ | Dataset: {dataset_name}")
    print("=" * 60)

    # Schritt 1: Modell laden
    print("\n[1] Lade fine-getuntes LiLT Teacher-Modell...")
    model = load_lilt_teacher(dataset_name)
    print("  ✓ Modell erfolgreich geladen")

    # Schritt 2: Quantisieren (noch nicht implementiert)
    # print("\n[2] Quantisiere Modell...")
    # model_quantized = quantize_lilt(model)

    # Schritt 3: Evaluieren (noch nicht implementiert)
    # Schritt 4: Ergebnisse speichern (noch nicht implementiert)


# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-Training Quantisierung für LiLT",
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
