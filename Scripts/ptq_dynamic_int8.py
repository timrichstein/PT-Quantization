"""
ptq_dynamic_int8.py

Lädt die fine-getunten LiLT Teacher-Modelle, wendet dynamische INT8-Quantisierung an
und benchmarkt sie mit dem SlimDoc eval-Framework (evaluate_model).

Unterstützte Datasets: FUNSD, SROIE, DocVQA, InfographicsVQA, WikiTableQuestions

Aufruf:
    python ptq_dynamic_int8.py --dataset FUNSD
    python ptq_dynamic_int8.py --dataset SROIE
    python ptq_dynamic_int8.py --dataset DocVQA
    python ptq_dynamic_int8.py --dataset InfographicsVQA
    python ptq_dynamic_int8.py --dataset WikiTableQuestions
    python ptq_dynamic_int8.py --all   # alle 5 Datasets nacheinander
"""

import argparse
import os
import shutil
import sys
import tempfile
import torch
import wandb

# ── Pfad zu slimdoc-main hinzufügen ─────────────────────────────────────────
SLIMDOC_PATH = "/data/stud/2026-richtstein-ba/slimdoc-main"
sys.path.insert(0, SLIMDOC_PATH)

from slimdoc import DATASET_CONF, ENV, TASKS, DUModel
from slimdoc.model import get_model
from slimdoc.eval.eval import evaluate_model


# ── Mapping: Dataset → Teacher-Run-Name ──────────────────────────────────────
TEACHER_RUN_NAMES = {
    "FUNSD":              "LiLT-TextFlow_ft-teacher_funsd_50epochs",
    "SROIE":              "LiLT-TextFlow_ft-teacher_sroie_50epochs",
    "DocVQA":             "LiLT-TextFlow_ft-teacher_docvqa_30epochs",
    "InfographicsVQA":    "LiLT-TextFlow_ft-teacher_infographicsvqa_30epochs",
    "WikiTableQuestions": "LiLT-TextFlow_ft-teacher_wikitablequestions_30epochs",
}

SUPPORTED_DATASETS = list(TEACHER_RUN_NAMES.keys())


# ── Hilfsfunktion: Modellgröße in MB ─────────────────────────────────────────
def get_model_size_mb(model: torch.nn.Module) -> float:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        size_mb = os.path.getsize(f.name) / (1024 * 1024)
    os.unlink(f.name)
    return size_mb


# ── Hauptfunktion ─────────────────────────────────────────────────────────────
def quantize_and_evaluate(dataset_name: str):
    print("\n" + "=" * 60)
    print(f"  Dataset: {dataset_name}")
    print("=" * 60)

    run_name = TEACHER_RUN_NAMES[dataset_name]
    ds_conf = DATASET_CONF[dataset_name]
    task = ds_conf.task
    num_labels = ds_conf.num_labels
    device = torch.device("cpu")  # quantize_dynamic läuft nur auf CPU

    # ── Schritt 1: Teacher-Checkpoint laden ───────────────────────────────────
    print(f"\n[1/4] Lade Teacher-Checkpoint: {run_name}")
    chk_path = ENV.MODELS_DIR / run_name / "best.pth"

    if not chk_path.exists():
        print(f"  FEHLER: Checkpoint nicht gefunden unter {chk_path}")
        return None

    checkpoint = torch.load(chk_path, map_location=device)
    baseline_accuracy = checkpoint["best_accuracy"]
    print(f"  Baseline Accuracy (FP32): {baseline_accuracy:.4f}")

    # Modell mit korrekter Architektur laden
    model_fp32 = get_model(
        model_type=DUModel.LiLT_TextFlow,
        task=task,
        is_student=False,
        num_labels=num_labels,
        vocab_map=None,
        device=device,
        teacher_run_name=None,
        student_layer_map=None,
    )
    model_fp32.load_state_dict(checkpoint["model_state_dict"])
    model_fp32.eval()

    fp32_size = get_model_size_mb(model_fp32)
    print(f"  FP32 Modellgröße: {fp32_size:.1f} MB")

    # ── Schritt 2: INT8-Quantisierung ─────────────────────────────────────────
    print(f"\n[2/4] Wende INT8-Quantisierung an...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    model_int8.eval()

    int8_size = get_model_size_mb(model_int8)
    size_reduction = (1 - int8_size / fp32_size) * 100
    print(f"  INT8 Modellgröße: {int8_size:.1f} MB")
    print(f"  Größenreduktion:  {size_reduction:.1f}%")

    # ── Schritt 3: Quantisiertes Modell dauerhaft speichern ───────────────────
    print(f"\n[3/4] Speichere quantisiertes Modell...")
    save_dir = f"models/quantized/{run_name}_int8"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model_int8.state_dict(), f"{save_dir}/model_int8.pt")
    print(f"  Gespeichert unter: {save_dir}/")

    # ── Schritt 4: Benchmark mit evaluate_model() ─────────────────────────────
    # evaluate_model() parst den run_name strikt – "_int8" würde den Parser
    # zum Absturz bringen. Lösung: originalen Checkpoint temporär mit dem
    # INT8-Modell überschreiben, dann mit originalem run_name evaluieren,
    # danach Original wiederherstellen.
    print(f"\n[4/4] Starte Benchmark auf {dataset_name}...")

    backup_path = ENV.MODELS_DIR / run_name / "best_fp32_backup.pth"
    try:
        # Backup des originalen FP32-Checkpoints
        shutil.copy(chk_path, backup_path)

        # INT8-Checkpoint im SlimDoc-Format an den originalen Pfad schreiben
        torch.save({
            "epoch":               checkpoint["epoch"],
            "model_state_dict":    model_int8.state_dict(),
            "optimizer_state_dict": None,
            "best_accuracy":       baseline_accuracy,
            "is_student":          False,
            "student_layer_map":   None,
            "vocab_name":          None,
        }, chk_path)

        # Benchmark mit originalem run_name (Parser-kompatibel)
        int8_score = evaluate_model(
            run_name=run_name,
            split="test",
            batch_size=16,
        )

    finally:
        # Original immer wiederherstellen – auch bei Fehler
        if backup_path.exists():
            shutil.copy(backup_path, chk_path)
            os.remove(backup_path)
            print(f"  FP32-Checkpoint wiederhergestellt.")

    print(f"\n  Ergebnisse für {dataset_name}:")
    print(f"  FP32 Score (Baseline): {baseline_accuracy:.4f}")
    print(f"  INT8 Score:            {int8_score:.4f}")
    print(f"  Accuracy-Verlust:      {(baseline_accuracy - int8_score):.4f}")

    # ── W&B Logging ───────────────────────────────────────────────────────────
    wandb.log({
        "dataset":            dataset_name,
        "fp32_score":         baseline_accuracy,
        "int8_score":         int8_score,
        "accuracy_loss":      baseline_accuracy - int8_score,
        "fp32_size_mb":       fp32_size,
        "int8_size_mb":       int8_size,
        "size_reduction_pct": size_reduction,
    })

    return {
        "dataset":            dataset_name,
        "fp32_score":         baseline_accuracy,
        "int8_score":         int8_score,
        "accuracy_loss":      baseline_accuracy - int8_score,
        "fp32_size_mb":       fp32_size,
        "int8_size_mb":       int8_size,
        "size_reduction_pct": size_reduction,
    }


# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamische INT8-Quantisierung und Benchmark für LiLT Teacher-Modelle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASETS,
        help="Dataset auf dem quantisiert und evaluiert wird.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Alle 5 Datasets nacheinander quantisieren und evaluieren.",
    )
    return parser.parse_args()


# ── Einstiegspunkt ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    if not args.all and not args.dataset:
        print("Fehler: Bitte --dataset DATASETNAME oder --all angeben.")
        sys.exit(1)

    datasets_to_run = SUPPORTED_DATASETS if args.all else [args.dataset]

    # W&B initialisieren
    wandb.init(
        project="lilt-quantization-benchmark",
        name=f"ptq-dynamic-int8-{'all' if args.all else args.dataset}",
        config={
            "model":        "LiLT-TextFlow (fine-tuned Teacher)",
            "quantization": "dynamic-int8",
            "method":       "torch.quantization.quantize_dynamic",
            "datasets":     datasets_to_run,
        },
    )

    all_results = []
    for dataset in datasets_to_run:
        result = quantize_and_evaluate(dataset)
        if result:
            all_results.append(result)

    # ── Gesamtzusammenfassung ─────────────────────────────────────────────────
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("  GESAMTZUSAMMENFASSUNG")
        print("=" * 60)
        print(f"{'Dataset':<25} {'FP32':>8} {'INT8':>8} {'Verlust':>8} {'Reduktion':>10}")
        print("-" * 60)
        for r in all_results:
            print(
                f"{r['dataset']:<25} "
                f"{r['fp32_score']:>8.4f} "
                f"{r['int8_score']:>8.4f} "
                f"{r['accuracy_loss']:>8.4f} "
                f"{r['size_reduction_pct']:>9.1f}%"
            )

    wandb.finish()
    print("\nFertig! Ergebnisse in W&B Dashboard.")