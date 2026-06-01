"""
scripts/ptq_dynamic_int8.py

Dynamische INT8 Post-Training Quantisierung für LiLT und LayoutLMv3
Teacher-Modelle.

Misst für das quantisierte INT8-Modell:
  - Score (weighted F1 für SER, ANLS für VQA)
  - Modellgröße (MB)
  - Forward-Pass-Latenz pro Sample (ms)
  - Throughput (Samples/Sekunde)

Die FP32-Baseline-Scores liegen bereits in den JSON-Dateien des
SlimDoc-Frameworks vor und müssen nicht neu gemessen werden.

Ergebnisse werden in W&B und lokal als JSON in results/ gespeichert.

Aufruf:
    # Einzelnes Dataset, LiLT:
    python scripts/ptq_dynamic_int8.py --dataset FUNSD --model lilt

    # Alle Datasets, LayoutLMv3:
    python scripts/ptq_dynamic_int8.py --all --model layoutlmv3

    # Alle Datasets, beide Modelle:
    python scripts/ptq_dynamic_int8.py --all --model both
"""

import argparse
import json
import os
import sys
import torch
import wandb

# ── Pfade einrichten ──────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIMDOC_PATH = "/data/stud/2026-richtstein-ba/slimdoc-main"
sys.path.insert(0, SLIMDOC_PATH)
sys.path.insert(0, REPO_ROOT)

from slimdoc import DATASET_CONF, DUModel
from utils.model_utils import get_model_size_mb, load_teacher_model, save_quantized_model
from eval.evaluate import evaluate_quantized_model


# ── Konfiguration ─────────────────────────────────────────────────────────────
SUPPORTED_DATASETS = [
    "FUNSD",
    "SROIE",
    "DocVQA",
    "InfographicsVQA",
    "WikiTableQuestions",
]

TEACHER_RUN_NAMES = {
    # LiLT
    ("FUNSD",              "lilt"): "LiLT-TextFlow_ft-teacher_funsd_50epochs",
    ("SROIE",              "lilt"): "LiLT-TextFlow_ft-teacher_sroie_50epochs",
    ("DocVQA",             "lilt"): "LiLT-TextFlow_ft-teacher_docvqa_30epochs",
    ("InfographicsVQA",    "lilt"): "LiLT-TextFlow_ft-teacher_infographicsvqa_30epochs",
    ("WikiTableQuestions", "lilt"): "LiLT-TextFlow_ft-teacher_wikitablequestions_30epochs",
    # LayoutLMv3
    ("FUNSD",              "layoutlmv3"): "LayoutLMv3-TextAndImage_ft-teacher_funsd_50epochs",
    ("SROIE",              "layoutlmv3"): "LayoutLMv3-TextAndImage_ft-teacher_sroie_50epochs",
    ("DocVQA",             "layoutlmv3"): "LayoutLMv3-TextAndImage_ft-teacher_docvqa_30epochs",
    ("InfographicsVQA",    "layoutlmv3"): "LayoutLMv3-TextAndImage_ft-teacher_infographicsvqa_30epochs",
    ("WikiTableQuestions", "layoutlmv3"): "LayoutLMv3-TextAndImage_ft-teacher_wikitablequestions_30epochs",
}

MODEL_TYPE_MAP = {
    "lilt":       DUModel.LiLT_TextFlow,
    "layoutlmv3": DUModel.LayoutLMv3_TextAndImage,
}

RESULTS_DIR = os.path.join(REPO_ROOT, "results")

def fix_layoutlmv3_rel_pos_bias(model):
    """
    Nach quantize_dynamic werden rel_pos_bias Layer beschädigt da sie
    direkt als Lookup-Tabelle (weight.t()[index]) statt als normaler
    Forward-Pass verwendet werden (Zeilen 614, 638, 639 in modeling_layoutlmv3.py).
    Diese Funktion stellt die drei betroffenen Layer als FP32 wieder her.
    """
    for name, module in model.named_modules():
        for attr in ['rel_pos_bias', 'rel_2d_pos_x_bias', 'rel_2d_pos_y_bias']:
            if hasattr(module, attr):
                original = getattr(module, attr)
                new_linear = torch.nn.Linear(
                    original.in_features,
                    original.out_features,
                    bias=False
                )
                new_linear.weight = torch.nn.Parameter(
                    original.weight().dequantize()
                )
                setattr(module, attr, new_linear)
    return model


# ── Hauptfunktion ─────────────────────────────────────────────────────────────
def quantize_and_evaluate(dataset_name: str, model_key: str) -> dict:
    print("\n" + "=" * 60)
    print(f"  Dataset: {dataset_name} | Modell: {model_key.upper()}")
    print("=" * 60)

    run_name = TEACHER_RUN_NAMES.get((dataset_name, model_key))
    if run_name is None:
        print(f"  FEHLER: Kein Run-Name für ({dataset_name}, {model_key}) definiert.")
        return None

    model_type = MODEL_TYPE_MAP[model_key]
    ds_conf = DATASET_CONF[dataset_name]
    task = ds_conf.task
    num_labels = ds_conf.num_labels
    device = torch.device("cpu")

    # ── Schritt 1: Teacher-Checkpoint laden ───────────────────────────────────
    print(f"\n[1/4] Lade Teacher-Checkpoint: {run_name}")
    try:
        model_fp32, _ = load_teacher_model(
            run_name=run_name,
            task=task,
            num_labels=num_labels,
            model_type=model_type,
            device=device,
        )
    except FileNotFoundError as e:
        print(f"  FEHLER: {e}")
        return None

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

    # Fix für LayoutLMv3 rel_pos_bias Layer
    if model_key == "layoutlmv3":
        print("  Stelle rel_pos_bias Layer wieder her (LayoutLMv3-Fix)...")
        model_int8 = fix_layoutlmv3_rel_pos_bias(model_int8)
        print("  Fix angewendet.")

    int8_size = get_model_size_mb(model_int8)
    size_reduction = (1 - int8_size / fp32_size) * 100
    print(f"  INT8 Modellgröße: {int8_size:.1f} MB")
    print(f"  Größenreduktion:  {size_reduction:.1f}%")

    # ── Schritt 3: INT8 evaluieren ────────────────────────────────────────────
    print(f"\n[3/4] Evaluiere INT8-Modell auf {dataset_name}...")
    int8_results = evaluate_quantized_model(
        model=model_int8,
        dataset_name=dataset_name,
        task=task,
        model_type=model_type,
    )
    print(f"  Score:      {int8_results['score']:.4f}")
    print(f"  Latenz:     {int8_results['avg_forward_pass_per_sample_ms']:.2f} ms/Sample")
    print(f"  Throughput: {int8_results['throughput_samples_s']:.1f} Samples/s")

    # ── Schritt 4: Ergebnisse speichern ───────────────────────────────────────
    print(f"\n[4/4] Speichere Ergebnisse...")

    save_dir = save_quantized_model(
        model_int8, run_name, quantization_type="dynamic_int8"
    )
    print(f"  Modell gespeichert unter: {save_dir}/")

    result = {
        # Metadaten
        "dataset":                         dataset_name,
        "model":                           model_key,
        "run_name":                        run_name,
        "quantization":                    "dynamic_int8",

        # Score
        "int8_score":                      int8_results["score"],

        # Modellgröße
        "fp32_size_mb":                    fp32_size,
        "int8_size_mb":                    int8_size,
        "size_reduction_pct":              size_reduction,

        # Latenz und Throughput
        "int8_latency_per_sample_ms":      int8_results["avg_forward_pass_per_sample_ms"],
        "int8_throughput_samples_s":       int8_results["throughput_samples_s"],

        # Messparameter
        "total_samples":                   int8_results["total_samples"],
    }

    # Lokal als JSON speichern
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(
        RESULTS_DIR, f"ptq_dynamic_int8_{model_key}_{dataset_name}.json"
    )
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON gespeichert unter: {json_path}")

    # W&B loggen
    wandb.log(result)

    # Zusammenfassung
    print(f"\n  ── Zusammenfassung ──────────────────────────────")
    print(f"  Score:      {int8_results['score']:.4f}")
    print(f"  Größe:      FP32={fp32_size:.1f}MB → INT8={int8_size:.1f}MB  (-{size_reduction:.1f}%)")
    print(f"  Latenz:     {int8_results['avg_forward_pass_per_sample_ms']:.2f} ms/Sample")
    print(f"  Throughput: {int8_results['throughput_samples_s']:.1f} Samples/s")

    return result


# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamische INT8-PTQ für LiLT und LayoutLMv3 Teacher-Modelle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASETS,
        help="Dataset auf dem evaluiert wird.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lilt", "layoutlmv3", "both"],
        default="lilt",
        help="Welches Modell quantisiert wird.",
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

    datasets_to_run = SUPPORTED_DATASETS if args.all else [args.dataset]
    models_to_run = ["lilt", "layoutlmv3"] if args.model == "both" else [args.model]

    wandb.init(
        project="lilt-quantization-benchmark",
        entity="tim-richstein-provadis-hochschule",
        name=f"ptq-dynamic-int8-{args.model}-{'all' if args.all else args.dataset}",
        config={
            "quantization": "dynamic_int8",
            "method":       "torch.quantization.quantize_dynamic",
            "datasets":     datasets_to_run,
            "models":       models_to_run,
            "device":       "cpu",
        },
    )

    all_results = []
    for model_key in models_to_run:
        for dataset in datasets_to_run:
            result = quantize_and_evaluate(dataset, model_key)
            if result:
                all_results.append(result)

    # Gesamtzusammenfassung
    if len(all_results) > 1:
        print("\n" + "=" * 72)
        print("  GESAMTZUSAMMENFASSUNG")
        print("=" * 72)
        print(f"{'Dataset':<22} {'Modell':<12} {'Score':>8} {'Größe-':>8} {'Latenz':>10} {'Throughput':>12}")
        print(f"{'':22} {'':12} {'INT8':>8} {'Red.%':>8} {'ms/Sample':>10} {'Samples/s':>12}")
        print("-" * 72)
        for r in all_results:
            print(
                f"{r['dataset']:<22} "
                f"{r['model']:<12} "
                f"{r['int8_score']:>8.4f} "
                f"{r['size_reduction_pct']:>7.1f}% "
                f"{r['int8_latency_per_sample_ms']:>10.2f} "
                f"{r['int8_throughput_samples_s']:>12.1f}"
            )

    wandb.finish()
    print("\nFertig! Ergebnisse in W&B und results/")