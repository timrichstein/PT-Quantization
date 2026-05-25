"""
utils/model_utils.py

Gemeinsame Hilfsfunktionen für alle Quantisierungsskripte.
Wird von scripts/ptq_dynamic_int8.py, scripts/ptq_static_int8.py etc. verwendet.
"""

import os
import sys
import tempfile
import torch

SLIMDOC_PATH = "/data/stud/2026-richtstein-ba/slimdoc-main"
if SLIMDOC_PATH not in sys.path:
    sys.path.insert(0, SLIMDOC_PATH)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Misst die Modellgröße in MB durch temporäres Speichern des state_dict.
    Funktioniert für FP32 und INT8 Modelle.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        size_mb = os.path.getsize(f.name) / (1024 * 1024)
    os.unlink(f.name)
    return size_mb


def load_teacher_model(run_name, task, num_labels, model_type, device):
    """
    Lädt einen fine-getunten Teacher-Checkpoint vom SlimDoc-Framework.

    Args:
        run_name:   Name des Runs (= Ordnername unter ENV.MODELS_DIR)
        task:       TASKS.SER oder TASKS.VQA
        num_labels: Anzahl der Label-Klassen
        model_type: DUModel-Enum
        device:     torch.device

    Returns:
        model:      Geladenes float32-Modell im eval()-Modus
        checkpoint: Der vollständige Checkpoint-Dict

    Raises:
        FileNotFoundError: Wenn der Checkpoint nicht gefunden wird
    """
    from slimdoc import ENV
    from slimdoc.model import get_model

    chk_path = ENV.MODELS_DIR / run_name / "best.pth"

    if not chk_path.exists():
        raise FileNotFoundError(
            f"Checkpoint nicht gefunden unter: {chk_path}\n"
            f"Verfügbare Modelle: {[d.name for d in ENV.MODELS_DIR.iterdir() if d.is_dir()]}"
        )

    checkpoint = torch.load(chk_path, map_location=device)

    model = get_model(
        model_type=model_type,
        task=task,
        is_student=False,
        num_labels=num_labels,
        vocab_map=None,
        device=device,
        teacher_run_name=None,
        student_layer_map=None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def save_quantized_model(model, run_name, quantization_type="dynamic_int8", save_base_dir="models/quantized"):
    """
    Speichert ein quantisiertes Modell lokal.

    Args:
        model:             Das quantisierte Modell
        run_name:          Basis-Run-Name
        quantization_type: z.B. "dynamic_int8", "static_int8"
        save_base_dir:     Basisordner für quantisierte Modelle

    Returns:
        save_dir: Pfad zum gespeicherten Modell
    """
    save_dir = os.path.join(save_base_dir, f"{run_name}_{quantization_type}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"model_{quantization_type}.pt"))
    return save_dir