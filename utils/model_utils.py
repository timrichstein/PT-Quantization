"""
utils/model_utils.py

Gemeinsame Hilfsfunktionen für alle Quantisierungsskripte.

Enthält:
  - get_model_size_mb(): Modellgröße in MB messen
  - save_quantized_model(): Quantisiertes Modell lokal speichern
"""

import os
import tempfile
import torch


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Misst die Modellgröße in MB anhand des model_state_dict.

    Speichert den model_state_dict temporär als Datei und misst die
    Dateigröße. Diese Methode misst nur die reinen Modellgewichte –
    ohne Optimizer-State, Epoch-Information oder andere Metadaten.

    Warum nicht den vollständigen Checkpoint messen?
    Der vollständige Checkpoint (best.pth) enthält zusätzlich den
    Optimizer-State (Adam speichert Gradienten und Momentschätzer),
    der bei einem quantisierten Modell nicht existiert. Daher wäre
    ein Vergleich von Checkpoint-Größen unfair. Die model_state_dict-
    Größe ist für alle Modelle (FP32, INT8, Student) vergleichbar.

    Args:
        model: PyTorch-Modell (FP32 oder quantisiert)

    Returns:
        Modellgröße in Megabyte (MB)
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        size_mb = os.path.getsize(f.name) / (1024 * 1024)
    os.unlink(f.name)
    return size_mb


def save_quantized_model(
    model: torch.nn.Module,
    run_name: str,
    quantization_type: str,
    save_base_dir: str = "models/quantized",
) -> str:
    """
    Speichert ein quantisiertes Modell lokal als .pt-Datei.

    Args:
        model:             Das quantisierte Modell
        run_name:          Basis-Run-Name des originalen Teacher-Modells
                           (z.B. 'LiLT-TextFlow_ft-teacher_funsd_50epochs')
        quantization_type: Bezeichnung der Quantisierungsmethode
                           (z.B. 'dynamic_symmetric_int8')
        save_base_dir:     Basisordner für quantisierte Modelle

    Returns:
        save_dir: Pfad zum Ordner wo das Modell gespeichert wurde
    """
    save_dir = os.path.join(save_base_dir, f"{run_name}_{quantization_type}")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"model_{quantization_type}.pt")
    torch.save(model.state_dict(), save_path)

    return save_dir