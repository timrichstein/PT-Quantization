"""
eval/evaluate.py

Entkoppelte Evaluierungsfunktion für quantisierte Modelle.
Unterstützt LiLT und LayoutLMv3, SER- und VQA-Tasks.

Misst:
  - Score (weighted F1 für SER, ANLS für VQA)
  - Forward-Pass-Latenz pro Sample (ms)
  - Throughput (Samples/Sekunde)

Identische Evaluierungslogik wie das SlimDoc-Framework (eval.py),
damit die Ergebnisse direkt mit Teacher- und Student-Modellen
vergleichbar sind. Einziger Unterschied: das Modell wird direkt
übergeben statt aus einem Checkpoint geladen zu werden (notwendig
da INT8 state_dicts nicht mit FP32-Modell-Skeletten kompatibel sind).
"""

import sys
import time
import numpy as np
import torch
from sklearn.metrics import f1_score

SLIMDOC_PATH = "/data/stud/2026-richtstein-ba/slimdoc-main"
if SLIMDOC_PATH not in sys.path:
    sys.path.insert(0, SLIMDOC_PATH)

from slimdoc import TASKS, DUModel
from slimdoc.data.hf_dataset import load_dataset
from slimdoc.data.utils import create_dataloader
from slimdoc.eval.due_eval import evaluate_due_results
from slimdoc.model import forward
from slimdoc.train.utils import extract_text_logits


def evaluate_quantized_model(
    model: torch.nn.Module,
    dataset_name: str,
    task: str,
    model_type: DUModel,
    split: str = "test",
    batch_size: int = 16,
) -> dict:
    """
    Evaluiert ein quantisiertes Modell auf einem Dataset.

    Unterstützte Modelltypen:
        DUModel.LiLT_TextFlow
        DUModel.LiLT_TextAndLayoutFlow
        DUModel.LayoutLMv3_TextAndImage
        DUModel.LayoutLMv3_TextOnly

    Unterstützte Tasks:
        TASKS.SER  → weighted F1-Score
        TASKS.VQA  → ANLS

    Args:
        model:        Quantisiertes Modell-Objekt (INT8)
        dataset_name: z.B. "FUNSD", "SROIE", "DocVQA"
        task:         TASKS.SER oder TASKS.VQA
        model_type:   DUModel-Enum
        split:        "test" (Standard) oder "dev"
        batch_size:   Batch-Größe für Inferenz

    Returns:
        dict mit:
            score                          – F1 oder ANLS (0–1)
            avg_forward_pass_ms            – Durchschn. Forward-Pass-Zeit pro Batch (ms)
            avg_forward_pass_per_sample_ms – Forward-Pass-Zeit pro Sample (ms)
            throughput_samples_s           – Samples pro Sekunde
            total_samples                  – Anzahl evaluierter Samples
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Dataset laden (identisch zu SlimDoc eval.py)
    dataset = load_dataset([dataset_name], use_cache=True, use_chatgpt_labels=False)
    test_dataloader = create_dataloader(
        dataset[split],
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,
    )

    total_samples = len(dataset[split])
    sample_ids_preds = []
    forward_pass_times_ms = []

    with torch.no_grad():
        for inputs in test_dataloader:
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            text_seq_length = inputs["input_ids"].shape[1]

            # LayoutLMv3 benötigt pixel_values, LiLT nicht
            pixel_values = (
                inputs["pixel_values"]
                if model_type == DUModel.LayoutLMv3_TextAndImage
                else None
            )

            # ── Zeitmessung: nur der Forward Pass ────────────────────────────
            start = time.perf_counter()

            outputs = forward(
                model=model,
                model_type=model_type,
                output_internals=False,
                input_ids=inputs["input_ids"],
                bbox=inputs["bbox"],
                attention_mask=inputs["attention_mask"],
                pixel_values=pixel_values,
            )

            end = time.perf_counter()
            forward_pass_times_ms.append((end - start) * 1000)

            labels = inputs["labels"]

            # Predictions extrahieren (identisch zu SlimDoc eval.py)
            if task == TASKS.SER:
                logits = extract_text_logits(
                    model_type=model_type,
                    logits=outputs["logits"],
                    text_seq_length=text_seq_length,
                    task=task,
                )
                predictions = torch.argmax(logits, dim=-1)

            elif task == TASKS.VQA:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    "SCUT-DLVCLab/lilt-roberta-en-base"
                )
                start_logits = extract_text_logits(
                    model_type=model_type,
                    logits=outputs["start_logits"],
                    text_seq_length=text_seq_length,
                    task=task,
                )
                end_logits = extract_text_logits(
                    model_type=model_type,
                    logits=outputs["end_logits"],
                    text_seq_length=text_seq_length,
                    task=task,
                )
                predicted_start = torch.argmax(start_logits, dim=1)
                predicted_end = torch.argmax(end_logits, dim=1)
                selected_ranges = [
                    inputs["input_ids"][i, start:end]
                    for i, (start, end) in enumerate(
                        zip(predicted_start, predicted_end)
                    )
                ]
                predictions = tokenizer.batch_decode(selected_ranges)

            sample_ids_preds.append(
                (inputs["sample_id"], inputs["dataset_name"], predictions, labels)
            )

    # ── Score berechnen (identisch zu SlimDoc eval.py __eval_final) ──────────
    if task == TASKS.SER:
        ser_scores = []
        for sample_ids, dataset_names, predictions, labels in sample_ids_preds:
            for sample_id, ds_name, prediction, label in zip(
                sample_ids, dataset_names, predictions, labels
            ):
                pred_flat = prediction.view(-1).cpu().numpy()
                label_flat = label.view(-1).cpu().numpy()
                mask = label_flat != -100
                f1 = f1_score(
                    label_flat[mask], pred_flat[mask], average="weighted"
                )
                ser_scores.append(f1)
        score = float(np.mean(ser_scores))

    elif task == TASKS.VQA:
        due_predictions = {}
        for sample_ids, dataset_names, predictions, labels in sample_ids_preds:
            for sample_id, ds_name, prediction, label in zip(
                sample_ids, dataset_names, predictions, labels
            ):
                due_predictions[sample_id] = prediction
        score = evaluate_due_results(
            dataset_name=dataset_name,
            split=split,
            predictions=due_predictions,
            only_our_samples=True,
        )

    # ── Zeitmessung auswerten ─────────────────────────────────────────────────
    avg_batch_ms = float(np.mean(forward_pass_times_ms))
    avg_sample_ms = avg_batch_ms / batch_size
    throughput = (1000 / avg_sample_ms) if avg_sample_ms > 0 else 0.0

    return {
        "score":                           score,
        "avg_forward_pass_ms":             avg_batch_ms,
        "avg_forward_pass_per_sample_ms":  avg_sample_ms,
        "throughput_samples_s":            throughput,
        "total_samples":                   total_samples,
    }


def benchmark_latency(
    model: torch.nn.Module,
    model_type: DUModel,
    batch_size: int = 16,
    seq_len: int = 512,
    n_warmup: int = 10,
    n_measure: int = 100,
    n_repeats: int = 10,
    num_threads: int = 8,
) -> dict:
    """
    Latenz-Benchmark nach SlimDoc-Protokoll (Abschnitt 4.4.6), auf CPU,
    mit synthetischen Inputs fester Form. Misst NUR den Forward-Pass.

    Warum synthetisch: Die Forward-Pass-Zeit hängt nur von Architektur,
    Präzision, Sequenzlänge, Batch-Größe und Hardware ab – nicht vom Inhalt.
    Synthetische 512-Token-Inputs sind daher zeitlich äquivalent zu echten
    DocVQA-Samples und vollständig reproduzierbar.

    Args:
        model:       Zu vermessendes Modell (FP32 oder INT8)
        model_type:  DUModel-Enum
        batch_size:  16 (wie SlimDoc)
        seq_len:     512 (max. Sequenzlänge, wie SlimDoc)
        n_warmup:    10 Warmup-Batches (wie SlimDoc)
        n_measure:   100 Batches pro Wiederholung (wie SlimDoc)
        n_repeats:   10 Wiederholungen (wie SlimDoc)
        num_threads: Feste, zu berichtende CPU-Threadzahl

    Returns:
        dict mit batch_ms_mean/std, sample_ms_mean, throughput_samples_s,
        num_threads, batch_size, seq_len
    """
    torch.backends.quantized.engine = "fbgemm"
    torch.set_num_threads(num_threads)
    device = torch.device("cpu")
    model = model.to(device).eval()

    VOCAB = 50265
    input_ids      = torch.randint(0, VOCAB, (batch_size, seq_len), device=device)
    bbox           = torch.randint(0, 1000, (batch_size, seq_len, 4), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    pixel_values   = (
        torch.rand(batch_size, 3, 224, 224, device=device)
        if model_type == DUModel.LayoutLMv3_TextAndImage else None
    )

    def _fwd():
        return forward(
            model=model, model_type=model_type, output_internals=False,
            input_ids=input_ids, bbox=bbox,
            attention_mask=attention_mask, pixel_values=pixel_values,
        )

    with torch.no_grad():
        for _ in range(n_warmup):
            _fwd()
        per_repeat_ms = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            for _ in range(n_measure):
                _fwd()
            per_repeat_ms.append((time.perf_counter() - t0) * 1000.0 / n_measure)

    arr = np.array(per_repeat_ms)
    return {
        "batch_ms_mean":        float(arr.mean()),
        "batch_ms_std":         float(arr.std()),
        "sample_ms_mean":       float(arr.mean() / batch_size),
        "throughput_samples_s": float(batch_size / (arr.mean() / 1000.0)),
        "num_threads":          torch.get_num_threads(),
        "batch_size":           batch_size,
        "seq_len":              seq_len,
    }