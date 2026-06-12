"""
Scripts/quant_demo.py

Demonstration der Quantisierung an echten Modellwerten – für die Arbeit.

Zeigt zwei Fälle an echten Zahlen aus LayoutLMv3:
  (A) GEWICHTE einer Linear-Schicht: symmetrisch, per-channel, qint8.
      Wir extrahieren PyTorchs gespeicherte INT8-Werte, Skalen und Zero-Points
      und rechnen sie von Hand nach (q = clamp(round(w/s), -128, 127)).
  (B) AKTIVIERUNGEN derselben Schicht: dynamisch, asymmetrisch (affin),
      per-tensor, quint8. Wir greifen die echte FP32-Aktivierung per Hook ab,
      quantisieren sie mit PyTorchs dynamischer Per-Tensor-Operation und rechnen
      Skala, Zero-Point und INT8-Werte von Hand nach.

Validierung: In beiden Fällen wird gezeigt, dass die Handrechnung exakt PyTorchs
Ergebnis reproduziert (mismatch == 0).

Ausgabe: drei CSVs in results/
  quant_weights.csv      – Wert-für-Wert-Tabelle der Gewichtsquantisierung
  quant_activations.csv  – Wert-für-Wert-Tabelle der Aktivierungsquantisierung
  quant_summary.csv      – die Parameter (Skala, Zero-Point) und Validierung

Aufruf:
    python -u Scripts/quant_demo.py
"""

import copy
import csv
import os
import sys

import torch

REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIMDOC_PATH = "/data/stud/2026-richtstein-ba/slimdoc-main"
sys.path.insert(0, SLIMDOC_PATH)
sys.path.insert(0, REPO_ROOT)

from torch.ao.nn.quantized.dynamic import Linear as DynQLinear
from torch.ao.quantization import per_channel_dynamic_qconfig

from slimdoc import DUModel
from slimdoc.data.hf_dataset import load_dataset
from slimdoc.data.utils import create_dataloader
from slimdoc.model import forward

from layoutlmv3_ptq import load_layoutlmv3_teacher

# ── Konfiguration ─────────────────────────────────────────────────────────────
DATASET   = "FUNSD"
TARGET    = "du_model.encoder.layer.0.attention.self.query"  # 768x768 Linear
ROW       = 0      # welcher Ausgabe-Kanal (per-channel: jede Zeile hat eigene Skala)
K         = 8      # wie viele Einzelwerte in die CSV-Tabelle
RESULTS   = os.path.join(REPO_ROOT, "results")


def _write_csv(name, header, rows):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  -> {path}")


# ── (A) Gewichtsquantisierung ─────────────────────────────────────────────────
def demo_weights(model):
    print("\n[A] Gewichtsquantisierung (symmetrisch, per-channel, qint8)")
    lin = dict(model.named_modules())[TARGET]
    fp32_w = lin.weight.detach().clone()                  # [out=768, in=768]

    # Genau diesen Layer isoliert quantisieren (wie im echten Skript via from_float)
    m = copy.deepcopy(lin)
    m.qconfig = per_channel_dynamic_qconfig
    qm = DynQLinear.from_float(m)

    qw = qm.weight()                                      # quantisierter Tensor
    assert qw.qscheme() in (torch.per_channel_symmetric,
                            torch.per_channel_affine), qw.qscheme()
    int8   = qw.int_repr()                                # [768,768] int8
    scales = qw.q_per_channel_scales()                    # [768] (eine pro Zeile)
    zps    = qw.q_per_channel_zero_points()               # [768] (0 bei symmetrisch)

    s   = scales[ROW].item()
    zp  = int(zps[ROW].item())

    # Herleitung der Skala: symmetrisch qint8 -> s = max(|w|) / ((127-(-128))/2)
    #                                              = max(|w|) / 127.5
    max_abs       = fp32_w[ROW].abs().max().item()
    scale_manual  = max_abs / 127.5
    scale_ratio   = scale_manual / s

    # Validierung über die GANZE Zeile: q = clamp(round(w/s)+zp, -128, 127)
    q_manual_row = torch.clamp(torch.round(fp32_w[ROW] / s) + zp, -128, 127).to(torch.int8)
    mismatches   = int((q_manual_row != int8[ROW]).sum().item())

    # Detail-Tabelle für die ersten K Werte
    rows = []
    for i in range(K):
        w      = fp32_w[ROW, i].item()
        q_pt   = int(int8[ROW, i].item())
        q_man  = int(q_manual_row[i].item())
        w_deq  = (q_pt - zp) * s
        rows.append([ROW, i, f"{w:.6f}", f"{s:.8e}", zp,
                     q_pt, q_man, f"{w_deq:.6f}", f"{abs(w - w_deq):.6f}"])

    _write_csv("quant_weights.csv",
               ["channel", "idx", "w_fp32", "scale", "zero_point",
                "q_int8_pytorch", "q_int8_manual", "w_dequant", "abs_error"],
               rows)

    print(f"    Skala (PyTorch):        {s:.8e}")
    print(f"    Skala (Handrechnung):   {scale_manual:.8e}  (Verhältnis {scale_ratio:.4f})")
    print(f"    Zero-Point:             {zp}  (0 = symmetrisch)")
    print(f"    Validierung Zeile {ROW}: {mismatches} Abweichungen von {fp32_w.shape[1]}")
    if abs(scale_ratio - 1.0) > 0.01:
        print("    ! Skala weicht >1% ab – Divisor in deiner Version pruefen (127 vs 127.5)")

    return dict(scale_pytorch=s, scale_manual=scale_manual, zero_point=zp,
                mismatches=mismatches, n=fp32_w.shape[1])


# ── (B) Aktivierungsquantisierung ─────────────────────────────────────────────
def demo_activations(model):
    print("\n[B] Aktivierungsquantisierung (dynamisch, asymmetrisch, per-tensor, quint8)")
    lin = dict(model.named_modules())[TARGET]

    captured = {}
    def hook(_mod, inp):
        captured["x"] = inp[0].detach().clone()           # FP32-Input der Schicht
    h = lin.register_forward_pre_hook(hook)

    # Ein echtes Sample durch das Modell schicken
    ds = load_dataset([DATASET], use_cache=True, use_chatgpt_labels=False)
    dl = create_dataloader(ds["test"], num_workers=0, batch_size=16, shuffle=False)
    batch = next(iter(dl))
    batch = {k: (v.to("cpu") if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}
    with torch.no_grad():
        forward(model=model, model_type=DUModel.LayoutLMv3_TextAndImage,
                output_internals=False, input_ids=batch["input_ids"],
                bbox=batch["bbox"], attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"])
    h.remove()

    x = captured["x"]                                     # [batch, seq, 768]

    # PyTorchs dynamische Per-Tensor-Quantisierung (die der dyn. Linear intern nutzt).
    # quint8, voller 8-Bit-Bereich (reduce_range=False) fuer klare Herleitung.
    xq    = torch.quantize_per_tensor_dynamic(x, torch.quint8, False)
    s     = xq.q_scale()
    zp    = int(xq.q_zero_point())
    int_u = xq.int_repr()                                 # uint8

    # Herleitung: affin, quint8 (qmin=0, qmax=255)
    #   xmin = min(x.min(), 0); xmax = max(x.max(), 0)
    #   s_manual  = (xmax - xmin) / 255
    #   zp_manual = clamp(round(0 - xmin/s_manual), 0, 255)
    xmin = min(x.min().item(), 0.0)
    xmax = max(x.max().item(), 0.0)
    scale_manual = (xmax - xmin) / 255.0
    zp_manual    = int(max(0, min(255, round(-xmin / scale_manual))))

    flat_x, flat_q = x.flatten(), int_u.flatten()
    q_manual_all = torch.clamp(torch.round(flat_x / s) + zp, 0, 255).to(torch.int32)
    mismatches   = int((q_manual_all != flat_q.to(torch.int32)).sum().item())

    rows = []
    for i in range(K):
        xv     = flat_x[i].item()
        q_pt   = int(flat_q[i].item())
        q_man  = int(q_manual_all[i].item())
        x_deq  = (q_pt - zp) * s
        rows.append([i, f"{xv:.6f}", f"{s:.8e}", zp,
                     q_pt, q_man, f"{x_deq:.6f}", f"{abs(xv - x_deq):.6f}"])

    _write_csv("quant_activations.csv",
               ["idx", "x_fp32", "scale", "zero_point",
                "q_uint8_pytorch", "q_uint8_manual", "x_dequant", "abs_error"],
               rows)

    print(f"    Tensor-Form:            {tuple(x.shape)}  (per-tensor: 1 Skala fuer alles)")
    print(f"    Skala (PyTorch):        {s:.8e}")
    print(f"    Skala (Handrechnung):   {scale_manual:.8e}")
    print(f"    Zero-Point (PyTorch):   {zp}   (!= 0 = asymmetrisch)")
    print(f"    Zero-Point (Handr.):    {zp_manual}")
    print(f"    Validierung gesamt:     {mismatches} Abweichungen von {flat_x.numel()}")

    return dict(scale_pytorch=s, scale_manual=scale_manual,
                zero_point_pytorch=zp, zero_point_manual=zp_manual,
                mismatches=mismatches, n=flat_x.numel())


# ── Hauptablauf ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.backends.quantized.engine = "fbgemm"
    print(f"Lade Modell ({DATASET}) ...")
    model = load_layoutlmv3_teacher(DATASET)

    w = demo_weights(model)
    a = demo_activations(model)

    _write_csv("quant_summary.csv",
               ["kind", "scheme", "dtype", "scale_pytorch", "scale_manual",
                "zero_point_pytorch", "zero_point_manual", "mismatches", "n_values"],
               [
                   ["weight", "symmetric_per_channel", "qint8",
                    f"{w['scale_pytorch']:.8e}", f"{w['scale_manual']:.8e}",
                    w["zero_point"], w["zero_point"], w["mismatches"], w["n"]],
                   ["activation", "affine_per_tensor", "quint8",
                    f"{a['scale_pytorch']:.8e}", f"{a['scale_manual']:.8e}",
                    a["zero_point_pytorch"], a["zero_point_manual"],
                    a["mismatches"], a["n"]],
               ])
    print("\nFertig. Drei CSVs in results/ geschrieben.")