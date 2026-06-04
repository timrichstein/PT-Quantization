# scripts/inspect_model.py
import argparse
import torch
import torch.nn as nn
from collections import Counter

# Wiederverwendung deiner Ladefunktion (kein argparse, da unter __main__ gekapselt)
from layoutlmv3_ptq import load_layoutlmv3_teacher, TEACHER_RUN_NAMES
from slimdoc import ENV


def main(dataset: str):
    model = load_layoutlmv3_teacher(dataset)

    print("\n=== (1) Modultypen + Häufigkeit ===")
    print(Counter(type(m).__name__ for _, m in model.named_modules()))

    print("\n=== (2) Relevante Layer: Typ | Pfad | Gewichtsform ===")
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Embedding, nn.Conv2d, nn.LayerNorm)):
            w = getattr(m, "weight", None)
            shape = tuple(w.shape) if w is not None else None
            print(f"{type(m).__name__:12s} {name:60s} {shape}")

    print("\n=== (3) Checkpoint-Struktur ===")
    chk_path = ENV.MODELS_DIR / TEACHER_RUN_NAMES[dataset] / "best.pth"
    ckpt = torch.load(chk_path, map_location="cpu")
    print("type:", type(ckpt))
    if isinstance(ckpt, dict):
        print("keys:", list(ckpt.keys()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="FUNSD", choices=list(TEACHER_RUN_NAMES.keys()))
    args = p.parse_args()
    main(args.dataset)