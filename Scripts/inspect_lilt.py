import argparse, os, sys
import torch, torch.nn as nn
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIMDOC_PATH = "/data/stud/2026-richtstein-ba/slimdoc-main"
sys.path.insert(0, SLIMDOC_PATH); sys.path.insert(0, REPO_ROOT)

from slimdoc import DATASET_CONF, ENV, DUModel
from slimdoc.model import get_model


def main(run_name, dataset):
    ds_conf = DATASET_CONF[dataset]
    device = torch.device("cpu")
    chk = torch.load(ENV.MODELS_DIR / run_name / "best.pth", map_location=device)

    model = get_model(
        model_type=DUModel.LiLT_TextFlow,   # ggf. LiLT_TextAndLayoutFlow
        task=ds_conf.task, is_student=False, num_labels=ds_conf.num_labels,
        vocab_map=None, device=device, teacher_run_name=None, student_layer_map=None,
    )
    model.load_state_dict(chk["model_state_dict"]); model.eval()

    print("\n=== Modultypen ===")
    print(Counter(type(m).__name__ for _, m in model.named_modules()))
    print("\n=== Linear / Embedding / LayerNorm: Typ | Pfad | Form ===")
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            w = getattr(m, "weight", None)
            print(f"{type(m).__name__:11s} {name:60s} {tuple(w.shape) if w is not None else None}")
    print("\n=== Checkpoint-Keys ===")
    print(list(chk.keys()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONF.keys()))
    args = p.parse_args()
    main(args.run_name, args.dataset)