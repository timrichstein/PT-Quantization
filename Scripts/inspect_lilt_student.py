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
        model_type=DUModel.LiLT_TextFlow,
        task=ds_conf.task,
        is_student=chk["is_student"],
        num_labels=ds_conf.num_labels,
        vocab_map=None,
        device=device,
        teacher_run_name=None,
        student_layer_map=chk["student_layer_map"],
    )
    model.load_state_dict(chk["model_state_dict"]); model.eval()

    n_enc_linear = sum(
        1 for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n.startswith("du_model.encoder.layer.")
    )
    n_text   = sum(1 for n, m in model.named_modules()
                   if isinstance(m, nn.Linear) and ".encoder.layer." in n and "layout" not in n)
    n_layout = sum(1 for n, m in model.named_modules()
                   if isinstance(m, nn.Linear) and ".encoder.layer." in n and "layout" in n)

    print(f"\nconfig.num_hidden_layers = {model.config.num_hidden_layers}")
    print(f"student_layer_map        = {chk['student_layer_map']}")
    print(f"Encoder-Linear gesamt    = {n_enc_linear}  (Teacher hatte 144)")
    print(f"  davon Text-Flow        = {n_text}")
    print(f"  davon Layout-Flow      = {n_layout}")
    print(f"\nModultypen: {Counter(type(m).__name__ for _, m in model.named_modules())}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONF.keys()))
    args = p.parse_args()
    main(args.run_name, args.dataset)