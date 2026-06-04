import torch
import torch.nn as nn
# Pfade gelten für aktuelle torch-Versionen (torch.ao.*).
# Bei sehr altem torch hieße es torch.nn.quantized.* statt torch.ao.nn.quantized.*
from torch.ao.nn.quantized.dynamic import Linear as DynQLinear
from torch.ao.nn.quantized import Embedding as QEmbedding
from torch.ao.quantization import (
    per_channel_dynamic_qconfig,       # Gewichte: symmetrisch, per-channel
    float_qparams_weight_only_qconfig, # Embeddings: weight-only
)


def quantize_layoutlmv3(model: torch.nn.Module) -> torch.nn.Module:
    """
    Manuelle, explizite dynamische INT8-PTQ.

    Quantisiert:
      - alle 72 Encoder-Linear (du_model.encoder.layer.*): dynamisch INT8,
        Gewichte symmetrisch/per-channel, Aktivierungen affin/per-tensor
        (zur Laufzeit pro Batch).
      - word_embeddings: weight-only INT8.

    Bleibt FP32:
      rel_pos_*_bias, classifier, alle LayerNorm, patch_embed.proj (Conv2d),
      alle Biases, sowie die kleinen Layout-/Positions-Embeddings.
    """
    torch.backends.quantized.engine = "fbgemm"  # x86-Server-Backend
    model = model.cpu().eval()

    def _set_module(root, dotted_name, new_module):
        *parents, leaf = dotted_name.split(".")
        obj = root
        for p in parents:           # funktioniert auch durch ModuleList-Indizes ("0", "1", ...)
            obj = getattr(obj, p)
        setattr(obj, leaf, new_module)

    # 1) Encoder-Linear -> dynamisch quantisiert
    linear_targets = [
        (n, m) for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n.startswith("du_model.encoder.layer.")
    ]
    for name, module in linear_targets:
        module.qconfig = per_channel_dynamic_qconfig
        _set_module(model, name, DynQLinear.from_float(module))

    # 2) word_embeddings -> weight-only quantisiert
    emb_name = "du_model.embeddings.word_embeddings"
    emb = dict(model.named_modules())[emb_name]
    emb.qconfig = float_qparams_weight_only_qconfig
    _set_module(model, emb_name, QEmbedding.from_float(emb))

    print(f"  Quantisiert: {len(linear_targets)} Encoder-Linear + word_embeddings "
          f"(erwartet 72 Linear)")
    return model