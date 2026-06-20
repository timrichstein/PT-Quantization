"""
Scripts/make_figures.py

Erzeugt die drei Kenngrößen-Diagramme für den Ergebnisteil aus den
Ergebnis-CSVs. Schlicht-wissenschaftliche, graustufentaugliche Gestaltung.

Diagramme (nach results/figures/):
  fig_groesse.png    – Modellgröße (MB), gruppiert nach Modellvariante,
                       FP32 vs. INT8 farb-/schraffurcodiert.
  fig_latenz.png     – Latenz (ms/Batch) mit Standardabweichung als Fehlerbalken.
  fig_retention.png  – Genauigkeitserhalt nach Quantisierung (INT8/FP32 in %)
                       je Datensatz, gruppiert nach Modellvariante.

Aufruf:
    python -u Scripts/make_figures.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # kein Display auf dem Server nötig
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS   = os.path.join(REPO_ROOT, "results")
FIGDIR    = os.path.join(RESULTS, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ── Einheitliche, schlichte Gestaltung ────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",          # seriös, passt zu wiss. Texten
    "axes.spines.top": False,        # keine überflüssigen Rahmenlinien
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.axis": "y",
    "grid.alpha": 0.3,
    "figure.dpi": 150,               # ausreichend scharf fürs Einbetten
})

# Graustufentaugliche Codierung: FP32 hell + Schraffur, INT8 dunkel
STYLE = {
    "fp32": dict(color="#bdbdbd", hatch="//", edgecolor="black", label="FP32"),
    "int8": dict(color="#525252", hatch="",   edgecolor="black", label="INT8"),
}

# Reihenfolge und Anzeigenamen der vier Modellvarianten auf der x-Achse
VARIANTS = [
    ("layoutlmv3_teacher",   "LayoutLMv3\nTeacher"),
    ("layoutlmv3_student_4L","LayoutLMv3\nStudent"),
    ("lilt_teacher",         "LiLT\nTeacher"),
    ("lilt_student_4L",      "LiLT\nStudent"),
]

acc = pd.read_csv(os.path.join(RESULTS, "accuracy_size.csv"))
lat = pd.read_csv(os.path.join(RESULTS, "latency.csv"))


def _grouped_bars(ax, value_fn, err_fn=None):
    """
    Zeichnet pro Modellvariante zwei Balken (FP32, INT8) nebeneinander.
    value_fn(model, precision) -> Wert; err_fn optional -> Fehlerbalken.
    """
    x = np.arange(len(VARIANTS))
    width = 0.38
    for i, prec in enumerate(["fp32", "int8"]):
        vals = [value_fn(m, prec) for m, _ in VARIANTS]
        errs = [err_fn(m, prec) for m, _ in VARIANTS] if err_fn else None
        ax.bar(x + (i - 0.5) * width, vals, width,
               yerr=errs, capsize=4, **STYLE[prec])
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in VARIANTS])
    ax.legend(frameon=False)


# ── Diagramm 1: Modellgröße ───────────────────────────────────────────────────
def fig_groesse():
    # Größe ist datensatzunabhängig -> ein Wert je model/precision
    def size_of(model, prec):
        sub = acc[(acc["model"] == model) & (acc["precision"] == prec)]
        return sub["size_mb"].iloc[0] if len(sub) else 0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _grouped_bars(ax, size_of)
    ax.set_ylabel("Modellgröße (MB)")
    ax.set_title("Modellgröße vor und nach Quantisierung")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "fig_groesse.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {path}")


# ── Diagramm 2: Latenz ────────────────────────────────────────────────────────
def fig_latenz():
    def lat_of(model, prec):
        sub = lat[(lat["model"] == model) & (lat["precision"] == prec)]
        return sub["batch_ms_mean"].iloc[0] if len(sub) else 0
    def err_of(model, prec):
        sub = lat[(lat["model"] == model) & (lat["precision"] == prec)]
        return sub["batch_ms_std"].iloc[0] if len(sub) else 0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _grouped_bars(ax, lat_of, err_of)
    ax.set_ylabel("Latenz (ms/Batch)")
    ax.set_title("Inferenzlatenz vor und nach Quantisierung\n(Fehlerbalken: Standardabweichung)")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "fig_latenz.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {path}")


# ── Diagramm 3: Genauigkeitserhalt (Retention) ────────────────────────────────
def fig_retention():
    """
    Retention = INT8-Score / FP32-Score je (Modell, Datensatz), in Prozent.
    Gruppiert nach Modellvariante, ein Balken je Datensatz.
    """
    datasets = ["FUNSD", "SROIE", "DocVQA", "InfographicsVQA", "WikiTableQuestions"]
    # Graustufen-Palette für die fünf Datensätze
    greys = ["#252525", "#636363", "#969696", "#cccccc", "#f0f0f0"]

    x = np.arange(len(VARIANTS))
    width = 0.16
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for j, ds in enumerate(datasets):
        rets = []
        for model, _ in VARIANTS:
            fp = acc[(acc["model"] == model) & (acc["precision"] == "fp32") & (acc["dataset"] == ds)]
            iq = acc[(acc["model"] == model) & (acc["precision"] == "int8") & (acc["dataset"] == ds)]
            if len(fp) and len(iq) and fp["score"].iloc[0] > 0:
                rets.append(100.0 * iq["score"].iloc[0] / fp["score"].iloc[0])
            else:
                rets.append(np.nan)
        ax.bar(x + (j - 2) * width, rets, width,
               color=greys[j], edgecolor="black", linewidth=0.5, label=ds)

    ax.axhline(100, color="black", linewidth=0.8, linestyle="--")  # 100%-Referenz
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in VARIANTS])
    ax.set_ylabel("Genauigkeitserhalt INT8/FP32 (%)")
    ax.set_title("Genauigkeitserhalt nach Quantisierung je Datensatz")
    ax.set_ylim(80, 105)             # Ausschnitt, der die Unterschiede zeigt
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "fig_retention.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {path}")


if __name__ == "__main__":
    fig_groesse()
    fig_latenz()
    fig_retention()
    print("\nFertig. Drei Abbildungen in results/figures/")