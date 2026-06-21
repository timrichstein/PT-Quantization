"""
Scripts/make_tables.py

Erzeugt die drei Vergleichstabellen für den Ergebnisteil aus den Ergebnis-CSVs.
Schreibt eine Excel-Datei mit drei Blättern nach results/tables.xlsx:

  Blatt 1 "Quantisierung"   – Teacher FP32 vs. Teacher INT8 (beide Modelle)
  Blatt 2 "Dist_vs_Quant"   – Teacher FP32 (Referenz), Teacher INT8, Student FP32
  Blatt 3 "Gesamt"          – alle vier Modellvarianten in beiden Präzisionen

Spalten je Tabelle: Modell | Rolle | Präzision | Größe (MB) | Latenz (ms/Batch)
                    | F1 FUNSD | F1 SROIE | ANLS DocVQA | ANLS InfoVQA | Acc WTQ

Aufruf (lokal, wenn die CSVs lokal liegen):
    python -u Scripts/make_tables.py
"""

import os
import pandas as pd

# Pfad zum results-Ordner – lokal ggf. anpassen (z.B. "results")
RESULTS = "results"
acc = pd.read_csv(os.path.join(RESULTS, "accuracy_size.csv"))
lat = pd.read_csv(os.path.join(RESULTS, "latency.csv"))

# Anzeigenamen und Reihenfolge
MODEL_INFO = {
    "layoutlmv3_teacher":    ("LayoutLMv3", "Teacher"),
    "layoutlmv3_student_4L": ("LayoutLMv3", "Student (4L)"),
    "lilt_teacher":          ("LiLT",       "Teacher"),
    "lilt_student_4L":       ("LiLT",       "Student (4L)"),
}
# Datensätze mit Anzeigespalte und Metrikname
DATASETS = [
    ("FUNSD",              "F1 FUNSD"),
    ("SROIE",              "F1 SROIE"),
    ("DocVQA",             "ANLS DocVQA"),
    ("InfographicsVQA",    "ANLS InfoVQA"),
    ("WikiTableQuestions", "Acc WTQ"),
]


def _row(model, precision):
    """Baut eine Tabellenzeile für ein (model, precision)-Paar."""
    disp_model, disp_role = MODEL_INFO[model]
    row = {
        "Modell":    disp_model,
        "Rolle":     disp_role,
        "Präzision": precision.upper(),
    }
    # Größe (datensatzunabhängig – irgendeine Zeile reicht)
    sub = acc[(acc["model"] == model) & (acc["precision"] == precision)]
    row["Größe (MB)"] = sub["size_mb"].iloc[0] if len(sub) else None
    # Latenz
    ls = lat[(lat["model"] == model) & (lat["precision"] == precision)]
    row["Latenz (ms)"] = round(ls["batch_ms_mean"].iloc[0], 1) if len(ls) else None
    # Leistung je Datensatz
    for ds_key, ds_col in DATASETS:
        cell = acc[(acc["model"] == model) & (acc["precision"] == precision)
                   & (acc["dataset"] == ds_key)]
        row[ds_col] = round(cell["score"].iloc[0], 4) if len(cell) else None
    return row


def build_table(spec):
    """spec: Liste von (model, precision)-Tupeln in gewünschter Zeilenreihenfolge."""
    return pd.DataFrame([_row(m, p) for m, p in spec])


# ── Tabellendefinitionen ──────────────────────────────────────────────────────
# Tabelle 1: reine Quantisierungswirkung (Teacher FP32 -> INT8)
t1 = build_table([
    ("layoutlmv3_teacher", "fp32"), ("layoutlmv3_teacher", "int8"),
    ("lilt_teacher",       "fp32"), ("lilt_teacher",       "int8"),
])

# Tabelle 2: Distillation vs. Quantisierung (Referenz + zwei Einzelverfahren)
t2 = build_table([
    ("layoutlmv3_teacher",    "fp32"),  # Referenz
    ("layoutlmv3_teacher",    "int8"),  # nur quantisiert
    ("layoutlmv3_student_4L", "fp32"),  # nur distilliert
    ("lilt_teacher",          "fp32"),
    ("lilt_teacher",          "int8"),
    ("lilt_student_4L",       "fp32"),
])

# Tabelle 3: Gesamtvergleich aller Varianten
t3 = build_table([
    ("layoutlmv3_teacher",    "fp32"), ("layoutlmv3_teacher",    "int8"),
    ("layoutlmv3_student_4L", "fp32"), ("layoutlmv3_student_4L", "int8"),
    ("lilt_teacher",          "fp32"), ("lilt_teacher",          "int8"),
    ("lilt_student_4L",       "fp32"), ("lilt_student_4L",       "int8"),
])

# ── Ausgabe als Excel mit drei Blättern ───────────────────────────────────────
out = os.path.join(RESULTS, "tables.xlsx")
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    t1.to_excel(writer, sheet_name="Quantisierung",  index=False)
    t2.to_excel(writer, sheet_name="Dist_vs_Quant",  index=False)
    t3.to_excel(writer, sheet_name="Gesamt",         index=False)

print(f"-> {out}")
for name, t in [("Quantisierung", t1), ("Dist_vs_Quant", t2), ("Gesamt", t3)]:
    print(f"\n=== {name} ===")
    print(t.to_string(index=False))