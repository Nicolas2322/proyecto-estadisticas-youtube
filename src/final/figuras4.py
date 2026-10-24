"""Figuras del informe Avances IV. Uso: python figuras4.py <carpeta_repo> <resultados_finales.json> <carpeta_salida>"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO, RJ, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"],
                     "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False})
C_INI, C_MID, C_FIN, C_GR = "#e07a5f", "#6fa3c4", "#2a6f97", "#bfcfdb"
R = json.load(open(RJ)); A2 = json.load(open(REPO / "reports/resultados.json"))
A3 = json.load(open(REPO / "reports/avances3/resultados_a.json")); B3 = json.load(open(REPO / "reports/avances3/resultados_b.json"))["RES"]
T3 = json.load(open(REPO / "reports/avances3/resultados_transformer.json"))
EA, EB = R["etapa_A"], R["etapa_B"]["con_sentimiento_roberta"]

# Figura 1: evolución del desempeño a lo largo del proyecto
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
pa = [("VADER (léxico)\nreferencia", A2["stageA"]["VADER (léxico)"]["f1_macro"][0], 0, C_GR),
      ("SVM TF-IDF\nAvances II", A2["stageA"]["SVM lineal (word+char)"]["f1_macro"][0], A2["stageA"]["SVM lineal (word+char)"]["f1_macro"][1], C_INI),
      ("SVM híbrido A4\nAvances III", A3["A4 + Ajuste anidado de umbrales por clase"]["f1_macro"][0], A3["A4 + Ajuste anidado de umbrales por clase"]["f1_macro"][1], C_MID),
      ("RoBERTa\nzero-shot", T3["zero_shot"]["f1_macro"], float(np.std(T3["zero_shot"]["f1_macro_folds"])), C_MID),
      ("RoBERTa ajustado\nsolución final", EA["f1_macro"], EA["f1_sd"], C_FIN)]
x = np.arange(len(pa)); ax[0].bar(x, [p[1] for p in pa], yerr=[p[2] for p in pa], color=[p[3] for p in pa], width=.65)
for i, p in enumerate(pa): ax[0].text(i, p[1] + p[2] + .012, f"{p[1]:.3f}", ha="center", fontsize=8)
ax[0].set_xticks(x, [p[0] for p in pa], fontsize=7.5); ax[0].set_ylim(.4, 1); ax[0].set_ylabel("F1-macro (validación cruzada, 5 folds)")
ax[0].set_title("Etapa A: sentimiento del comentario")
pb = [("Solo metadatos\n(XGBoost)", A2["stageB_reg"]["XGBoost | Solo metadatos (categoría + antigüedad)"]["r2"], C_GR),
      ("XGBoost\nAvances II", A2["stageB_reg"][A2["bestB_reg"]]["r2"], C_INI),
      ("Stacking\nAvances III", B3["B8b Stacking (meta-modelo RidgeCV)"]["r2"], C_MID),
      ("Stacking + RoBERTa\nsolución final", [EB["r2"], EB["r2_sd"]], C_FIN)]
x = np.arange(len(pb)); ax[1].bar(x, [p[1][0] for p in pb], yerr=[p[1][1] for p in pb], color=[p[2] for p in pb], width=.6)
for i, p in enumerate(pb): ax[1].text(i, p[1][0] + p[1][1] + .006, f"{p[1][0]:.3f}", ha="center", fontsize=8)
ax[1].set_xticks(x, [p[0] for p in pb], fontsize=7.5); ax[1].set_ylim(.6, .85); ax[1].set_ylabel("R² sobre log10(vistas) (5 folds)")
ax[1].set_title("Etapa B: vistas del video")
plt.tight_layout(); plt.savefig(OUT / "h1_evolucion.png"); plt.close()

# Figura 2: F1 por clase, diseño inicial vs final, y matriz de confusión final
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
cls = ["Negativo", "Neutral", "Positivo"]; ini = [A2["bestA_report"][c]["f1-score"] for c in cls]
fin = [EA["reporte"][str(i)]["f1-score"] for i in range(3)]; x = np.arange(3); w = .38
ax[0].bar(x - w / 2, ini, w, color=C_INI, label="Diseño inicial (SVM TF-IDF)"); ax[0].bar(x + w / 2, fin, w, color=C_FIN, label="Solución final (RoBERTa ajustado)")
for i in range(3): ax[0].text(x[i] + w / 2, fin[i] + .015, f"+{fin[i] - ini[i]:.3f}", ha="center", fontsize=8)
ax[0].set_xticks(x, cls); ax[0].set_ylim(0, 1.1); ax[0].set_ylabel("F1 por clase"); ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
ax[0].set_title("Mejora por clase")
M = np.array(EA["matriz"]); ax[1].imshow(M, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3): ax[1].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] > .5 else "black")
ax[1].set_xticks(range(3), ["Neg", "Neu", "Pos"]); ax[1].set_yticks(range(3), ["Neg", "Neu", "Pos"]); ax[1].set_xlabel("Predicho"); ax[1].set_ylabel("Real")
ax[1].set_title("Matriz de confusión normalizada (final)", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "h2_clases.png"); plt.close()
print("ok")
