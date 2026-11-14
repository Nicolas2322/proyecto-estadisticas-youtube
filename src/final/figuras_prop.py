"""Figuras de la propuesta del proyecto final. Uso: python figuras_prop.py <repo> <salida>"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 160, "font.size": 9, "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"],
                     "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False})
AZ, AZC, NAR, GR, TX = "#2a6f97", "#d7e6f0", "#e07a5f", "#9a9a9a", "#1f1f1f"


def caja(ax, x, y, w, h, titulo, cuerpo, fc=AZC, ec=AZ, tsize=9.4, bsize=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h - 0.035, titulo, ha="center", va="top", fontsize=tsize, weight="bold", color=TX)
    ax.text(x + w / 2, y + h - 0.095, cuerpo, ha="center", va="top", fontsize=bsize, color=TX, linespacing=1.25)


def flecha(ax, a, b, color=AZ, estilo="-|>", rad=0.0, ls="-", lw=1.2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=estilo, mutation_scale=11, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------- Figura: pipeline de datos y modelo
fig, ax = plt.subplots(figsize=(7.6, 4.9)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
w, h = 0.215, 0.33
fila1 = [("1. Adquisición", "Kaggle YouTube Statistics\n(1.881 videos, 18.409\ncomentarios) · en producción:\nYouTube Data API v3\ndel canal de la empresa"),
         ("2. Limpieza y registro", "Duplicados exactos y lógicos,\ncontadores ocultos (−1),\nfaltantes; conteos antes y\ndespués, SHA-256 y CSV\nde respaldo (reversible)"),
         ("3. Preprocesamiento", "Normalización @user / http,\ntokenización BPE (RoBERTa);\nminúsculas, URL, stopwords\ny n-gramas (1,2) para\nlos tópicos TF-IDF"),
         ("4. Características", "Sentimiento por comentario\n(RoBERTa) · VADER, TextBlob ·\nlikes y Gini de comentarios ·\n20 tópicos (SVD) · antigüedad ·\ncategoría → 32 por video")]
fila2 = [("8. Monitoreo", "Registro JSON por consulta\n(sin texto), latencia,\ndistribución de clases y de\nentradas (deriva); alerta y\nreentrenamiento"),
         ("7. Despliegue", "Modelos serializados;\ninterfaz Gradio y API HTTP;\nproyecto final: servicio\npermanente (Hugging Face\nSpaces) con versiones"),
         ("6. Evaluación", "Validación cruzada 5 folds\n(semilla 42), CV anidada,\nF1-macro, R², MAE, cobertura\nconformal; McNemar,\nWilcoxon, ablación"),
         ("5. Modelado", "Etapa A: RoBERTa-Twitter\najustado (3 clases)\nEtapa B: stacking XGBoost +\nHGB + RF + Ridge (RidgeCV)\n+ intervalo conformal 90 %")]
xs = [0.012, 0.262, 0.512, 0.762]; y1, y2 = 0.60, 0.10
for x, (t, b) in zip(xs, fila1): caja(ax, x, y1, w, h, t, b)
for x, (t, b) in zip(xs, fila2):
    caja(ax, x, y2, w, h, t, b, fc="#fbe3db" if t.startswith(("7", "8")) else AZC, ec=NAR if t.startswith(("7", "8")) else AZ)
for i in range(3):
    flecha(ax, (xs[i] + w + 0.004, y1 + h / 2), (xs[i + 1] - 0.004, y1 + h / 2))
    flecha(ax, (xs[i + 1] - 0.004, y2 + h / 2), (xs[i] + w + 0.004, y2 + h / 2))
flecha(ax, (xs[3] + w / 2, y1 - 0.004), (xs[3] + w / 2, y2 + h + 0.004))
flecha(ax, (xs[0] + w / 2, y2 + h + 0.004), (xs[0] + w / 2, y1 - 0.004), color=NAR, ls="--")
ax.text(xs[0] + w / 2 + 0.012, 0.5, "deriva o caída\n→ nuevos datos", fontsize=7.6, color=NAR, va="center", ha="left")
flecha(ax, (xs[2] + w / 2 - 0.03, y2 + h + 0.004), (xs[2] + w / 2 - 0.03, y1 - 0.004), color=NAR, ls="--")
ax.text(xs[2] + w / 2 - 0.018, 0.5, "métrica bajo el umbral\n→ volver a preprocesar", fontsize=7.6, color=NAR, va="center", ha="left")
ax.text(0.5, 0.995, "Desarrollo (Google Colab, GPU T4)", ha="center", va="top", fontsize=8.6, color=AZ, style="italic")
ax.text(0.5, 0.03, "Operación (servicio y monitoreo)", ha="center", va="bottom", fontsize=8.6, color=NAR, style="italic")
plt.tight_layout(); plt.savefig(OUT / "p1_pipeline.png"); plt.close()

# ---------- Figura: workflow del proyecto (fases × entregas)
fases = ["Entendimiento\ndel negocio", "Comprensión\nde los datos", "Preparación\nde los datos", "Modelado", "Evaluación", "Despliegue y\nmonitoreo"]
entregas = [("Avances II\n(sem. 8)", [1, 1, 1, 1, 1, 0], "Pregunta de negocio, EDA,\n43 configuraciones base"),
            ("Avances III\n(sem. 9)", [0, 0, 1, 1, 1, 0], "25 iteraciones registradas,\nregistro de limpieza"),
            ("Avances IV\n(sem. 10)", [0, 0, 0, 1, 1, 1], "RoBERTa ajustado, stacking,\nColab y demo Gradio"),
            ("Propuesta\n(17 nov.)", [1, 1, 0, 0, 1, 1], "Problema, datos, diseño,\npipeline y workflow"),
            ("Proyecto final\n(22 nov.)", [0, 1, 0, 0, 1, 1], "Servicio permanente, API,\nmonitoreo y deriva")]
fig, ax = plt.subplots(figsize=(8, 3.9))
M = np.array([e[1] for e in entregas]).T
for i in range(len(fases)):
    for j in range(len(entregas)):
        if M[i, j]:
            ax.add_patch(plt.Rectangle((j + 0.06, i + 0.12), 0.88, 0.76, color=NAR if i == 5 else AZ, alpha=.85 if i == 5 else .75))
ax.set_xlim(0, len(entregas)); ax.set_ylim(len(fases) + 0.85, -0.55)
ax.set_yticks(np.arange(len(fases)) + .5, fases, fontsize=8); ax.set_xticks([])
for j, (n, _, d) in enumerate(entregas):
    ax.text(j + .5, -0.12, n, ha="center", va="bottom", fontsize=8.6, weight="bold")
    ax.text(j + .5, len(fases) - 0.0 + 0.0, "", ha="center")
for j, (n, _, d) in enumerate(entregas):
    ax.text(j + .5, len(fases) + 0.42, d, ha="center", va="center", fontsize=7.4, color="#444")
for s in ["left", "bottom"]: ax.spines[s].set_visible(False)
ax.tick_params(length=0)
for j in range(1, len(entregas)): ax.axvline(j, color="#dddddd", lw=.8)
plt.tight_layout(); plt.savefig(OUT / "p2_workflow.png"); plt.close()

# ---------- Figura: todos los modelos de la etapa A y de la etapa B
A2 = json.load(open(REPO / "reports/resultados.json")); A3 = json.load(open(REPO / "reports/avances3/resultados_a.json"))
B3 = json.load(open(REPO / "reports/avances3/resultados_b.json"))["RES"]; T = json.load(open(REPO / "reports/avances3/resultados_transformer.json"))
F = json.load(open(REPO / "reports/avances4/resultados_finales.json"))
CORTO_A = {"A1": "A1 SVM + preprocesamiento social", "A2": "A2 SVM híbrido (léxico + categoría)", "A3a": "A3a GloVe-Twitter + reg. logística",
           "A3b": "A3b SVM híbrido + GloVe", "A4": "A4 Híbrido + umbrales anidados", "A5": "A5 Stacking de clasificadores"}
CORTO_B = {"B1": "B1 XGBoost + sentimiento mejorado", "B2": "B2 XGBoost + variables del título", "B3": "B3 XGBoost + GloVe del título",
           "B4": "B4 XGBoost + GloVe de comentarios", "B5": "B5 XGBoost sin tópicos (embeddings)", "B6": "B6 XGBoost con selección SHAP",
           "B7": "B7 XGBoost con Optuna (CV anidada)", "B8a": "B8a Ensamble por promedio", "B8b": "B8b Stacking (RidgeCV)",
           "B8 · HistGradientBoosting (mismas variables)": "B8 HistGradientBoosting", "B8 · Random Forest (mismas variables)": "B8 Random Forest",
           "B8 · Ridge (mismas variables)": "B8 Ridge"}
ea = [(k, v["f1_macro"][0], "II") for k, v in A2["stageA"].items()]
ea += [(CORTO_A.get(k.split(" ", 1)[0], k[:40]), v["f1_macro"][0], "III") for k, v in A3.items() if not k.startswith("_") and not k.startswith("A0")]
ea += [("RoBERTa-Twitter zero-shot", T["zero_shot"]["f1_macro"], "III"), ("RoBERTa-Twitter ajustado (solución final)", F["etapa_A"]["f1_macro"], "IV")]
ea.sort(key=lambda t: t[1])
eb = [(k.replace(" | NLP + metadatos (propuesto)", " (diseño inicial)").replace(" | Solo metadatos (categoría + antigüedad)", " · solo metadatos").replace(" | Solo NLP de comentarios", " · solo comentarios"), v["r2"][0], "II") for k, v in A2["stageB_reg"].items() if "fuga" not in k]
eb += [(CORTO_B.get(k.split(" ", 1)[0] if not k.startswith("B8 ·") else k, k[:44]), v["r2"][0], "III") for k, v in B3.items() if not k.startswith("_") and not k.startswith(("B0", "B9"))]
eb += [("Stacking + RoBERTa (solución final)", F["etapa_B"]["con_sentimiento_roberta"]["r2"], "IV")]
eb = [e for e in eb if e[1] > 0.3]; eb.sort(key=lambda t: t[1])
col = {"II": "#e8a58f", "III": "#8fb8d4", "IV": AZ}
fig, ax = plt.subplots(1, 2, figsize=(10.2, 6.4), gridspec_kw=dict(wspace=0.75))
for a_, data, xl, lim in [(ax[0], ea, "F1-macro (5 folds)", (0.2, 0.98)), (ax[1], eb, "R² sobre log10(vistas) (5 folds)", (0.5, 0.84))]:
    y = np.arange(len(data))
    a_.barh(y, [d[1] for d in data], color=[col[d[2]] for d in data], height=.72)
    for yi, d in zip(y, data): a_.text(d[1] + (lim[1] - lim[0]) * .01, yi, f"{d[1]:.3f}", va="center", fontsize=6.6)
    a_.set_yticks(y, [d[0] for d in data], fontsize=6.6); a_.set_xlim(*lim); a_.set_xlabel(xl)
ax[0].set_title(f"Etapa A: {len(ea)} clasificadores de sentimiento", fontsize=9.5)
ax[1].set_title(f"Etapa B: {len(eb)} modelos de vistas", fontsize=9.5)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=col["II"], label="Avances II (diseño inicial)"), Patch(color=col["III"], label="Avances III (iteraciones)"),
                    Patch(color=col["IV"], label="Solución final (Avances IV)")], loc="lower center", ncol=3, frameon=False, fontsize=8)
plt.subplots_adjust(left=0.2, right=0.97, top=0.94, bottom=0.12)
plt.savefig(OUT / "p3_modelos.png"); plt.close()
print("ok", len(ea), len(eb))
