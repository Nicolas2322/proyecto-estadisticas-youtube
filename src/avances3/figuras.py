"""Figuras del informe Avances III. Uso: python figuras.py <carpeta_salida>"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path(sys.argv[1])
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"],
                     "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False})
C_ADOP, C_DESC, C_BASE, C_REF = "#2a6f97", "#bfcfdb", "#e07a5f", "#9a9a9a"
A = json.load(open(OUT / "resultados_a.json")); BB = json.load(open(OUT / "resultados_b.json")); B = BB["RES"]
TR = OUT / "resultados_transformer.json"; T = json.load(open(TR)) if TR.exists() else None

# ---------- Figura: trayectoria de iteraciones etapa A
names = [k for k in A if not k.startswith("_")]
lab = {k: k.split(" ", 1)[0] + " " + k.split(" ", 1)[1].replace("+ ", "") for k in names}
vals = [A[k]["f1_macro"][0] for k in names]; sds = [A[k]["f1_macro"][1] for k in names]
if T:
    for key, nm in [("zero_shot", "A6a RoBERTa-Twitter zero-shot"), ("ajuste_fino", "A6b RoBERTa-Twitter ajuste fino")]:
        names.append(nm); lab[nm] = nm; vals.append(T[key]["f1_macro"]); sds.append(float(np.std(T[key]["f1_macro_folds"])))
final_a = A["_final"]
col = []
for k in names:
    if k.startswith("A0"): col.append(C_BASE)
    elif k == final_a: col.append(C_ADOP)
    elif k.startswith(("A3a", "A5", "A6a")): col.append(C_DESC)
    else: col.append("#6fa3c4")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
yy = np.arange(len(names))[::-1]
ax[0].barh(yy, vals, xerr=sds, color=col, height=.65)
for yv, v_, sd_ in zip(yy, vals, sds): ax[0].text(v_ + sd_ + .006, yv, f"{v_:.3f}", va="center", fontsize=8)
ax[0].set_yticks(yy, [lab[k] if len(lab[k]) < 52 else lab[k][:50] + "…" for k in names], fontsize=7.5)
ax[0].set_xlim(.55, .85); ax[0].set_xlabel("F1-macro (validación cruzada, 5 folds)"); ax[0].set_title("Trayectoria de iteraciones – etapa A")
ax[0].axvline(A[names[0]]["f1_macro"][0], color=C_BASE, lw=.8, ls="--")
cls = ["Negativo", "Neutral", "Positivo"]; x = np.arange(3); w = .38
a0 = [A[names[0]][k] for k in ["f1_neg", "f1_neu", "f1_pos"]]; af = [A[final_a][k] for k in ["f1_neg", "f1_neu", "f1_pos"]]
ax[1].bar(x - w / 2, a0, w, color=C_BASE, label="A0 (Avances II)"); ax[1].bar(x + w / 2, af, w, color=C_ADOP, label=final_a.split()[0] + " (final)")
for i in range(3): ax[1].text(x[i] + w / 2, af[i] + .01, f"+{af[i]-a0[i]:.3f}", ha="center", fontsize=8)
ax[1].set_xticks(x, cls); ax[1].set_ylim(0, 1); ax[1].set_ylabel("F1 por clase"); ax[1].legend(frameon=False, fontsize=8, loc="upper left"); ax[1].set_title("Mejora por clase")
plt.tight_layout(); plt.savefig(OUT / "g1_iter_a.png"); plt.close()

# ---------- Figura: matrices de confusión A0 vs final
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
for a_, M, t in [(ax[0], np.array(A["_cm_base"]), "A0 – Avances II"), (ax[1], np.array(A["_cm_final"]), f"{final_a.split()[0]} – modelo final")]:
    a_.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3): a_.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] > .5 else "black")
    a_.set_xticks(range(3), ["Neg", "Neu", "Pos"]); a_.set_yticks(range(3), ["Neg", "Neu", "Pos"]); a_.set_xlabel("Predicho"); a_.set_ylabel("Real"); a_.set_title(t, fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "g2_confusion.png"); plt.close()

# ---------- Figura: trayectoria etapa B
order = [k for k in B if not k.startswith("_") and not k.startswith("B9")]
short = {k: k.replace(" (XGBoost, 32 variables)", "").replace("Optimización bayesiana (Optuna, CV anidada 5×3, 25 ensayos)", "Optuna (CV anidada)")
         .replace("Selección por SHAP (95 % de la importancia acumulada)", "Selección por SHAP").replace("Sentimiento del clasificador mejorado (etapa A final)", "Sentimiento mejorado (A final)")
         .replace("+ Variables del título (longitud, mayúsculas, signos, formato)", "+ Variables del título").replace(" (mismas variables)", "")
         .replace("Ensamble por promedio (XGB + HGB + RF + Ridge)", "Promedio XGB+HGB+RF+Ridge").replace("Stacking (meta-modelo RidgeCV)", "Stacking (RidgeCV)") for k in order}
vb = [B[k]["r2"][0] for k in order]; sb = [B[k]["r2"][1] for k in order]
cb = []
for k in order:
    if k.startswith("B0"): cb.append(C_BASE)
    elif k == B["_final"]: cb.append(C_ADOP)
    elif k.startswith("B8 ·"): cb.append(C_REF)
    elif "Adoptada" in B[k].get("decision", ""): cb.append("#6fa3c4")
    else: cb.append(C_DESC)
fig, ax = plt.subplots(figsize=(10, 4.2)); yy = np.arange(len(order))[::-1]
ax.barh(yy, vb, xerr=sb, color=cb, height=.65)
for yv, v_ in zip(yy, vb): ax.text(.8465, yv, f"{v_:.3f}", va="center", ha="right", fontsize=8)
ax.set_yticks(yy, [short[k] for k in order], fontsize=7.5); ax.set_xlim(.70, .85); ax.axvline(B[order[0]]["r2"][0], color=C_BASE, lw=.8, ls="--")
ax.set_xlabel("R² sobre log10(vistas) (validación cruzada, 5 folds)"); ax.set_title("Trayectoria de iteraciones – etapa B")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_BASE, label="Base Avances II"), Patch(color=C_DESC, label="Descartada"), Patch(color=C_REF, label="Componente individual"),
                   Patch(color="#6fa3c4", label="Adoptada"), Patch(color=C_ADOP, label="Modelo final")], frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=5)
plt.tight_layout(); plt.savefig(OUT / "g3_iter_b.png"); plt.close()

# ---------- Figura: Optuna
H = BB["OPT_HIST"]
fig, ax = plt.subplots(figsize=(7.5, 2.8))
for k, h in enumerate(H):
    h = np.array(h, float); ax.plot(np.arange(1, len(h) + 1), np.maximum.accumulate(h), marker="o", ms=2.5, lw=1, label=f"Fold externo {k+1}")
ax.axhline(B[order[0]]["r2"][0], color=C_BASE, ls="--", lw=.8, label="B0 (CV externa)")
ax.set_xlabel("Ensayo de Optuna"); ax.set_ylabel("Mejor R² interno (3 folds)"); ax.legend(fontsize=7, frameon=False, ncol=3); ax.set_title("Historial de la optimización bayesiana (TPE)")
plt.tight_layout(); plt.savefig(OUT / "g4_optuna.png"); plt.close()

# ---------- Figura: SHAP
import shap
S = np.load(OUT / "shap_b.npz", allow_pickle=True)
nice = {"clikes_log_max": "Likes del comentario más votado (log)", "log_age": "Antigüedad del video (log días)", "clikes_gini": "Gini de likes entre comentarios",
        "clikes_log_mean": "Likes medios de comentarios (log)", "vader_mean": "VADER medio", "pred_pos_share": "% comentarios positivos", "pred_neg_share": "% comentarios negativos",
        "tb_subj": "Subjetividad TextBlob", "tb_pol": "Polaridad TextBlob", "vader_std": "Dispersión VADER", "vader_wmean": "VADER ponderado por likes"}
fn = [nice.get(n, n.replace("topic_", "Tópico ").replace("Keyword_", "Categoría: ")) for n in S["names"]]
plt.figure()
shap.summary_plot(S["sv"], S["X"], feature_names=fn, max_display=12, show=False, plot_size=(8, 4.2), color_bar_label="Valor de la variable")
plt.xlabel("Valor SHAP (impacto en log10 de las vistas)", fontsize=10)
fig_ = plt.gcf()
for a_ in fig_.axes:
    a_.tick_params(labelsize=9)
    if a_.get_ylabel() == "Valor de la variable":
        a_.set_yticklabels(["Bajo", "Alto"]); a_.set_ylabel("Valor de la variable", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "g5_shap.png"); plt.close()

# ---------- Figura: intervalos conformales
P = np.load(OUT / "pred_b.npz")
yv, lo, hi, pc = P["y"], P["lo"], P["hi"], P["pc"]
rng = np.random.default_rng(42); idx = np.sort(rng.choice(len(yv), 120, replace=False)); idx = idx[np.argsort(pc[idx])]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw=dict(width_ratios=[2, 1]))
xs = np.arange(len(idx)); inside = (yv[idx] >= lo[idx]) & (yv[idx] <= hi[idx])
ax[0].fill_between(xs, lo[idx], hi[idx], color="#cfe0ea", label="Intervalo del 90 %"); ax[0].plot(xs, pc[idx], color=C_ADOP, lw=1, label="Predicción")
ax[0].scatter(xs[inside], yv[idx][inside], s=8, color="#333333", label="Real dentro"); ax[0].scatter(xs[~inside], yv[idx][~inside], s=12, color=C_BASE, marker="x", label="Real fuera")
ax[0].set_xlabel("120 videos de muestra, ordenados por predicción"); ax[0].set_ylabel("log10(vistas)"); ax[0].legend(fontsize=7, frameon=False, ncol=2); ax[0].set_title("Intervalos de predicción conformales")
cv_ = B["_conformal"]["cobertura_folds"]; ax[1].bar(range(1, 6), cv_, color=C_ADOP); ax[1].axhline(.9, color=C_BASE, ls="--", lw=1)
ax[1].set_ylim(.75, 1); ax[1].set_xlabel("Fold"); ax[1].set_ylabel("Cobertura empírica"); ax[1].set_title("Cobertura por fold (objetivo 90 %)")
plt.tight_layout(); plt.savefig(OUT / "g6_conformal.png"); plt.close()
print("ok")
