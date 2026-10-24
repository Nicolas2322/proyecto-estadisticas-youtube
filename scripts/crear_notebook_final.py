"""Genera notebooks/Avances_IV_proyecto_final.ipynb (ejecutable de punta a punta en Google Colab con GPU)."""
import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
md = lambda t: C.append(nbf.v4.new_markdown_cell(t.strip()))
code = lambda t: C.append(nbf.v4.new_code_cell(t.strip()))

md("""
# Proyecto Estadísticas de YouTube – Avances IV (entrega final)
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nicolas2322/proyecto-estadisticas-youtube/blob/main/notebooks/Avances_IV_proyecto_final.ipynb)

**Universidad de Cundinamarca · Especialización en Inteligencia Artificial · Procesamiento de Lenguaje Natural**
Docente: Paul Alexander Díaz Montaña · Estudiante: Yohan Nicolás Gómez Castañeda (trabajo individual aprobado por el docente)
Repositorio: https://github.com/Nicolas2322/proyecto-estadisticas-youtube

Solución final de punta a punta:
1. Registro de limpieza de los datos (con respaldo de los registros retirados).
2. **Etapa A** – sentimiento de cada comentario con RoBERTa-Twitter ajustado (5 folds, GPU).
3. **Etapa B** – predicción de las vistas del video con un *stacking* de cuatro modelos e intervalos conformales del 90 %.
4. Comparación de los resultados con el diseño inicial (Avances II).
5. **Modelo consumible**: interfaz Gradio con enlace público.

**Requisitos:** entorno con GPU (Entorno de ejecución → Cambiar tipo de entorno → T4) y la carpeta `MyDrive/youtube-statistics/` con `videos-stats.csv` y `comments.csv` del dataset [YouTube Statistics](https://www.kaggle.com/datasets/advaypatil/youtube-statistics).
""")
md("## 0. Entorno")
code("""
!pip -q install emoji vaderSentiment textblob xgboost gradio
!rm -rf repo && git clone -q --depth 1 https://github.com/Nicolas2322/proyecto-estadisticas-youtube.git repo
import sys, os, json, time, shutil, warnings
sys.path.insert(0, "repo/src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("GPU:", torch.cuda.get_device_name(0) if DEV == "cuda" else "no disponible")
os.makedirs("salida", exist_ok=True); RES = {}
""")
md("## 1. Datos desde Google Drive")
code("""
NECESARIOS = ["videos-stats.csv", "comments.csv"]
if not all(os.path.exists(f) for f in NECESARIOS):
    from google.colab import drive
    drive.mount("/content/drive")
    for f in NECESARIOS:
        shutil.copy(os.path.join("/content/drive/MyDrive/youtube-statistics", f), f)
print({f: os.path.getsize(f) for f in NECESARIOS})
""")
md("## 2. Registro de limpieza\nCada paso deja conteos antes y después, el criterio aplicado y un CSV con los registros retirados, de modo que cualquier decisión se puede revertir.")
code("""
!python repo/src/avances3/registro_limpieza.py videos-stats.csv comments.csv salida/registro_limpieza > /dev/null
reg = json.load(open("salida/registro_limpieza/registro_limpieza.json"))
display(pd.DataFrame(reg["pasos"])[["tabla", "paso", "antes", "despues", "accion", "respaldo"]])
print("Huellas SHA-256 de los archivos originales:", reg["fuentes"])
""")
md("## 3. Etapa A – Sentimiento con RoBERTa-Twitter ajustado (5 folds)")
code("""
from final.final import cargar
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
v, c = cargar("videos-stats.csv", "comments.csv")
y = c.Sentiment.values
assert len(c) == 18139 and list(np.bincount(y)) == [2328, 4590, 11221], "Los datos no coinciden con los del informe"
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
print(len(v), "videos ·", len(c), "comentarios ·", np.bincount(y))
""")
code("""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
MODELO = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tok = AutoTokenizer.from_pretrained(MODELO)
def normalizar(t): return " ".join("@user" if w.startswith("@") and len(w) > 1 else "http" if w.startswith("http") else w for w in str(t).split())
TEXTOS = [normalizar(t) for t in c.Comment.astype(str)]

@torch.no_grad()
def probabilidades(modelo, textos, bs=128):
    modelo.eval(); out = []
    for i in range(0, len(textos), bs):
        b = tok(textos[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEV)
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            out.append(torch.softmax(modelo(**b).logits.float(), -1).cpu().numpy())
    return np.vstack(out)

def ajustar(idx, epocas=2, bs=32, lr=2e-5, semilla=42):
    torch.manual_seed(semilla)
    m = AutoModelForSequenceClassification.from_pretrained(MODELO).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    pasos = epocas * int(np.ceil(len(idx) / bs)); sch = get_linear_schedule_with_warmup(opt, int(.06 * pasos), pasos)
    pesos = torch.tensor(len(y) / (3 * np.bincount(y)), dtype=torch.float).to(DEV)
    perdida = torch.nn.CrossEntropyLoss(weight=pesos); esc = torch.cuda.amp.GradScaler(enabled=DEV == "cuda")
    for ep in range(epocas):
        m.train(); perm = np.random.default_rng(semilla + ep).permutation(idx)
        for i in range(0, len(perm), bs):
            bi = perm[i:i + bs]
            b = tok([TEXTOS[j] for j in bi], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEV)
            with torch.autocast(DEV, enabled=DEV == "cuda"):
                l = perdida(m(**b).logits.float(), torch.tensor(y[bi]).to(DEV))
            opt.zero_grad(); esc.scale(l).backward(); esc.step(opt); esc.update(); sch.step()
    return m

P = np.zeros((len(y), 3)); f1_folds = []; t0 = time.time()
for k, (tr, te) in enumerate(FOLDS):
    m = ajustar(tr); P[te] = probabilidades(m, [TEXTOS[j] for j in te])
    f1_folds.append(f1_score(y[te], P[te].argmax(1), average="macro"))
    print(f"fold {k + 1}: F1-macro {f1_folds[-1]:.4f} ({(time.time() - t0) / 60:.1f} min)")
    del m; torch.cuda.empty_cache()
pred_a = P.argmax(1)
RES["etapa_A"] = dict(modelo=MODELO, f1_macro=float(np.mean(f1_folds)), f1_sd=float(np.std(f1_folds)), f1_folds=[float(x) for x in f1_folds],
                      exactitud=float(accuracy_score(y, pred_a)), reporte=classification_report(y, pred_a, output_dict=True),
                      matriz=confusion_matrix(y, pred_a, normalize="true").round(3).tolist())
print(classification_report(y, pred_a, target_names=["Negativo", "Neutral", "Positivo"], digits=3))
pd.DataFrame({"Video ID": c["Video ID"], "y": y, "p0": P[:, 0], "p1": P[:, 1], "p2": P[:, 2]}).to_csv("salida/oof_roberta.csv", index=False)
""")
code("""
M = np.array(RES["etapa_A"]["matriz"]); fig, ax = plt.subplots(figsize=(4, 3.4))
ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] > .5 else "black")
ax.set_xticks(range(3), ["Neg", "Neu", "Pos"]); ax.set_yticks(range(3), ["Neg", "Neu", "Pos"]); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
ax.set_title("RoBERTa-Twitter ajustado (fuera de fold)"); plt.tight_layout(); plt.savefig("salida/confusion_roberta.png", dpi=150); plt.show()
""")
md("### Modelo de sentimiento final (todos los comentarios) para la demo")
code("""
t0 = time.time(); MODELO_A = ajustar(np.arange(len(y)))
print(f"Entrenado en {(time.time() - t0) / 60:.1f} min")
def clasificar(textos): return probabilidades(MODELO_A, [normalizar(t) for t in textos]).argmax(1)
""")
md("## 4. Etapa B – Vistas del video (stacking + intervalos conformales)")
code("""
from final.final import variables_comentario, construir_videos, validar, entrenar_final, NUM, SENT
cv_ = variables_comentario(c); cv_["pred"] = pred_a
D, TOPICOS = construir_videos(v, cv_)
t0 = time.time(); met, pred_b, lo, hi = validar(D)
sin_sent = [x for x in NUM if x not in SENT]; met_ss, _, _, _ = validar(D, num=sin_sent)
RES["etapa_B"] = dict(con_sentimiento_roberta=met, sin_variables_de_sentimiento=met_ss)
print(f"Stacking con sentimiento RoBERTa: R² {met['r2']:.3f} ± {met['r2_sd']:.3f} · MAE {met['mae']:.3f} (×{met['factor']:.2f}) · cobertura 90 %: {met['cobertura']:.3f}")
print(f"Sin variables de sentimiento:     R² {met_ss['r2']:.3f} ± {met_ss['r2_sd']:.3f}")
print(f"({(time.time() - t0) / 60:.1f} min)")
""")
code("""
idx = np.sort(np.random.default_rng(42).choice(len(D), 120, replace=False)); idx = idx[np.argsort(pred_b[idx])]
yv = D.log_views.values; x = np.arange(len(idx)); dentro = (yv[idx] >= lo[idx]) & (yv[idx] <= hi[idx])
fig, ax = plt.subplots(figsize=(9, 3.3)); ax.fill_between(x, lo[idx], hi[idx], color="#cfe0ea", label="Intervalo del 90 %")
ax.plot(x, pred_b[idx], color="#2a6f97", label="Predicción"); ax.scatter(x[dentro], yv[idx][dentro], s=8, color="#333", label="Real dentro")
ax.scatter(x[~dentro], yv[idx][~dentro], s=14, color="#e07a5f", marker="x", label="Real fuera")
ax.set_ylabel("log10(vistas)"); ax.set_xlabel("120 videos de muestra"); ax.legend(fontsize=8, ncol=4); plt.tight_layout()
plt.savefig("salida/intervalos.png", dpi=150); plt.show()
""")
md("## 5. Resultados frente al diseño inicial (Avances II)")
code("""
A2 = json.load(open("repo/reports/resultados.json"))
B3 = json.load(open("repo/reports/avances3/resultados_b.json"))["RES"]
filas = [
 ["Etapa A · sentimiento (F1-macro)", "SVM lineal TF-IDF", A2["stageA"]["SVM lineal (word+char)"]["f1_macro"][0], "RoBERTa-Twitter ajustado", RES["etapa_A"]["f1_macro"]],
 ["Etapa A · exactitud", "SVM lineal TF-IDF", A2["stageA"]["SVM lineal (word+char)"]["accuracy"][0], "RoBERTa-Twitter ajustado", RES["etapa_A"]["exactitud"]],
 ["Etapa A · F1 clase negativa", "SVM lineal TF-IDF", A2["bestA_report"]["Negativo"]["f1-score"], "RoBERTa-Twitter ajustado", RES["etapa_A"]["reporte"]["0"]["f1-score"]],
 ["Etapa B · R² log10(vistas)", "XGBoost", A2["stageB_reg"][A2["bestB_reg"]]["r2"][0], "Stacking 4 modelos + sentimiento RoBERTa", RES["etapa_B"]["con_sentimiento_roberta"]["r2"]],
 ["Etapa B · MAE log10(vistas)", "XGBoost", A2["stageB_reg"][A2["bestB_reg"]]["mae"][0], "Stacking 4 modelos + sentimiento RoBERTa", RES["etapa_B"]["con_sentimiento_roberta"]["mae"]],
 ["Éxito relativo (ROC-AUC)", "Regresión logística", A2["stageB_cls"][A2["bestB_cls"]]["auc"][0], "Regresión logística (C = 100)", B3["_clasificacion"]["Regresión logística (C = 100)"]["auc"][0]],
]
T = pd.DataFrame(filas, columns=["Indicador", "Diseño inicial", "Valor inicial", "Solución final", "Valor final"])
T["Cambio"] = T["Valor final"] - T["Valor inicial"]; RES["comparacion"] = T.to_dict("records")
display(T.round(3))
""")
md("## 6. Modelo consumible: interfaz Gradio\nPegue los comentarios del video (uno por línea; opcionalmente `likes | comentario`), elija la categoría y los días desde la publicación. Cada consulta deja un registro JSON para el monitoreo, sin guardar el texto.")
code("""
from final.final import predecir_video
MODELO_B, Q = entrenar_final(D, pred_b)
ETQ = {0: "Negativo", 1: "Neutral", 2: "Positivo"}
def demo_fn(comentarios, categoria, dias):
    t0 = time.time()
    tabla, r = predecir_video(comentarios, categoria, dias, clasificar, MODELO_B, TOPICOS, Q)
    tabla = tabla.assign(Sentimiento=tabla.pred.map(ETQ)).drop(columns="pred")
    print(json.dumps({"evento": "prediccion", "n_comentarios": len(tabla), "categoria": categoria, "vistas": round(r["vistas"]),
                      "latencia_ms": round(1000 * (time.time() - t0))}))
    resumen = (f"### Vistas estimadas: **{r['vistas']:,.0f}**\\nIntervalo del 90 %: {r['minimo']:,.0f} – {r['maximo']:,.0f}\\n\\n"
               f"Comentarios positivos: {r['positivos']:.0%} · negativos: {r['negativos']:.0%}")
    return tabla, resumen

EJEMPLO = "320 | This is the best explanation I have ever seen, thank you!\\n45 | Great video, very clear\\n12 | not good, the audio is terrible\\nwho is watching in 2022?"
tabla, resumen = demo_fn(EJEMPLO, "tutorial", 30); display(tabla); print(resumen)
""")
code("""
import gradio as gr
CATS = sorted(D.Keyword.unique().tolist())
demo = gr.Interface(fn=demo_fn,
    inputs=[gr.Textbox(lines=8, label="Comentarios (uno por línea; opcional: likes | comentario)", value=EJEMPLO),
            gr.Dropdown(CATS, value="tutorial", label="Categoría"), gr.Number(value=30, label="Días desde la publicación")],
    outputs=[gr.Dataframe(label="Sentimiento por comentario"), gr.Markdown()],
    title="Estadísticas de YouTube: sentimiento y vistas estimadas",
    description="Prototipo académico – Procesamiento de Lenguaje Natural, Universidad de Cundinamarca.")
demo.launch(share=True)
""")
md("## 7. Resultados")
code("""
RES["fecha"] = time.strftime("%Y-%m-%d %H:%M"); RES["gpu"] = torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu"
json.dump(RES, open("salida/resultados_finales.json", "w"), indent=1, default=float)
if os.path.isdir("/content/drive/MyDrive/youtube-statistics"):
    shutil.copytree("salida", "/content/drive/MyDrive/youtube-statistics/salida_final", dirs_exist_ok=True)
print(json.dumps({k: RES[k] for k in ["etapa_B"]}, indent=1, default=float))
print("ETAPA_A", json.dumps({k: RES["etapa_A"][k] for k in ["f1_macro", "f1_sd", "f1_folds", "exactitud", "matriz"]}))
print("F1_CLASES", {k: round(RES["etapa_A"]["reporte"][k]["f1-score"], 4) for k in ["0", "1", "2"]})
print("PROCESO TERMINADO")
""")
nb["cells"] = C
nb["metadata"].update({"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "accelerator": "GPU",
                       "colab": {"gpuType": "T4", "provenance": []}})
nbf.write(nb, "notebooks/Avances_IV_proyecto_final.ipynb"); print("ok", len(C))
