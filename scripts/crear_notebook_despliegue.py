"""Genera notebooks/Despliegue_Hugging_Face.ipynb: entrena los modelos finales y los publica en Hugging Face (modelo + Space)."""
import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
md = lambda t: C.append(nbf.v4.new_markdown_cell(t.strip()))
code = lambda t: C.append(nbf.v4.new_code_cell(t.strip()))

md("""
# Proyecto Estadísticas de YouTube – Despliegue en la nube (Hugging Face Spaces)
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nicolas2322/proyecto-estadisticas-youtube/blob/main/notebooks/Despliegue_Hugging_Face.ipynb)

**Universidad de Cundinamarca · Especialización en Inteligencia Artificial · Procesamiento de Lenguaje Natural**
Estudiante: Yohan Nicolás Gómez Castañeda · Repositorio: https://github.com/Nicolas2322/proyecto-estadisticas-youtube

Este notebook entrena los modelos finales con todos los datos y los publica en la nube:
1. **Repositorio de modelos** en Hugging Face Hub: RoBERTa-Twitter ajustado (etapa A) y el *stacking* con su intervalo conformal (etapa B).
2. **Space de Hugging Face** con la interfaz web (Gradio) y la API, que descarga los modelos del repositorio anterior.

**Requisitos:** entorno con GPU T4; el Space se crea en **ZeroGPU**, la opción gratuita para Spaces de Gradio; cuenta en Hugging Face con un *token* de escritura (guardado como secreto `HF_TOKEN` en Colab o escrito cuando se pida);
y en `MyDrive/youtube-statistics/` los archivos `videos-stats.csv`, `comments.csv` y `salida_final/oof_roberta.csv` (predicciones fuera de fold del notebook de Avances IV).
""")
md("## 0. Autenticación en Hugging Face")
code("""
from getpass import getpass
from huggingface_hub import login, whoami
try:
    from google.colab import userdata
    TOKEN = userdata.get("HF_TOKEN")
except Exception:
    TOKEN = None
TOKEN = TOKEN or getpass("Token de escritura de Hugging Face: ")
login(token=TOKEN, add_to_git_credential=False)
USUARIO = whoami()["name"]
REPO_MODELO, REPO_SPACE = f"{USUARIO}/youtube-sentimiento-vistas", f"{USUARIO}/estadisticas-youtube"
print("Usuario:", USUARIO, "| modelo:", REPO_MODELO, "| space:", REPO_SPACE)
""")
md("## 1. Entorno y datos")
code("""
!pip -q install vaderSentiment textblob xgboost gradio
!rm -rf repo && git clone -q --depth 1 https://github.com/Nicolas2322/proyecto-estadisticas-youtube.git repo
import sys, os, json, time, shutil, warnings
sys.path.insert(0, "repo/src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, joblib
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("GPU:", torch.cuda.get_device_name(0) if DEV == "cuda" else "no disponible")
from google.colab import drive
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/youtube-statistics"
for f in ["videos-stats.csv", "comments.csv"]: shutil.copy(f"{BASE}/{f}", f)
shutil.copy(f"{BASE}/salida_final/oof_roberta.csv", "oof_roberta.csv")
from final.final import cargar
v, c = cargar("videos-stats.csv", "comments.csv"); y = c.Sentiment.values
oof = pd.read_csv("oof_roberta.csv")
assert len(oof) == len(c) and (oof["Video ID"].values == c["Video ID"].values).all() and (oof.y.values == y).all()
pred_a = oof[["p0", "p1", "p2"]].values.argmax(1)
from sklearn.metrics import f1_score, accuracy_score
F1, ACC = f1_score(y, pred_a, average="macro"), accuracy_score(y, pred_a)
print(len(v), "videos ·", len(c), "comentarios · F1-macro fuera de fold:", round(F1, 4), "· exactitud:", round(ACC, 4))
os.makedirs("paquete/sentimiento", exist_ok=True)
""")
md("## 2. Etapa A: RoBERTa-Twitter ajustado con todos los comentarios")
code("""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
BASE_MODELO = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tok = AutoTokenizer.from_pretrained(BASE_MODELO)
def normalizar(t): return " ".join("@user" if w.startswith("@") and len(w) > 1 else "http" if w.startswith("http") else w for w in str(t).split())
TEXTOS = [normalizar(t) for t in c.Comment.astype(str)]

def ajustar(idx, epocas=2, bs=32, lr=2e-5, semilla=42):
    torch.manual_seed(semilla)
    m = AutoModelForSequenceClassification.from_pretrained(BASE_MODELO).to(DEV)
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
    return m.eval()

t0 = time.time(); MODELO_A = ajustar(np.arange(len(y)))
MODELO_A.config.id2label = {0: "Negativo", 1: "Neutral", 2: "Positivo"}; MODELO_A.config.label2id = {v_: k for k, v_ in MODELO_A.config.id2label.items()}
MODELO_A.cpu().save_pretrained("paquete/sentimiento"); tok.save_pretrained("paquete/sentimiento")
print(f"Entrenado y guardado en {(time.time() - t0) / 60:.1f} min")
""")
md("## 3. Etapa B: stacking e intervalo conformal")
code("""
from final.final import variables_comentario, construir_videos, validar, entrenar_final, predecir_video
cv_ = variables_comentario(c); cv_["pred"] = pred_a
D, TOPICOS = construir_videos(v, cv_)
t0 = time.time(); met, pred_b, lo, hi = validar(D)
MODELO_B, Q = entrenar_final(D, pred_b)
print(f"R² {met['r2']:.3f} ± {met['r2_sd']:.3f} · MAE {met['mae']:.3f} (×{met['factor']:.2f}) · cobertura {met['cobertura']:.3f} · q {Q:.3f} ({(time.time() - t0) / 60:.1f} min)")
import sklearn, xgboost, transformers, gradio
VERSIONES = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__,
             "transformers": transformers.__version__, "torch": torch.__version__.split("+")[0], "gradio": gradio.__version__, "joblib": joblib.__version__}
PAQ = {"modelo_b": MODELO_B, "topicos": TOPICOS, "q": Q, "categorias": sorted(D.Keyword.unique().tolist()),
       "referencia": {"dist_clases": (np.bincount(pred_a, minlength=3) / len(pred_a)).tolist(),
                      "longitud_mediana": float(c.Comment.astype(str).str.len().median())},
       "metricas": {"f1_macro": float(F1), "exactitud": float(ACC), "r2": met["r2"], "r2_sd": met["r2_sd"], "mae": met["mae"],
                    "factor": met["factor"], "cobertura": met["cobertura"], "ancho_log10": met["ancho_log10"]},
       "version": time.strftime("1.0.%Y%m%d"), "versiones": VERSIONES}
joblib.dump(PAQ, "paquete/etapa_b.joblib", compress=3)
print(VERSIONES)
""")
md("## 4. Prueba local del paquete antes de publicar")
code("""
@torch.no_grad()
def clasificar(textos):
    b = tok([normalizar(t) for t in textos], padding=True, truncation=True, max_length=128, return_tensors="pt")
    return MODELO_A(**b).logits.argmax(-1).numpy()
EJEMPLO = "320 | This is the best explanation I have ever seen, thank you!\\n45 | Great video, very clear\\n12 | not good, the audio is terrible\\nwho is watching in 2022?"
t0 = time.time(); tabla, r = predecir_video(EJEMPLO, "tutorial", 30, clasificar, MODELO_B, TOPICOS, Q)
display(tabla); print({k: round(v_) if k in ("vistas", "minimo", "maximo") else round(v_, 3) for k, v_ in r.items()}, f"{1000 * (time.time() - t0):.0f} ms en CPU")
""")
md("## 5. Publicación: repositorio de modelos y Space")
code("""
from huggingface_hub import HfApi
api = HfApi()
open("paquete/README.md", "w").write(f\"\"\"---
language: en
license: mit
library_name: transformers
pipeline_tag: text-classification
tags: [sentiment-analysis, youtube, roberta, stacking, conformal-prediction]
base_model: {BASE_MODELO}
---
# Sentimiento de comentarios y vistas de videos de YouTube

Modelos del proyecto final de Procesamiento de Lenguaje Natural (Universidad de Cundinamarca).

- `sentimiento/`: {BASE_MODELO} ajustado con 18.139 comentarios del conjunto *YouTube Statistics* (Kaggle). F1-macro fuera de fold: {F1:.3f}; exactitud: {ACC:.3f}.
- `etapa_b.joblib`: stacking (XGBoost + HistGradientBoosting + Random Forest + Ridge, meta-modelo RidgeCV) que estima log10(vistas) a partir de 32 variables por video, con intervalo conformal del 90 %. R² {met['r2']:.3f}; cobertura {met['cobertura']:.3f}.

Código: https://github.com/Nicolas2322/proyecto-estadisticas-youtube · Interfaz y API: https://huggingface.co/spaces/{REPO_SPACE}
\"\"\")
api.create_repo(REPO_MODELO, repo_type="model", exist_ok=True)
api.upload_folder(folder_path="paquete", repo_id=REPO_MODELO, repo_type="model", commit_message=f"Modelos v{PAQ['version']}")

os.makedirs("space/final", exist_ok=True)
shutil.copy("repo/despliegue/space/app.py", "space/app.py")
shutil.copy("repo/src/final/final.py", "space/final/final.py"); open("space/final/__init__.py", "w").close()
V = VERSIONES
open("space/requirements.txt", "w").write("\\n".join([
    "torch", f"transformers=={V['transformers']}",
    f"scikit-learn=={V['scikit-learn']}", f"xgboost=={V['xgboost']}", f"numpy=={V['numpy']}", f"pandas=={V['pandas']}",
    f"joblib=={V['joblib']}", "vaderSentiment", "textblob", "requests", "huggingface_hub"]) + "\\n")
open("space/README.md", "w").write(f\"\"\"---
title: Estadísticas de YouTube PLN
emoji: 📊
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: {V['gradio']}
app_file: app.py
pinned: false
license: mit
short_description: Sentimiento de comentarios y vistas estimadas de YouTube
---
Interfaz y API del proyecto final de Procesamiento de Lenguaje Natural. Código: https://github.com/Nicolas2322/proyecto-estadisticas-youtube
\"\"\")
# Las cuentas gratuitas pueden alojar hasta 2 Spaces de Gradio en ZeroGPU (CPU basic requiere PRO desde julio de 2026).
api.create_repo(REPO_SPACE, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g", exist_ok=True)
api.add_space_variable(REPO_SPACE, "MODEL_REPO", REPO_MODELO)
api.upload_folder(folder_path="space", repo_id=REPO_SPACE, repo_type="space", commit_message=f"Servicio v{PAQ['version']}")
print("Modelo: https://huggingface.co/" + REPO_MODELO); print("Space:  https://huggingface.co/spaces/" + REPO_SPACE)
""")
md("## 6. Verificación del servicio en la nube")
code("""
for i in range(90):
    etapa = api.get_space_runtime(REPO_SPACE).stage
    if etapa in ("RUNNING", "RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR"): break
    time.sleep(20)
print("Estado del Space:", etapa)
from gradio_client import Client
cliente = Client(REPO_SPACE, verbose=False)
t0 = time.time(); r = cliente.predict(EJEMPLO, "tutorial", 30, api_name="/predecir")
print(json.dumps(r, ensure_ascii=False, indent=1)); print(f"Respuesta de la API en {time.time() - t0:.1f} s")
host = api.space_info(REPO_SPACE).host if hasattr(api.space_info(REPO_SPACE), "host") else None
json.dump({"space": f"https://huggingface.co/spaces/{REPO_SPACE}", "host": host, "modelo": f"https://huggingface.co/{REPO_MODELO}",
           "metricas": PAQ["metricas"], "versiones": VERSIONES, "prueba_api": r}, open("despliegue.json", "w"), indent=1, ensure_ascii=False)
shutil.copy("despliegue.json", f"{BASE}/salida_final/despliegue.json")
print("PROCESO TERMINADO")
""")
nb["cells"] = C
nb["metadata"].update({"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "accelerator": "GPU",
                       "colab": {"gpuType": "T4", "provenance": []}})
nbf.write(nb, "notebooks/Despliegue_Hugging_Face.ipynb"); print("ok", len(C))
