"""Servicio en la nube del proyecto Estadísticas de YouTube (Hugging Face Spaces).

Interfaz web y API del modelo final:
  - Etapa A: RoBERTa-Twitter ajustado clasifica el sentimiento de cada comentario.
  - Etapa B: stacking (XGBoost + HGB + RF + Ridge) estima las vistas con un intervalo conformal del 90 %.
Los modelos se descargan del repositorio de modelos indicado en la variable MODEL_REPO.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, deque
from datetime import datetime, timezone

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import requests
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from final.final import predecir_video

MODEL_REPO = os.environ.get("MODEL_REPO", "")
YT_KEY = os.environ.get("YOUTUBE_API_KEY", "")
ETQ = {0: "Negativo", 1: "Neutral", 2: "Positivo"}
REPO_GITHUB = "https://github.com/Nicolas2322/proyecto-estadisticas-youtube"

# ---------------------------------------------------------------- carga de modelos
t0 = time.time()
CARPETA = snapshot_download(MODEL_REPO) if MODEL_REPO else os.environ.get("MODEL_DIR", "modelo")
TOK = AutoTokenizer.from_pretrained(os.path.join(CARPETA, "sentimiento"))
MODELO_A = AutoModelForSequenceClassification.from_pretrained(os.path.join(CARPETA, "sentimiento")).eval()
torch.set_num_threads(max(1, os.cpu_count() or 1))
PAQ = joblib.load(os.path.join(CARPETA, "etapa_b.joblib"))
MODELO_B, TOPICOS, Q = PAQ["modelo_b"], PAQ["topicos"], PAQ["q"]
CATS, REF, METRICAS = PAQ["categorias"], PAQ["referencia"], PAQ["metricas"]
VERSION = PAQ.get("version", "1.0")
print(f"Modelos cargados en {time.time() - t0:.1f} s desde {MODEL_REPO or CARPETA}")


def normalizar(t):
    return " ".join("@user" if w.startswith("@") and len(w) > 1 else "http" if w.startswith("http") else w for w in str(t).split())


@torch.no_grad()
def clasificar(textos, bs=32):
    out = []
    for i in range(0, len(textos), bs):
        b = TOK([normalizar(t) for t in textos[i:i + bs]], padding=True, truncation=True, max_length=128, return_tensors="pt")
        out.append(MODELO_A(**b).logits.argmax(-1).numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- monitoreo
REGISTRO = deque(maxlen=5000)
LOG_PATH = "registro_consultas.jsonl"


def registrar(evento: dict):
    REGISTRO.append(evento)
    print(json.dumps(evento, ensure_ascii=False), flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(evento, ensure_ascii=False) + "\n")
    except OSError:
        pass


def tablero():
    if not REGISTRO:
        return "Aún no hay consultas registradas en esta instancia.", pd.DataFrame()
    d = pd.DataFrame(list(REGISTRO))
    tot = Counter()
    for c in d.clases:
        tot.update(c)
    n = sum(tot.values()) or 1
    actual = np.array([tot.get(k, 0) / n for k in ["Negativo", "Neutral", "Positivo"]])
    ref = np.array(REF["dist_clases"])
    dvt = 0.5 * np.abs(actual - ref).sum()
    long_act = float((d.longitud_mediana * d.n_comentarios).sum() / max(d.n_comentarios.sum(), 1))
    suficiente = n >= 30
    alerta = suficiente and (dvt > 0.15 or abs(long_act - REF["longitud_mediana"]) / REF["longitud_mediana"] > 0.5)
    resumen = (f"**Consultas:** {len(d)} · **comentarios clasificados:** {int(d.n_comentarios.sum())}  \n"
               f"**Latencia:** mediana {d.latencia_ms.median():.0f} ms · p95 {d.latencia_ms.quantile(.95):.0f} ms  \n"
               f"**Distancia de variación total** entre las clases predichas y las de entrenamiento: {dvt:.3f} (umbral 0,15)  \n"
               f"**Longitud mediana** de los comentarios: {long_act:.0f} caracteres (entrenamiento: {REF['longitud_mediana']:.0f})  \n"
               + ("Se necesitan al menos 30 comentarios clasificados para evaluar la deriva." if not suficiente else
                  "**⚠ Alerta de deriva:** revisar los datos de entrada y considerar reentrenamiento." if alerta else "Sin alerta de deriva."))
    tabla = pd.DataFrame({"Clase": ["Negativo", "Neutral", "Positivo"], "Entrenamiento": ref.round(3), "Consultas": actual.round(3)})
    return resumen, tabla


# ---------------------------------------------------------------- predicción
def _predecir(lineas, categoria, dias, origen):
    t0 = time.time()
    if categoria not in CATS:
        raise gr.Error(f"Categoría no reconocida. Use una de: {', '.join(CATS[:8])}…")
    try:
        tabla, r = predecir_video(lineas, categoria, float(dias or 1), clasificar, MODELO_B, TOPICOS, Q)
    except ValueError as e:
        raise gr.Error(str(e))
    tabla = tabla.assign(Sentimiento=tabla.pred.map(ETQ)).drop(columns="pred").rename(columns={"Comment": "Comentario"})
    lat = round(1000 * (time.time() - t0))
    registrar({"fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"), "origen": origen, "categoria": categoria,
               "n_comentarios": len(tabla), "longitud_mediana": float(tabla.Comentario.str.len().median()),
               "clases": dict(Counter(tabla.Sentimiento)), "vistas": round(r["vistas"]), "latencia_ms": lat, "version": VERSION})
    return tabla, r, lat


def resumen_md(r, lat):
    return (f"### Vistas estimadas: **{r['vistas']:,.0f}**\n"
            f"Intervalo del 90 %: **{r['minimo']:,.0f} – {r['maximo']:,.0f}**\n\n"
            f"Comentarios positivos: **{r['positivos']:.0%}** · negativos: **{r['negativos']:.0%}**\n\n"
            f"<sub>Tiempo de respuesta: {lat} ms · modelo v{VERSION}</sub>")


def predecir_texto(comentarios, categoria, dias):
    tabla, r, lat = _predecir(comentarios, categoria, dias, "interfaz")
    return tabla, resumen_md(r, lat)


def api_predecir(comentarios: str, categoria: str, dias: float) -> dict:
    """API: devuelve el sentimiento de cada comentario y las vistas estimadas con su intervalo del 90 %."""
    tabla, r, lat = _predecir(comentarios, categoria, dias, "api")
    return {"comentarios": tabla.to_dict("records"), "vistas_estimadas": round(r["vistas"]),
            "intervalo_90": [round(r["minimo"]), round(r["maximo"])], "proporcion_positivos": round(r["positivos"], 3),
            "proporcion_negativos": round(r["negativos"], 3), "latencia_ms": lat, "version_modelo": VERSION}


def video_youtube(url, categoria):
    if not YT_KEY:
        raise gr.Error("Este servicio no tiene configurada la variable YOUTUBE_API_KEY.")
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", url or "")
    vid = m.group(1) if m else (url or "").strip()
    if not re.fullmatch(r"[\w-]{11}", vid):
        raise gr.Error("Ingrese un enlace o identificador de video válido.")
    base = "https://www.googleapis.com/youtube/v3/"
    v = requests.get(base + "videos", params={"part": "snippet,statistics", "id": vid, "key": YT_KEY}, timeout=15).json()
    if not v.get("items"):
        raise gr.Error("No se encontró el video o la cuota de la API se agotó.")
    pub = datetime.fromisoformat(v["items"][0]["snippet"]["publishedAt"].replace("Z", "+00:00"))
    dias = max((datetime.now(timezone.utc) - pub).days, 1)
    reales = int(v["items"][0]["statistics"].get("viewCount", 0))
    c = requests.get(base + "commentThreads", params={"part": "snippet", "videoId": vid, "order": "relevance", "maxResults": 10,
                                                      "textFormat": "plainText", "key": YT_KEY}, timeout=15).json()
    # Solo se toman el texto y los likes: el nombre del autor no se lee ni se guarda (Ley 1581 de 2012).
    lineas = [f"{it['snippet']['topLevelComment']['snippet']['likeCount']} | "
              f"{it['snippet']['topLevelComment']['snippet']['textOriginal'].replace(chr(10), ' ')}" for it in c.get("items", [])]
    if not lineas:
        raise gr.Error("El video no tiene comentarios públicos disponibles.")
    tabla, r, lat = _predecir("\n".join(lineas), categoria, dias, "youtube")
    extra = f"\n\nVistas reales al momento de la consulta: **{reales:,}** · días desde la publicación: {dias}"
    return tabla, resumen_md(r, lat) + extra


# ---------------------------------------------------------------- interfaz
EJEMPLO = ("320 | This is the best explanation I have ever seen, thank you!\n45 | Great video, very clear\n"
           "12 | not good, the audio is terrible\nwho is watching in 2022?")
CSS = ".gradio-container {max-width: 1150px !important}"

with gr.Blocks(title="Estadísticas de YouTube – PLN") as demo:
    gr.Markdown("# Estadísticas de YouTube: sentimiento de la audiencia y vistas estimadas\n"
                "Proyecto final · Procesamiento de Lenguaje Natural · Especialización en Inteligencia Artificial, Universidad de Cundinamarca. "
                f"[Repositorio]({REPO_GITHUB})")
    with gr.Tab("Predicción"):
        with gr.Row():
            with gr.Column(scale=5):
                txt = gr.Textbox(lines=9, value=EJEMPLO, label="Comentarios del video (uno por línea; opcional: likes | comentario)")
                with gr.Row():
                    cat = gr.Dropdown(CATS, value="tutorial" if "tutorial" in CATS else CATS[0], label="Categoría")
                    dias = gr.Number(value=30, minimum=1, label="Días desde la publicación")
                btn = gr.Button("Analizar", variant="primary")
            with gr.Column(scale=6):
                out_md = gr.Markdown()
                out_tab = gr.Dataframe(label="Sentimiento por comentario", wrap=True, column_widths=["68%", "12%", "20%"])
        btn.click(predecir_texto, [txt, cat, dias], [out_tab, out_md], api_name=False)
    with gr.Tab("Video de YouTube"):
        gr.Markdown("Analiza los 10 comentarios más relevantes de un video público usando la API de datos de YouTube. "
                    "No se leen ni se guardan los nombres de los autores." + ("" if YT_KEY else "\n\n*No disponible: falta configurar YOUTUBE_API_KEY.*"))
        with gr.Row():
            url = gr.Textbox(label="Enlace o identificador del video", scale=4)
            cat2 = gr.Dropdown(CATS, value="tutorial" if "tutorial" in CATS else CATS[0], label="Categoría", scale=2)
        btn2 = gr.Button("Descargar y analizar", variant="primary", interactive=bool(YT_KEY))
        out_md2 = gr.Markdown(); out_tab2 = gr.Dataframe(wrap=True, column_widths=["68%", "12%", "20%"])
        btn2.click(video_youtube, [url, cat2], [out_tab2, out_md2], api_name=False)
    with gr.Tab("Monitoreo"):
        gr.Markdown("Seguimiento de las consultas de esta instancia: latencia y deriva de los datos de entrada frente al entrenamiento. "
                    "El registro no guarda el texto de los comentarios.")
        mon_md = gr.Markdown(); mon_tab = gr.Dataframe()
        gr.Button("Actualizar").click(tablero, None, [mon_md, mon_tab], api_name=False)
    with gr.Tab("API"):
        gr.Markdown(f"""El modelo se consume como API con el cliente de Gradio o por HTTP.

**Python**
```python
from gradio_client import Client
cliente = Client("{os.environ.get('SPACE_ID', 'usuario/espacio')}")
r = cliente.predict("Great video!\\nterrible audio", "tutorial", 30, api_name="/predecir")
print(r["vistas_estimadas"], r["intervalo_90"])
```

**HTTP (curl)**
```bash
curl -X POST https://{os.environ.get('SPACE_HOST', 'usuario-espacio.hf.space')}/gradio_api/call/predecir \\
  -H "Content-Type: application/json" -d '{{"data": ["Great video!", "tutorial", 30]}}'
# devuelve un EVENT_ID; el resultado se obtiene con:
curl https://{os.environ.get('SPACE_HOST', 'usuario-espacio.hf.space')}/gradio_api/call/predecir/EVENT_ID
```
""")
        a_txt = gr.Textbox(label="comentarios", value="Great video!\nterrible audio"); a_cat = gr.Textbox(label="categoria", value="tutorial")
        a_dias = gr.Number(label="dias", value=30); a_out = gr.JSON(label="Respuesta")
        gr.Button("Probar la API").click(api_predecir, [a_txt, a_cat, a_dias], a_out, api_name="predecir")
    with gr.Tab("Acerca del modelo"):
        gr.Markdown(f"""**Versión del modelo:** {VERSION} · **repositorio de modelos:** `{MODEL_REPO}`

| Etapa | Modelo | Desempeño (validación cruzada de 5 folds) |
|---|---|---|
| A · sentimiento | RoBERTa-Twitter ajustado | F1-macro {METRICAS['f1_macro']:.3f} · exactitud {METRICAS['exactitud']:.3f} |
| B · vistas | Stacking XGBoost + HGB + RF + Ridge | R² {METRICAS['r2']:.3f} sobre log10(vistas) · error típico ×{METRICAS['factor']:.2f} · cobertura del intervalo del 90 %: {METRICAS['cobertura']:.3f} |

Datos de entrenamiento: *YouTube Statistics* (Kaggle), 1.867 videos y 18.139 comentarios en inglés de 41 categorías.
El intervalo indica el rango en el que caen las vistas reales en 9 de cada 10 videos: el modelo distingue órdenes de magnitud, no cifras exactas.
""")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(css=CSS)
