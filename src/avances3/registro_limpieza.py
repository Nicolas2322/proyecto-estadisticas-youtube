"""Registro (log) de limpieza de datos con trazabilidad y respaldo de los registros retirados.

Uso: python registro_limpieza.py <videos-stats.csv> <comments.csv> <carpeta_salida>
Genera: registro_limpieza.json (pasos, conteos antes/después, huellas SHA-256 de los archivos crudos)
        retirados_*.csv (registros eliminados, conservados para poder revertir cualquier paso)
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

VID, COM, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
LOG = {"fuentes": {VID.name: sha(VID), COM.name: sha(COM)}, "pasos": [], "perfil": {}}
def paso(tabla, nombre, antes, despues, accion, criterio, respaldo=None, **extra):
    LOG["pasos"].append(dict(tabla=tabla, paso=nombre, antes=int(antes), despues=int(despues), cambio=int(despues - antes),
                             accion=accion, criterio=criterio, respaldo=respaldo, **extra))

v = pd.read_csv(VID).drop(columns=["Unnamed: 0"], errors="ignore")
c = pd.read_csv(COM).drop(columns=["Unnamed: 0"], errors="ignore")
LOG["perfil"]["videos"] = dict(filas=len(v), tipos={k: str(t) for k, t in v.dtypes.items()}, faltantes=v.isna().sum().astype(int).to_dict())
LOG["perfil"]["comentarios"] = dict(filas=len(c), tipos={k: str(t) for k, t in c.dtypes.items()}, faltantes=c.isna().sum().astype(int).to_dict())

# ---------------- VIDEOS
# 1. Tipos y estandarización
fechas = pd.to_datetime(v["Published At"], errors="coerce")
kw_norm = v.Keyword.str.strip().str.lower()
paso("videos", "Tipos y estandarización", len(v), len(v), "Conversión de fecha a datetime y de contadores a numérico; categorías en minúscula sin espacios",
     f"{int(fechas.isna().sum())} fechas no convertibles; {int((kw_norm != v.Keyword).sum())} categorías modificadas; {v.Keyword.nunique()} categorías únicas")
# 2. Duplicados por identificador: exactos vs lógicos
dup_mask = v.duplicated("Video ID", keep=False)
grupos = v[dup_mask].groupby("Video ID")
exactos = int(v[dup_mask].duplicated(keep="first").sum())
def tipo(g):
    otras = g.drop(columns=["Keyword"])
    if otras.nunique().max() == 1: return "solo_categoria"
    num = g[["Likes", "Comments", "Views"]].where(lambda x: x >= 0)
    rel = ((num.max() - num.min()) / num.max().replace(0, np.nan)).fillna(0).max()
    return "diferencia_menor" if rel < 0.05 else "conflicto"
tipos = v[dup_mask].groupby("Video ID").apply(tipo)
logicos = int((tipos == "solo_categoria").sum()); menores = int((tipos == "diferencia_menor").sum()); conflictos = tipos[tipos == "conflicto"].index.tolist()
retirados_v = v[v.duplicated("Video ID", keep="first")]
retirados_v.to_csv(OUT / "retirados_videos_duplicados.csv", index=False)
paso("videos", "Duplicados del mismo video", len(v), len(v) - len(retirados_v), "Se conserva la primera aparición y se respalda el resto",
     f"{dup_mask.sum()} filas comparten identificador ({grupos.ngroups} videos) y ninguna es duplicado exacto ({exactos}): en {logicos} solo cambia la palabra clave de búsqueda; en {menores} los contadores difieren menos del 5 % (el mismo video capturado en momentos distintos); en {len(conflictos)} las vistas difieren de forma importante",
     "retirados_videos_duplicados.csv", solo_categoria=logicos, diferencia_menor=menores, conflicto=conflictos)
v = v.drop_duplicates("Video ID", keep="first")
# 3. Contadores ocultos (-1): recategorizar, no borrar
ocultos = {col: int((v[col] < 0).sum()) for col in ["Likes", "Comments"]}
for col in ["Likes", "Comments"]: v.loc[v[col] < 0, col] = np.nan
paso("videos", "Contadores ocultos (valor −1)", len(v), len(v), "Se recategorizan como faltantes (NaN, «contador oculto») y se imputan con la mediana en el modelado; no se eliminan",
     f"likes ocultos: {ocultos['Likes']}; comentarios ocultos: {ocultos['Comments']}. El valor −1 es la convención de YouTube para un contador deshabilitado, no un cero")
# 4. Objetivo faltante
sin_vistas = v[v.Views.isna() | (v.Views <= 0)]
sin_vistas.to_csv(OUT / "retirados_videos_sin_vistas.csv", index=False)
paso("videos", "Variable objetivo faltante", len(v), len(v) - len(sin_vistas), "Se retiran del modelado y se respaldan",
     f"{len(sin_vistas)} videos sin vistas: sin el valor a predecir no pueden usarse para entrenar ni evaluar", "retirados_videos_sin_vistas.csv",
     detalle=sin_vistas[["Video ID", "Title"]].to_dict("records"))
v = v[~v.index.isin(sin_vistas.index)]
ids_validos = set(v["Video ID"])

# ---------------- COMENTARIOS
sin_texto = c[c.Comment.isna() | c.Sentiment.isna()]
sin_texto.to_csv(OUT / "retirados_comentarios_sin_texto.csv", index=False)
paso("comentarios", "Comentario o etiqueta faltante", len(c), len(c) - len(sin_texto), "Se retiran y se respaldan",
     f"{len(sin_texto)} filas sin texto: un comentario vacío no aporta información al análisis de sentimiento", "retirados_comentarios_sin_texto.csv")
c = c[~c.index.isin(sin_texto.index)]
d_mask = c.duplicated(["Video ID", "Comment"], keep=False)
g = c[d_mask].groupby(["Video ID", "Comment"])
ex = int(c[d_mask].duplicated(keep="first").sum())
difieren_likes = int((g.Likes.nunique() > 1).sum()); difieren_sent = int((g.Sentiment.nunique() > 1).sum())
retirados_c = c[c.duplicated(["Video ID", "Comment"], keep="first")]
retirados_c.to_csv(OUT / "retirados_comentarios_duplicados.csv", index=False)
paso("comentarios", "Comentarios duplicados en el mismo video", len(c), len(c) - len(retirados_c), "Se conserva la primera aparición y se respalda el resto",
     f"{len(retirados_c)} repeticiones; {ex} son copias exactas (texto, likes y etiqueta). En {difieren_likes} grupos los likes difieren y en {difieren_sent} difiere la etiqueta: son el mismo comentario capturado en momentos distintos, no comentarios diferentes",
     "retirados_comentarios_duplicados.csv")
c = c.drop_duplicates(["Video ID", "Comment"], keep="first")
huerfanos = c[~c["Video ID"].isin(ids_validos)]
huerfanos.to_csv(OUT / "retirados_comentarios_sin_video.csv", index=False)
paso("comentarios", "Comentarios de videos retirados", len(c), len(c) - len(huerfanos), "Se retiran y se respaldan",
     f"{len(huerfanos)} comentarios pertenecen a videos sin variable objetivo", "retirados_comentarios_sin_video.csv")
c = c[c["Video ID"].isin(ids_validos)]
etq = sorted(c.Sentiment.unique().tolist())
paso("comentarios", "Validación de etiquetas", len(c), len(c), "Verificación del dominio de la etiqueta", f"valores presentes: {etq} (0 negativo, 1 neutral, 2 positivo)")

LOG["final"] = dict(videos=len(v), comentarios=len(c), comentarios_por_video_mediana=float(c.groupby("Video ID").size().median()),
                    faltantes_videos=v.isna().sum().astype(int).to_dict())
json.dump(LOG, open(OUT / "registro_limpieza.json", "w"), ensure_ascii=False, indent=1, default=str)
for p in LOG["pasos"]:
    print(f"[{p['tabla']}] {p['paso']}: {p['antes']} → {p['despues']} | {p['criterio']}")
print("FINAL", LOG["final"]); print("SHA", LOG["fuentes"])
