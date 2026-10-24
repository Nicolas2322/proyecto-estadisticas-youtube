"""Avances IV – solución final: variables por video, modelo de desempeño (stacking), intervalos conformales
y la función de predicción que usa la demo. El mismo código construye las variables en entrenamiento y en
la demo, lo que evita diferencias entre el entrenamiento y el uso real del modelo.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

SEED = 42
SENT = ["pred_pos_share", "pred_neg_share", "vader_mean", "vader_std", "vader_wmean", "tb_pol", "tb_subj"]
INTER = ["clikes_log_mean", "clikes_log_max", "clikes_gini"]
TOP = [f"topic_{i}" for i in range(20)]
NUM = SENT + INTER + TOP + ["log_age"]
CAT = ["Keyword"]


def cargar(vid_path, com_path):
    """Carga y limpieza documentadas en el registro de limpieza (Avances III, sección 3.3)."""
    v = pd.read_csv(vid_path).drop(columns=["Unnamed: 0"], errors="ignore")
    c = pd.read_csv(com_path).drop(columns=["Unnamed: 0"], errors="ignore")
    v = v.drop_duplicates("Video ID").dropna(subset=["Views"])
    for col in ["Likes", "Comments"]:
        v.loc[v[col] < 0, col] = np.nan
    v = v[v.Views > 0].copy()
    v["Published At"] = pd.to_datetime(v["Published At"], errors="coerce")
    v["age_days"] = (v["Published At"].max() - v["Published At"]).dt.days.clip(lower=1)
    c = c.dropna(subset=["Comment", "Sentiment"]).drop_duplicates(["Video ID", "Comment"])
    c["Sentiment"] = c["Sentiment"].astype(int)
    c = c[c.Sentiment.isin([0, 1, 2]) & c["Video ID"].isin(v["Video ID"])].reset_index(drop=True)
    return v.reset_index(drop=True), c


def limpiar(s):
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    s = re.sub(r"[^a-záéíóúñü0-9\s!?']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if n and x.sum() > 0 else 0.0


def variables_comentario(c: pd.DataFrame) -> pd.DataFrame:
    """Léxico (VADER, TextBlob), likes en log y texto limpio por comentario."""
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    va = SentimentIntensityAnalyzer(); c = c.copy()
    c["vader"] = c.Comment.astype(str).map(lambda s: va.polarity_scores(s)["compound"])
    tb = c.Comment.astype(str).map(lambda s: TextBlob(s).sentiment)
    c["tb_pol"] = tb.map(lambda t: t.polarity); c["tb_subj"] = tb.map(lambda t: t.subjectivity)
    c["lw"] = np.log1p(c.Likes.fillna(0).clip(lower=0)); c["text"] = c.Comment.map(limpiar)
    return c


def agregar(g: pd.DataFrame) -> dict:
    """Variables de un video a partir de sus comentarios (requiere la columna `pred` del clasificador A)."""
    w = g.lw + 1
    return dict(pred_pos_share=(g.pred == 2).mean(), pred_neg_share=(g.pred == 0).mean(),
                vader_mean=g.vader.mean(), vader_std=g.vader.std(ddof=1) if len(g) > 1 else 0.0,
                vader_wmean=np.average(g.vader, weights=w), tb_pol=g.tb_pol.mean(), tb_subj=g.tb_subj.mean(),
                clikes_log_mean=g.lw.mean(), clikes_log_max=g.lw.max(), clikes_gini=gini(np.expm1(g.lw)))


class Topicos:
    """TF-IDF + SVD de 20 componentes sobre el texto concatenado de los comentarios de cada video."""

    def fit(self, docs):
        self.tf = TfidfVectorizer(min_df=3, max_df=.6, ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(docs)
        self.svd = TruncatedSVD(20, random_state=SEED).fit(self.tf.transform(docs)); return self

    def transform(self, docs):
        return self.svd.transform(self.tf.transform(docs))


def construir_videos(v: pd.DataFrame, c: pd.DataFrame, topicos: Topicos | None = None):
    """Tabla de videos con las 32 variables del modelo final. Devuelve (tabla, topicos ajustados)."""
    filas = pd.DataFrame({vid: agregar(g) for vid, g in c.groupby("Video ID")}).T
    filas.index.name = "Video ID"; filas = filas.reset_index()
    docs = c.groupby("Video ID").text.apply(" ".join).reindex(filas["Video ID"]).fillna("")
    topicos = topicos or Topicos().fit(docs)
    T = topicos.transform(docs)
    for i in range(20):
        filas[f"topic_{i}"] = T[:, i]
    d = v.merge(filas, on="Video ID", how="inner")
    d["log_age"] = np.log10(d.age_days); d["log_views"] = np.log10(d.Views)
    return d, topicos


def preparar(num=NUM, cat=CAT):
    t = [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)]
    if cat:
        t.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return ColumnTransformer(t)


def modelo_stacking(num=NUM, cat=CAT):
    """Modelo final de la etapa B (Avances III, B8b)."""
    base = [("xgb", make_pipeline(preparar(num, cat), XGBRegressor(n_estimators=800, learning_rate=.06, max_depth=6, subsample=.8,
                                                                  colsample_bytree=.7, reg_lambda=2, random_state=SEED, n_jobs=2))),
            ("hgb", make_pipeline(preparar(num, cat), HistGradientBoostingRegressor(max_iter=500, learning_rate=.05, max_leaf_nodes=31,
                                                                                   l2_regularization=1, random_state=SEED))),
            ("rf", make_pipeline(preparar(num, cat), RandomForestRegressor(500, min_samples_leaf=2, max_features=.5, n_jobs=2, random_state=SEED))),
            ("ridge", make_pipeline(preparar(num, cat), Ridge(alpha=3)))]
    return StackingRegressor(base, final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)), cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=1)


def validar(d: pd.DataFrame, num=NUM, cat=CAT, nivel=.9):
    """Validación cruzada de 5 folds. Los intervalos son conformales por validación cruzada: el ancho de cada
    fold se calcula con los residuos fuera de fold de los otros cuatro, de modo que la cobertura se mide
    sobre datos que no participaron en la calibración."""
    y = d.log_views.values; pred = np.zeros(len(d)); folds = list(KFold(5, shuffle=True, random_state=SEED).split(d))
    for tr, te in folds:
        pred[te] = modelo_stacking(num, cat).fit(d.iloc[tr][num + cat], y[tr]).predict(d.iloc[te][num + cat])
    res = np.abs(y - pred); lo = np.zeros(len(d)); hi = np.zeros(len(d)); r2, mae, cob = [], [], []
    for k, (tr, te) in enumerate(folds):
        otros = np.concatenate([f[1] for j, f in enumerate(folds) if j != k])
        q = np.quantile(res[otros], min(1, np.ceil((len(otros) + 1) * nivel) / len(otros)))
        lo[te], hi[te] = pred[te] - q, pred[te] + q
        r2.append(r2_score(y[te], pred[te])); mae.append(mean_absolute_error(y[te], pred[te])); cob.append(np.mean(res[te] <= q))
    return dict(r2=float(np.mean(r2)), r2_sd=float(np.std(r2)), r2_folds=[float(x) for x in r2], mae=float(np.mean(mae)),
                factor=float(10 ** np.mean(mae)), cobertura=float(np.mean(cob)), ancho_log10=float(np.mean(hi - lo))), pred, lo, hi


def entrenar_final(d: pd.DataFrame, pred_oof, num=NUM, cat=CAT, nivel=.9):
    """Modelo final con todos los videos; el ancho del intervalo sale de los residuos fuera de fold."""
    q = float(np.quantile(np.abs(d.log_views.values - pred_oof), nivel))
    return modelo_stacking(num, cat).fit(d[num + cat], d.log_views.values), q


def predecir_video(lineas: str, categoria: str, dias: float, clasificar, modelo, topicos: Topicos, q: float):
    """Predicción para un video nuevo. `lineas`: un comentario por línea, opcionalmente 'likes | comentario'.
    `clasificar(lista_de_textos)` devuelve las clases 0/1/2 del clasificador de sentimiento."""
    filas = []
    for ln in [x.strip() for x in str(lineas).splitlines() if x.strip()]:
        likes, texto = (ln.split("|", 1) if "|" in ln and ln.split("|", 1)[0].strip().isdigit() else ("0", ln))
        filas.append({"Video ID": "nuevo", "Comment": texto.strip(), "Likes": float(likes)})
    if not filas:
        raise ValueError("Ingrese al menos un comentario.")
    c = variables_comentario(pd.DataFrame(filas)); c["pred"] = clasificar(c.Comment.tolist())
    v = pd.DataFrame({"Video ID": ["nuevo"], "Keyword": [categoria], "Views": [1.0], "age_days": [max(float(dias), 1.0)]})
    d, _ = construir_videos(v, c, topicos)
    lv = float(modelo.predict(d[NUM + CAT])[0])
    return c[["Comment", "Likes", "pred"]], dict(vistas=10 ** lv, minimo=10 ** (lv - q), maximo=10 ** (lv + q),
                                                 positivos=float((c.pred == 2).mean()), negativos=float((c.pred == 0).mean()))
