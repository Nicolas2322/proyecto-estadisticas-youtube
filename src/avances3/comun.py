"""Funciones comunes de carga y limpieza (idénticas a las de Avances II) para las iteraciones de Avances III."""
import re
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

SEED = 42


def cargar(vid_path, com_path):
    v = pd.read_csv(vid_path)
    c = pd.read_csv(com_path)
    v = v.drop(columns=[x for x in ["Unnamed: 0"] if x in v]); c = c.drop(columns=[x for x in ["Unnamed: 0"] if x in c])
    v = v.drop_duplicates("Video ID").dropna(subset=["Views"])
    for col in ["Likes", "Comments"]:
        v.loc[v[col] < 0, col] = np.nan
    v = v[v.Views > 0].copy()
    v["Published At"] = pd.to_datetime(v["Published At"], errors="coerce")
    v["age_days"] = (v["Published At"].max() - v["Published At"]).dt.days.clip(lower=1)
    c = c.dropna(subset=["Comment", "Sentiment"]).drop_duplicates(["Video ID", "Comment"])
    c["Sentiment"] = c["Sentiment"].astype(int)
    c = c[c.Sentiment.isin([0, 1, 2]) & c["Video ID"].isin(v["Video ID"])].reset_index(drop=True)
    c = c.merge(v[["Video ID", "Keyword"]], on="Video ID", how="left")
    return v.reset_index(drop=True), c


def limpiar_basico(s):
    """Limpieza de Avances II."""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    s = re.sub(r"[^a-záéíóúñü0-9\s!?']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


NEG = {"not", "no", "never", "nothing", "nobody", "none", "nor", "cannot", "dont", "doesnt", "didnt", "isnt", "wasnt",
       "arent", "werent", "wont", "wouldnt", "shouldnt", "couldnt", "cant", "aint", "hardly", "without"}


def limpiar_mejorado(s):
    """Iteración A1: emojis a texto, alargamientos, contracciones negativas y marcado de negación."""
    import emoji
    s = emoji.demojize(str(s), delimiters=(" EMO_", " "))
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " URL ", s)
    s = re.sub(r"@\w+", " USER ", s)
    s = re.sub(r"n't\b", " not", s).replace("’", "'")
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)                  # sooooo -> soo
    s = re.sub(r"!{2,}", " MULTIEXCL ", s); s = re.sub(r"\?{2,}", " MULTIQ ", s)
    s = re.sub(r"[^a-z0-9_\s!?']", " ", s)
    toks, out, neg = s.split(), [], 0
    for t in toks:
        if t in {"!", "?", ".", ","} or t.endswith(("!", "?")):
            neg = 0
        out.append(("NEG_" + t) if neg > 0 and t not in NEG else t)
        if t in NEG: neg = 3                                   # ventana de 3 tokens
        elif neg > 0: neg -= 1
    return " ".join(out)


def folds_a(y):
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def folds_b(n):
    return list(KFold(5, shuffle=True, random_state=SEED).split(np.zeros(n)))
