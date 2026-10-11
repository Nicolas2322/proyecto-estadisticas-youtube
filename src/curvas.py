"""Curvas de aprendizaje: ¿son suficientes los datos? (etapa A y etapa B)
Uso: python curvas.py <videos-stats.csv> <comments.csv> <carpeta_salida>
Requiere haber ejecutado analisis.py (usa out/features_video.pkl).
"""
import sys, json, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")
VID, COM, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3]); SEED = 42
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"],
                     "axes.spines.top": False, "axes.spines.right": False})
R = json.load(open(OUT / "resultados.json"))
c = pd.read_csv(COM).dropna(subset=["Comment"]).drop_duplicates(["Video ID", "Comment"])
v = pd.read_csv(VID).drop_duplicates("Video ID")
c = c[c["Video ID"].isin(v["Video ID"])]; c = c[c.Sentiment.isin([0, 1, 2])]
def clean_text(s):
    s = str(s).lower(); s = re.sub(r"http\S+|www\.\S+", " ", s); s = re.sub(r"[@#]\w+", " ", s)
    s = re.sub(r"[^a-záéíóúñü0-9\s!?']", " ", s); return re.sub(r"\s+", " ", s).strip()
X = c.Comment.map(clean_text).values; y = c.Sentiment.astype(int).values
bp = R["tuning"]["A"]["best_params"]
svm = Pipeline([("f", FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("ch", TfidfVectorizer(analyzer="char_wb", ngram_range=eval(bp["f__ch__ngram_range"]), min_df=3, sublinear_tf=True))])),
        ("m", LinearSVC(C=float(bp["m__C"]), class_weight="balanced"))])
sizes = np.array([.1, .25, .5, .75, 1.0])
nA, trA, teA = learning_curve(svm, X, y, train_sizes=sizes, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="f1_macro", n_jobs=2, shuffle=True, random_state=SEED)
d = pd.read_pickle(OUT / "features_video.pkl")
num, cat = R["_final_num"], ["Keyword"]
bpB = {k.replace("m__", ""): eval(v) if v.replace(".", "").isdigit() else v for k, v in R["tuning"]["B_reg"]["best_params"].items()}
pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
xgb = make_pipeline(pre, XGBRegressor(subsample=.8, colsample_bytree=.7, reg_lambda=2, random_state=SEED, n_jobs=1, **bpB))
nB, trB, teB = learning_curve(xgb, d[num + cat], d.log_views, train_sizes=sizes, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2", n_jobs=2, shuffle=True, random_state=SEED)
LC = dict(A=dict(n=nA.tolist(), train=trA.mean(1).tolist(), test=teA.mean(1).tolist(), test_sd=teA.std(1).tolist()),
          B=dict(n=nB.tolist(), train=trB.mean(1).tolist(), test=teB.mean(1).tolist(), test_sd=teB.std(1).tolist()))
json.dump(LC, open(OUT / "curvas.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for a, n, tr, te, lab, t in [(ax[0], nA, trA, teA, "F1-macro", "Etapa A: comentarios de entrenamiento"), (ax[1], nB, trB, teB, "R²", "Etapa B: videos de entrenamiento")]:
    a.plot(n, tr.mean(1), "o--", color="#81b29a", label="Entrenamiento")
    a.plot(n, te.mean(1), "o-", color="#2a6f97", label="Validación")
    a.fill_between(n, te.mean(1) - te.std(1), te.mean(1) + te.std(1), color="#2a6f97", alpha=.15)
    a.set_xlabel(t); a.set_ylabel(lab); a.legend(frameon=False, fontsize=8)
ax[0].set_title("Curva de aprendizaje – clasificador de sentimiento"); ax[1].set_title("Curva de aprendizaje – predicción de log10(vistas)")
plt.tight_layout(); plt.savefig(OUT / "f7_curvas.png"); plt.close()
print(json.dumps(LC))
