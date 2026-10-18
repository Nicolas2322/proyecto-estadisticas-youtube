"""Avances III – Etapa A: iteraciones del clasificador de sentimiento.
Uso: python iter_a.py <videos-stats.csv> <comments.csv> <carpeta_salida>
Cada iteración se evalúa con los mismos 5 folds estratificados y se registra en MLflow.
"""
import sys, os, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse, stats
from joblib import Memory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
warnings.filterwarnings("ignore"); os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
sys.path.insert(0, str(Path(__file__).parent))
from comun import cargar, limpiar_basico, limpiar_mejorado, folds_a, SEED

VID, COM, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
MEM = Memory(str(OUT / "cache_a"), verbose=0)
import mlflow
mlflow.set_tracking_uri("sqlite:///" + str((OUT / "mlflow.db").resolve())); mlflow.set_experiment("etapa_A_sentimiento")

v, c = cargar(VID, COM)
y = c.Sentiment.values; F = folds_a(y)
print("comentarios", len(c), np.bincount(y))

# ------------------------------------------------------------------ variables precalculadas
@MEM.cache
def precalcular(comments):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    import emoji
    va = SentimentIntensityAnalyzer()
    rows = []
    for s in comments:
        s = str(s); p = va.polarity_scores(s); tb = TextBlob(s).sentiment
        rows.append([p["compound"], p["pos"], p["neu"], p["neg"], tb.polarity, tb.subjectivity,
                     np.log1p(len(s.split())), int("?" in s), int("!" in s), emoji.emoji_count(s),
                     sum(ch.isupper() for ch in s) / max(sum(ch.isalpha() for ch in s), 1)])
    return np.array(rows, dtype=float)
LEX_COLS = ["vader_comp", "vader_pos", "vader_neu", "vader_neg", "tb_pol", "tb_subj", "log_len", "has_q", "has_excl", "n_emoji", "upper_ratio"]
lex = precalcular(c.Comment.astype(str).tolist())
df = pd.DataFrame(lex, columns=LEX_COLS)
df["t0"] = c.Comment.map(limpiar_basico); df["t1"] = c.Comment.map(limpiar_mejorado); df["Keyword"] = c.Keyword.values

@MEM.cache
def glove_matrix(texts):
    import gensim.downloader as api
    kv = api.load("glove-twitter-100")
    M = np.zeros((len(texts), 100)); cover = []
    for i, t in enumerate(texts):
        toks = [w for w in t.split() if w in kv.key_to_index]
        cover.append(len(toks) / max(len(t.split()), 1))
        if toks: M[i] = np.mean([kv[w] for w in toks], axis=0)
    return M, float(np.mean(cover))
G, glove_cov = glove_matrix(df.t0.tolist())
for j in range(100): df[f"g{j}"] = G[:, j]
GCOLS = [f"g{j}" for j in range(100)]
print("cobertura GloVe", round(glove_cov, 3))

# ------------------------------------------------------------------ definiciones de modelos
tfw = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
def textfeats(col):
    return ColumnTransformer([("w", TfidfVectorizer(**tfw), col),
                              ("ch", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True), col)])
def hybrid(col, glove=False):
    t = [("w", TfidfVectorizer(**tfw), col),
         ("ch", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True), col),
         ("lex", StandardScaler(), LEX_COLS),
         ("kw", OneHotEncoder(handle_unknown="ignore"), ["Keyword"])]
    if glove: t.append(("glove", make_pipeline(StandardScaler(), FunctionTransformer(lambda X: X * 0.3)), GCOLS))
    return ColumnTransformer(t)
SVM = lambda C=.1: LinearSVC(C=C, class_weight="balanced")
ITER = {
    "A0 Base Avances II (SVM palabras+caracteres)": make_pipeline(textfeats("t0"), SVM()),
    "A1 + Preprocesamiento (emojis, negación, alargamientos)": make_pipeline(textfeats("t1"), SVM()),
    "A2 + Híbrido léxico y contexto (VADER, TextBlob, categoría)": make_pipeline(hybrid("t1"), SVM()),
    "A3a GloVe Twitter solo (regresión logística)": make_pipeline(ColumnTransformer([("g", StandardScaler(), GCOLS)]), LogisticRegression(C=1, max_iter=3000, class_weight="balanced")),
    "A3b + Embeddings GloVe Twitter (híbrido)": make_pipeline(hybrid("t1", glove=True), SVM()),
}

MODELS = dict(ITER)
@MEM.cache
def _run(name, n):
    model = MODELS[name]
    oof = np.zeros(n, int); scores = np.zeros((n, 3)); fold_f1 = []; t0 = time.time()
    for tr, te in F:
        m = __import__("sklearn").base.clone(model).fit(df.iloc[tr], y[tr])
        s = m.decision_function(df.iloc[te]) if hasattr(m, "decision_function") else m.predict_proba(df.iloc[te])
        scores[te] = s; oof[te] = s.argmax(1); fold_f1.append(f1_score(y[te], oof[te], average="macro"))
    return oof, scores, fold_f1, time.time() - t0
def evaluar(name, model):
    MODELS[name] = model
    return _run(name, len(df))

RES, OOF, SC = {}, {}, {}
def registrar(name, oof, fold_f1, sec, extra=None):
    rep = classification_report(y, oof, output_dict=True)
    RES[name] = dict(f1_macro=[float(np.mean(fold_f1)), float(np.std(fold_f1))], folds=[float(x) for x in fold_f1], acc=float(accuracy_score(y, oof)),
                     f1_neg=rep["0"]["f1-score"], f1_neu=rep["1"]["f1-score"], f1_pos=rep["2"]["f1-score"], sec=round(sec, 1), **(extra or {}))
    with mlflow.start_run(run_name=name):
        mlflow.log_param("iteracion", name.split()[0])
        for k, val in RES[name].items():
            if isinstance(val, list) and all(isinstance(z, (int, float)) for z in val): mlflow.log_metric(k, val[0]); mlflow.log_metric(k + "_sd", val[1])
            elif isinstance(val, (int, float)): mlflow.log_metric(k, val)
    print(name, {k: (round(x[0], 4) if isinstance(x, list) and isinstance(x[0], float) else round(x, 4) if isinstance(x, float) else x) for k, x in RES[name].items()}, flush=True)

for name, model in ITER.items():
    oof, sc, ff, sec = evaluar(name, model); OOF[name] = oof; SC[name] = sc; registrar(name, oof, ff, sec)

# ------------------------------------------------------------------ A4: ajuste de sesgos por clase (umbral) anidado
best_lin = "A3b + Embeddings GloVe Twitter (híbrido)" if RES["A3b + Embeddings GloVe Twitter (híbrido)"]["f1_macro"][0] >= RES["A2 + Híbrido léxico y contexto (VADER, TextBlob, categoría)"]["f1_macro"][0] else "A2 + Híbrido léxico y contexto (VADER, TextBlob, categoría)"
GRID = np.round(np.arange(-.6, .61, .05), 2)
def best_bias(S, yy):
    best, bb = -1, (0, 0)
    for b0 in GRID:
        for b1 in GRID:
            f = f1_score(yy, (S + np.array([b0, b1, 0])).argmax(1), average="macro")
            if f > best: best, bb = f, (b0, b1)
    return np.array([bb[0], bb[1], 0.0])
@MEM.cache(ignore=["model"])
def run_a4(name, model, n):
    oof = np.zeros(n, int); ff = []; biases = []; t0 = time.time()
    for tr, te in F:
        inner = cross_val_predict(__import__("sklearn").base.clone(model), df.iloc[tr], y[tr], cv=StratifiedKFold(3, shuffle=True, random_state=SEED), method="decision_function")
        b = best_bias(inner, y[tr]); biases.append(b.tolist())
        m = __import__("sklearn").base.clone(model).fit(df.iloc[tr], y[tr])
        oof[te] = (m.decision_function(df.iloc[te]) + b).argmax(1); ff.append(f1_score(y[te], oof[te], average="macro"))
    return oof, ff, time.time() - t0, biases
name4 = "A4 + Ajuste anidado de umbrales por clase"
oof, ff, sec, biases = run_a4(name4, ITER[best_lin], len(df)); OOF[name4] = oof
registrar(name4, oof, ff, sec, dict(base=best_lin, sesgos_por_fold=biases))

# ------------------------------------------------------------------ A5: stacking (meta-aprendizaje)
def stack_model():
    est = [("svm_hib", make_pipeline(hybrid("t1", glove=True), SVM())),
           ("lr_pal", make_pipeline(ColumnTransformer([("w", TfidfVectorizer(**tfw), "t1")]), LogisticRegression(C=4, max_iter=3000, class_weight="balanced"))),
           ("cnb", make_pipeline(ColumnTransformer([("w", TfidfVectorizer(**tfw), "t1")]), ComplementNB(alpha=.3))),
           ("glove_lr", make_pipeline(ColumnTransformer([("g", StandardScaler(), GCOLS)]), LogisticRegression(C=1, max_iter=3000, class_weight="balanced"))),
           ("lexico", make_pipeline(ColumnTransformer([("l", StandardScaler(), LEX_COLS)]), LogisticRegression(max_iter=3000, class_weight="balanced")))]
    return StackingClassifier(est, final_estimator=LogisticRegression(C=1, max_iter=3000, class_weight="balanced"), cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=1)
name5 = "A5 Stacking (SVM híbrido + LR + NB + GloVe + léxico)"
oof, sc, ff, sec = evaluar(name5, stack_model()); OOF[name5] = oof; SC[name5] = sc; registrar(name5, oof, ff, sec)

# ------------------------------------------------------------------ comparación estadística final vs A0
base = "A0 Base Avances II (SVM palabras+caracteres)"
cands = [k for k in RES if k != base and not k.startswith("A3a")]
final = max(cands, key=lambda k: RES[k]["f1_macro"][0]); RES["_final"] = final
a, b_ = OOF[base] == y, OOF[final] == y
n01, n10 = int((~a & b_).sum()), int((a & ~b_).sum())
chi = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
rng = np.random.default_rng(SEED); diffs = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    diffs.append(f1_score(y[i], OOF[final][i], average="macro") - f1_score(y[i], OOF[base][i], average="macro"))
RES["_comparacion"] = dict(final=final, mejora_f1=RES[final]["f1_macro"][0] - RES[base]["f1_macro"][0], mcnemar=dict(final_acierta_base_falla=n01, base_acierta_final_falla=n10, chi2=float(chi), p=float(stats.chi2.sf(chi, 1))),
                           bootstrap_ic95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])
RES["_cm_final"] = confusion_matrix(y, OOF[final], normalize="true").round(3).tolist()
RES["_cm_base"] = confusion_matrix(y, OOF[base], normalize="true").round(3).tolist()
RES["_glove_cobertura"] = glove_cov
# errores típicos del modelo final (para análisis cualitativo)
err = c.assign(pred=OOF[final])[(OOF[final] != y)]
RES["_ejemplos_error"] = err[(err.Sentiment == 0) & (err.pred == 1)].Comment.str.slice(0, 140).head(6).tolist()
np.savez_compressed(OUT / "oof_a.npz", **{k.split()[0]: v_ for k, v_ in OOF.items()}, y=y)
pd.DataFrame({"Video ID": c["Video ID"], "y": y, "pred_final": OOF[final]}).to_csv(OUT / "oof_sentimiento_final.csv", index=False)
json.dump(RES, open(OUT / "resultados_a.json", "w"), ensure_ascii=False, indent=1)
print("FINAL", final, json.dumps(RES["_comparacion"]))
