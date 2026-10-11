"""Avances III – Etapa B: iteraciones del modelo de desempeño del video.
Uso: python iter_b.py <features_video.pkl de Avances II> <carpeta_salida (con oof_sentimiento_final.csv)>
"""
import sys, os, json, time, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Memory
from scipy import stats
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, brier_score_loss
from xgboost import XGBRegressor, XGBClassifier
warnings.filterwarnings("ignore"); os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
sys.path.insert(0, str(Path(__file__).parent))
from comun import SEED, folds_b
PKL, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MEM = Memory(str(OUT / "cache_b"), verbose=0)
import mlflow
mlflow.set_tracking_uri("sqlite:///" + str((OUT / "mlflow.db").resolve())); mlflow.set_experiment("etapa_B_desempeno")

d = pd.read_pickle(PKL).reset_index(drop=True)
y = d.log_views.values; F = folds_b(len(d))
STYLE = ["len_mean", "q_share", "excl_share", "emoji_mean", "non_ascii"]
TOP = [f"topic_{i}" for i in range(20)]
SENT0 = ["pred_pos_share", "pred_neg_share", "vader_mean", "vader_std", "vader_wmean", "tb_pol", "tb_subj"]
INTER = ["clikes_log_mean", "clikes_log_max", "clikes_gini"]
BASE_NUM = SENT0 + INTER + TOP + ["log_age"]
CAT = ["Keyword"]

# ------------------------------------------------------------------ nuevas variables
# B1: sentimiento con el clasificador final de la etapa A (predicciones fuera de fold)
oa = pd.read_csv(OUT / "oof_sentimiento_final.csv")
c_lk = pd.read_csv(sys.argv[3]) if len(sys.argv) > 3 else None
g = oa.groupby("Video ID").pred_final
s1 = pd.DataFrame({"a_pos": g.apply(lambda s: (s == 2).mean()), "a_neg": g.apply(lambda s: (s == 0).mean()),
                   "a_neu": g.apply(lambda s: (s == 1).mean()), "a_polar": g.apply(lambda s: ((s == 2).mean() - (s == 0).mean()))}).reset_index()
d = d.merge(s1, on="Video ID", how="left")
SENT1 = ["a_pos", "a_neg", "a_neu", "a_polar", "vader_mean", "vader_std", "vader_wmean", "tb_pol", "tb_subj"]

# B2: variables del título (disponibles al publicar el video)
def title_feats(t):
    t = str(t); letters = [ch for ch in t if ch.isalpha()]
    return dict(t_len=len(t), t_words=len(t.split()), t_upper=sum(ch.isupper() for ch in letters) / max(len(letters), 1),
                t_digits=int(bool(re.search(r"\d", t))), t_q=int("?" in t), t_excl=int("!" in t), t_pipe=int("|" in t or " - " in t),
                t_hash=int("#" in t), t_nonascii=sum(ord(ch) > 127 for ch in t) / max(len(t), 1),
                t_shorts=int("#shorts" in t.lower()), t_official=int("official" in t.lower()))
TF = pd.DataFrame([title_feats(t) for t in d.Title]); d = pd.concat([d, TF], axis=1)
TITLE_META = list(TF.columns)

@MEM.cache
def glove_titles_and_comments(titles, comments_by_video):
    import gensim.downloader as api
    kv = api.load("glove-twitter-100")
    def emb(text):
        toks = [w for w in re.sub(r"[^a-z0-9\s']", " ", str(text).lower()).split() if w in kv.key_to_index]
        return np.mean([kv[w] for w in toks], axis=0) if toks else np.zeros(100)
    return np.vstack([emb(t) for t in titles]), np.vstack([emb(t) for t in comments_by_video])
com = pd.read_csv(sys.argv[3]) if len(sys.argv) > 3 else None
docs = com.drop_duplicates(["Video ID", "Comment"]).groupby("Video ID").Comment.apply(lambda s: " ".join(map(str, s))).reindex(d["Video ID"]).fillna("")
GT, GC = glove_titles_and_comments(d.Title.astype(str).tolist(), docs.tolist())
for j in range(100): d[f"gt{j}"] = GT[:, j]; d[f"gc{j}"] = GC[:, j]
GT_COLS = [f"gt{j}" for j in range(100)]; GC_COLS = [f"gc{j}" for j in range(100)]

# ------------------------------------------------------------------ utilidades
def prep(num, cat=CAT, pca=None):
    t = [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)]
    if cat: t.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    for cols, k in (pca or []):
        t.append((f"pca_{cols[0]}", make_pipeline(StandardScaler(), PCA(k, random_state=SEED)), cols))
    return ColumnTransformer(t)
XGB_P = dict(n_estimators=800, learning_rate=.06, max_depth=6, min_child_weight=1, subsample=.8, colsample_bytree=.7, reg_lambda=2, random_state=SEED, n_jobs=2)
xgb = lambda **kw: XGBRegressor(**{**XGB_P, **kw})

RES = {}; PRED = {}
def cv_eval(name, model, cols, decision=None, target=None, log=True):
    yy = y if target is None else target
    pred = np.zeros(len(d)); r2s, maes = [], []; t0 = time.time()
    for tr, te in F:
        m = clone(model).fit(d.iloc[tr][cols], yy[tr]); p = m.predict(d.iloc[te][cols]); pred[te] = p
        r2s.append(r2_score(yy[te], p)); maes.append(mean_absolute_error(yy[te], p))
    RES[name] = dict(r2=[float(np.mean(r2s)), float(np.std(r2s))], mae=float(np.mean(maes)), factor=float(10 ** np.mean(maes)),
                     spearman=float(stats.spearmanr(yy, pred)[0]), n_vars=len([c_ for c_ in cols if c_ not in CAT]) + (1 if any(c_ in CAT for c_ in cols) else 0),
                     sec=round(time.time() - t0, 1), decision=decision or "")
    PRED[name] = pred
    if log:
        with mlflow.start_run(run_name=name):
            mlflow.log_param("iteracion", name.split()[0]); mlflow.log_param("n_vars", RES[name]["n_vars"])
            mlflow.log_metric("r2", RES[name]["r2"][0]); mlflow.log_metric("r2_sd", RES[name]["r2"][1]); mlflow.log_metric("mae", RES[name]["mae"])
    print(name, round(RES[name]["r2"][0], 4), "±", round(RES[name]["r2"][1], 4), "MAE", round(RES[name]["mae"], 4), "n", RES[name]["n_vars"], flush=True)
    return RES[name]["r2"][0]

# ------------------------------------------------------------------ iteraciones
cur_num, cur_pca = list(BASE_NUM), []
model_of = lambda num, pca: make_pipeline(prep(num, pca=pca), xgb())
cols_of = lambda num, pca: num + CAT + [c_ for cols, _ in pca for c_ in cols]
best = cv_eval("B0 Base Avances II (XGBoost, 32 variables)", model_of(cur_num, cur_pca), cols_of(cur_num, cur_pca), "Punto de partida")

def intentar(name, num, pca, why):
    global cur_num, cur_pca, best
    r = cv_eval(name, model_of(num, pca), cols_of(num, pca))
    if r > best + 0.002:
        RES[name]["decision"] = f"Adoptada (+{r - best:.3f} R²). {why}"; cur_num, cur_pca, best = num, pca, r
    else:
        RES[name]["decision"] = f"Descartada ({r - best:+.3f} R²). {why}"
    return r

intentar("B1 Sentimiento del clasificador mejorado (etapa A final)", [x for x in cur_num if x not in SENT0] + SENT1, cur_pca,
         "Sustituye las proporciones de sentimiento por las del clasificador A final.")
intentar("B2 + Variables del título (longitud, mayúsculas, signos, formato)", cur_num + TITLE_META, cur_pca,
         "El título está disponible al publicar el video.")
intentar("B3 + Embeddings GloVe del título (PCA 10)", cur_num, cur_pca + [(GT_COLS, 10)], "Semántica del título.")
intentar("B4 + Embeddings GloVe de los comentarios (PCA 10)", cur_num, cur_pca + [(GC_COLS, 10)], "Semántica densa de la conversación.")
intentar("B5 Sin tópicos TF-IDF/SVD (reemplazo por embeddings)", [x for x in cur_num if x not in TOP], cur_pca + ([(GC_COLS, 10)] if (GC_COLS, 10) not in cur_pca else []),
         "Prueba si los embeddings sustituyen a los tópicos.")

# B6: selección de variables con SHAP dentro de cada fold
import shap
@MEM.cache
def shap_select(num, pca, keep=0.95):
    sel_r2, sel_n = [], []; pred = np.zeros(len(d)); kept_all = []
    cols = cols_of(num, pca)
    for tr, te in F:
        pre = prep(num, pca=pca).fit(d.iloc[tr][cols]); Xtr = pre.transform(d.iloc[tr][cols]); Xte = pre.transform(d.iloc[te][cols])
        m = xgb().fit(Xtr, y[tr]); sv = np.abs(shap.TreeExplainer(m).shap_values(Xtr)).mean(0)
        order = np.argsort(-sv); cum = np.cumsum(sv[order]) / sv.sum(); k = int(np.searchsorted(cum, keep) + 1); idx = order[:k]
        m2 = xgb().fit(Xtr[:, idx], y[tr]); p = m2.predict(Xte[:, idx]); pred[te] = p
        sel_r2.append(r2_score(y[te], p)); sel_n.append(k); kept_all.append(Xtr.shape[1])
    return float(np.mean(sel_r2)), float(np.std(sel_r2)), sel_n, kept_all, pred
r2s, sd, ns, tot, pred = shap_select(cur_num, cur_pca)
name = "B6 Selección por SHAP (95 % de la importancia acumulada)"
RES[name] = dict(r2=[r2s, sd], mae=float(mean_absolute_error(y, pred)), factor=float(10 ** mean_absolute_error(y, pred)), spearman=float(stats.spearmanr(y, pred)[0]),
                 n_vars=f"{int(np.mean(ns))} de {tot[0]} columnas", sec=0,
                 decision=("Adoptada" if r2s > best + 0.002 else "Descartada") + f" ({r2s - best:+.3f} R²). Conserva en promedio {np.mean(ns):.0f} de {tot[0]} columnas transformadas.")
PRED[name] = pred; print(name, round(r2s, 4), ns)
with mlflow.start_run(run_name=name): mlflow.log_metric("r2", r2s); mlflow.log_metric("r2_sd", sd)

# B7: Optuna (TPE) con validación cruzada anidada
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
FINAL_COLS = cols_of(cur_num, cur_pca)
@MEM.cache
def optuna_nested(num, pca, n_trials=25):
    cols = cols_of(num, pca); out_r2, out_mae, params, pred, hist = [], [], [], np.zeros(len(d)), []
    for k, (tr, te) in enumerate(F):
        dtr = d.iloc[tr]; ytr = y[tr]; inner = list(KFold(3, shuffle=True, random_state=SEED).split(dtr))
        def objective(trial):
            p = dict(n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100), learning_rate=trial.suggest_float("learning_rate", .01, .2, log=True),
                     max_depth=trial.suggest_int("max_depth", 3, 8), min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                     subsample=trial.suggest_float("subsample", .6, 1.0), colsample_bytree=trial.suggest_float("colsample_bytree", .4, 1.0),
                     reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True), reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True))
            sc = []
            for itr, ite in inner:
                m = make_pipeline(prep(num, pca=pca), xgb(**p)).fit(dtr.iloc[itr][cols], ytr[itr])
                sc.append(r2_score(ytr[ite], m.predict(dtr.iloc[ite][cols])))
            return np.mean(sc)
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)); st.optimize(objective, n_trials=n_trials)
        m = make_pipeline(prep(num, pca=pca), xgb(**st.best_params)).fit(dtr[cols], ytr); p = m.predict(d.iloc[te][cols]); pred[te] = p
        out_r2.append(r2_score(y[te], p)); out_mae.append(mean_absolute_error(y[te], p)); params.append(st.best_params)
        hist.append([t.value for t in st.trials]); print("optuna fold", k, round(out_r2[-1], 4), flush=True)
    return out_r2, out_mae, params, pred, hist
o_r2, o_mae, o_params, o_pred, o_hist = optuna_nested(cur_num, cur_pca)
name = "B7 Optimización bayesiana (Optuna, CV anidada 5×3, 25 ensayos)"
RES[name] = dict(r2=[float(np.mean(o_r2)), float(np.std(o_r2))], mae=float(np.mean(o_mae)), factor=float(10 ** np.mean(o_mae)), spearman=float(stats.spearmanr(y, o_pred)[0]),
                 n_vars=RES[[k for k in RES if not k.startswith("B6")][-1]]["n_vars"], sec=0, params_por_fold=o_params,
                 decision=("Adoptada" if np.mean(o_r2) > best + 0.002 else "Descartada") + f" ({np.mean(o_r2) - best:+.3f} R²). Estimación sin sesgo del ajuste.")
PRED[name] = o_pred; print(name, RES[name]["r2"]); OPT_HIST = o_hist
with mlflow.start_run(run_name=name): mlflow.log_metric("r2", float(np.mean(o_r2)))
if np.mean(o_r2) > best + 0.002:
    best = float(np.mean(o_r2))
# hiperparámetros de consenso (mediana entre folds) para el modelo final
cons = {k: (float(np.median([p[k] for p in o_params])) if isinstance(o_params[0][k], float) else int(np.median([p[k] for p in o_params]))) for k in o_params[0]}
use_opt = np.mean(o_r2) > RES["B0 Base Avances II (XGBoost, 32 variables)"]["r2"][0] and np.mean(o_r2) >= best - 0.002
XF = (lambda: xgb(**cons)) if RES[name]["decision"].startswith("Adoptada") else (lambda: xgb())
RES["_hiper_consenso"] = cons

# B8: ensambles (¿basta un solo modelo?)
pre = lambda: prep(cur_num, pca=cur_pca)
base_models = [("xgb", make_pipeline(pre(), XF())),
               ("hgb", make_pipeline(pre(), HistGradientBoostingRegressor(max_iter=500, learning_rate=.05, max_leaf_nodes=31, l2_regularization=1, random_state=SEED))),
               ("rf", make_pipeline(pre(), RandomForestRegressor(500, min_samples_leaf=2, max_features=.5, n_jobs=2, random_state=SEED))),
               ("ridge", make_pipeline(pre(), Ridge(alpha=3)))]
best_single = max(RES[k]["r2"][0] for k in RES if not k.startswith("_"))
for nm, mdl in base_models[1:]:
    cv_eval(f"B8 · {dict(hgb='HistGradientBoosting', rf='Random Forest', ridge='Ridge')[nm]} (mismas variables)", mdl, FINAL_COLS, "Referencia individual para el ensamble")
cv_eval("B8a Ensamble por promedio (XGB + HGB + RF + Ridge)", VotingRegressor(base_models), FINAL_COLS)
cv_eval("B8b Stacking (meta-modelo RidgeCV)", StackingRegressor(base_models, final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)), cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=1), FINAL_COLS)
for nm in ["B8a Ensamble por promedio (XGB + HGB + RF + Ridge)", "B8b Stacking (meta-modelo RidgeCV)"]:
    r = RES[nm]["r2"][0]; ref = max(best, RES[name]["r2"][0])
    RES[nm]["decision"] = ("Adoptada" if r > ref + 0.002 else "Descartada") + f" ({r - ref:+.3f} R² frente al mejor modelo individual)."
final_name = max([k for k in RES if not k.startswith("_") and not k.startswith("B8 ·") and not k.startswith("B6")], key=lambda k: RES[k]["r2"][0])
RES["_final"] = final_name
# prueba de Wilcoxon por fold entre el final y B0
f0 = [r2_score(y[te], PRED["B0 Base Avances II (XGBoost, 32 variables)"][te]) for tr, te in F]
f1_ = [r2_score(y[te], PRED[final_name][te]) for tr, te in F]
err0 = np.abs(y - PRED["B0 Base Avances II (XGBoost, 32 variables)"]); err1 = np.abs(y - PRED[final_name])
RES["_comparacion"] = dict(final=final_name, delta_r2=float(np.mean(f1_) - np.mean(f0)), folds_mejor=int(sum(a > b for a, b in zip(f1_, f0))),
                           wilcoxon_errores_abs_p=float(stats.wilcoxon(err1, err0).pvalue), mae_final=float(err1.mean()), mae_base=float(err0.mean()))

# ------------------------------------------------------------------ análisis complementarios
# B9: robustez – sin likes de comentarios (solo texto + metadatos) y objetivo vistas/día
noint = [x for x in cur_num if x not in INTER]
cv_eval("B9a Robustez: sin likes de comentarios", make_pipeline(prep(noint, pca=cur_pca), XF()), noint + CAT + [c_ for cols, _ in cur_pca for c_ in cols], "Análisis de sensibilidad")
y_vpd = np.log10(d.Views / d.age_days)
cv_eval("B9b Objetivo alternativo: log10(vistas por día)", make_pipeline(prep([x for x in cur_num if x != "log_age"], pca=cur_pca), XF()),
        [x for x in cur_num if x != "log_age"] + CAT + [c_ for cols, _ in cur_pca for c_ in cols], "Análisis de sensibilidad", target=y_vpd.values)

# B10: intervalos de predicción conformales (90 %)
cov, width = [], []; lo_all = np.zeros(len(d)); hi_all = np.zeros(len(d)); pc_all = np.zeros(len(d))
for tr, te in F:
    rng = np.random.default_rng(SEED); trp = rng.permutation(tr); cut = int(.8 * len(trp)); fit_i, cal_i = trp[:cut], trp[cut:]
    m = make_pipeline(prep(cur_num, pca=cur_pca), XF()).fit(d.iloc[fit_i][FINAL_COLS], y[fit_i])
    res = np.abs(y[cal_i] - m.predict(d.iloc[cal_i][FINAL_COLS])); q = np.quantile(res, np.ceil((len(cal_i) + 1) * .9) / len(cal_i))
    p = m.predict(d.iloc[te][FINAL_COLS]); lo_all[te], hi_all[te], pc_all[te] = p - q, p + q, p
    cov.append(np.mean((y[te] >= p - q) & (y[te] <= p + q))); width.append(2 * q)
RES["_conformal"] = dict(cobertura=float(np.mean(cov)), cobertura_folds=[float(x) for x in cov], ancho_log10=float(np.mean(width)), factor=float(10 ** (np.mean(width) / 2)))
print("conformal", RES["_conformal"])

# Clasificación de éxito relativo con las variables finales
yc = d.high.values; skf = StratifiedKFold(5, shuffle=True, random_state=SEED); CL = {}
for nm, mdl in {"Regresión logística (C = 100)": LogisticRegression(C=100, max_iter=5000),
                "XGBoost clasificador": XGBClassifier(n_estimators=600, learning_rate=.05, max_depth=4, subsample=.8, colsample_bytree=.7, random_state=SEED, n_jobs=2, eval_metric="logloss")}.items():
    pr = np.zeros(len(d)); aucs = []
    for tr, te in skf.split(d, yc):
        m = make_pipeline(prep(cur_num, pca=cur_pca), mdl).fit(d.iloc[tr][FINAL_COLS], yc[tr]); pr[te] = m.predict_proba(d.iloc[te][FINAL_COLS])[:, 1]; aucs.append(roc_auc_score(yc[te], pr[te]))
    CL[nm] = dict(auc=[float(np.mean(aucs)), float(np.std(aucs))], brier=float(brier_score_loss(yc, pr))); PRED["cls_" + nm] = pr
RES["_clasificacion"] = CL; print(CL)

# SHAP del modelo final (todas las observaciones)
pre_f = prep(cur_num, pca=cur_pca).fit(d[FINAL_COLS]); Xf = pre_f.transform(d[FINAL_COLS])
names = [n.split("__", 1)[1] for n in pre_f.get_feature_names_out()]
mf = XF().fit(Xf, y); sv = shap.TreeExplainer(mf).shap_values(Xf)
imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
RES["_shap_top"] = imp.head(15).round(4).to_dict()
np.savez_compressed(OUT / "shap_b.npz", sv=sv, X=Xf, names=np.array(names))
RES["_final_num"] = cur_num; RES["_final_pca"] = [[cols[0], k] for cols, k in cur_pca]
np.savez_compressed(OUT / "pred_b.npz", y=y, lo=lo_all, hi=hi_all, pc=pc_all, **{k.split()[0].replace("·", "x"): v_ for k, v_ in PRED.items() if not k.startswith("cls_")})
json.dump(dict(RES=RES, OPT_HIST=OPT_HIST), open(OUT / "resultados_b.json", "w"), ensure_ascii=False, indent=1, default=str)
print("FINAL", final_name, RES["_comparacion"])
