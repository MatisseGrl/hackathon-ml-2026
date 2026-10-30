"""HGB simple (lr .05, 300 it) sur build+add_resid [+ add_my_feats], plis communs, quotas par user.
Usage (depuis essais/RMYU) : python -u lab/detective/evalf.py <tag> [none|all|feat1,feat2...] [drop:col,...]"""
import sys, time, os, pickle; sys.path.append("../.."); sys.path.append("."); sys.path.append("lab/detective")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from src.validation import load_data, get_folds
from feats import build, add_resid
from quota import constrained_predict
tag = sys.argv[1]; which = sys.argv[2] if len(sys.argv) > 2 else "none"
train, test, _ = load_data()
cachef = "lab/detective/F_base.pkl"
if os.path.exists(cachef):
    F = pd.read_pickle(cachef)
else:
    F = add_resid(build(train, test)); F.to_pickle(cachef)
base_cols = set(F.columns)
if which != "none":
    from myfeats import add_my_feats
    F = add_my_feats(F)
    new = [c for c in F.columns if c not in base_cols]
    if which != "all":
        keep = which.split(",")
        F = F.drop(columns=[c for c in new if not any(c.startswith(k) for k in keep)])
    print("new cols:", [c for c in F.columns if c not in base_cols], flush=True)
X = F.drop(columns=["part", "workout_id"]).reset_index(drop=True)
ntr = (F.part == "tr").sum(); Xtr, Xte = X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True)
y = train.intensity.reset_index(drop=True); yi = y.map({"easy": 0, "moderate": 1, "hard": 2}).values
a, q = [], []; t0 = time.time(); out = {}
for f, (tr_i, va_i) in enumerate(get_folds(y).split(Xtr, y)):
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, random_state=42).fit(Xtr.iloc[tr_i], yi[tr_i])
    Xu = pd.concat([Xtr.iloc[va_i], Xte]).reset_index(drop=True); nv = len(va_i)
    P = m.predict_proba(Xu); out[f"P{f}"] = P; out[f"va{f}"] = va_i
    a.append(f1_score(yi[va_i], P[:nv].argmax(1), average="macro"))
    q.append(f1_score(yi[va_i], constrained_predict(P, Xu.user_id.values, Xtr.user_id.values[tr_i], yi[tr_i])[:nv], average="macro"))
    print(f"fold {f}: argmax {a[-1]:.4f} quota {q[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
print(f"RESULT {tag} [{which}] => argmax {np.mean(a):.4f} ± {np.std(a):.4f} | quota {np.mean(q):.4f} ± {np.std(q):.4f} | per-fold quota {np.round(q,4).tolist()}", flush=True)
np.savez(f"lab/detective/cache_det_{tag}.npz", **out)
