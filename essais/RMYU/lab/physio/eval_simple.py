"""Mesure simple : HGB 300 it (lr .05) sur build+add_resid [+ add_physio], 5 plis communs, argmax et quotas par user.
Usage (depuis C:/Users/curie/hk/essais/RMYU) : python -u lab/physio/eval_simple.py <variant> [variant...]
variants : base, phys, phys_light"""
import sys, time; sys.path.append("../.."); sys.path.append("."); sys.path.append("lab/physio")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from src.validation import load_data, get_folds
from feats import build, add_resid
from resid2 import _num
from quota import constrained_predict
from physio import add_physio

train, test, _ = load_data()
t0 = time.time()
F0 = add_resid(build(train, test))
print("base feats", F0.shape, f"{time.time()-t0:.0f}s", flush=True)
y = train.intensity.reset_index(drop=True); yi = y.map({"easy": 0, "moderate": 1, "hard": 2}).values
ntr = (F0.part == "tr").sum()
variants = sys.argv[1:] or ["base", "phys"]
for v in variants:
    if v == "base":
        F = F0
    else:
        F = add_physio(F0, light=(v == "phys_light"))
        print("physio feats", F.shape, f"{time.time()-t0:.0f}s", flush=True)
    X = F.drop(columns=["part", "workout_id"]).reset_index(drop=True)
    A = _num(X); u = X.user_id.values
    Atr, Ate = A[:ntr], A[ntr:]
    a, q = [], []
    oof = np.zeros((ntr, 3))
    for f, (tr_i, va_i) in enumerate(get_folds(y).split(Atr, y)):
        m = HistGradientBoostingClassifier(random_state=42, learning_rate=0.05, max_iter=300).fit(Atr[tr_i], yi[tr_i])
        P = m.predict_proba(np.vstack([Atr[va_i], Ate])); nv = len(va_i); oof[va_i] = P[:nv]
        a.append(f1_score(yi[va_i], P[:nv].argmax(1), average="macro"))
        gu = np.concatenate([u[:ntr][va_i], u[ntr:]])
        q.append(f1_score(yi[va_i], constrained_predict(P, gu, u[:ntr][tr_i], yi[tr_i])[:nv], average="macro"))
        print(f"  {v} fold {f}: argmax {a[-1]:.4f} quota {q[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"RESULT {v}: argmax {np.mean(a):.4f} ± {np.std(a):.4f} | quota {np.mean(q):.4f} ± {np.std(q):.4f} | per-fold quota {np.round(q,4).tolist()}", flush=True)
    np.save(f"lab/physio/oof_simple_{v}.npy", oof)
