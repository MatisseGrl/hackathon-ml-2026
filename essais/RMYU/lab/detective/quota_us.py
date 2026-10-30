"""Compare quotas par user vs par (user, sport) sur un cache au format cache.py (P{f}, va{f})."""
import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from src.validation import load_data
from quota import constrained_predict
tr, te, _ = load_data(); yi = tr.intensity.map({"easy":0,"moderate":1,"hard":2}).values
C = np.load(sys.argv[1])
ku = lambda d: d.user_id.astype(str).values
kus = lambda d: (d.user_id.astype(str) + "|" + d.sport.astype(str)).values
res = {"argmax": [], "q_user": [], "q_user_sport": []}
multi = tr.groupby("user_id").sport.nunique(); 
for f in range(5):
    va = C[f"va{f}"]; P = C[f"P{f}"]; nv = len(va); lab = np.setdiff1d(np.arange(len(tr)), va)
    U = pd.concat([tr.iloc[va], te]); L = tr.iloc[lab]
    res["argmax"].append(f1_score(yi[va], P[:nv].argmax(1), average="macro"))
    for name, k in [("q_user", ku), ("q_user_sport", kus)]:
        pred = constrained_predict(P, k(U), k(L), yi[lab])
        res[name].append(f1_score(yi[va], pred[:nv], average="macro"))
for k, v in res.items(): print(f"{sys.argv[1]} {k:14s} {np.mean(v):.4f} ± {np.std(v):.4f}  {np.round(v,4).tolist()}")
