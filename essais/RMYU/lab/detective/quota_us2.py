import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from src.validation import load_data
from quota import constrained_predict
tr, te, _ = load_data(); yi = tr.intensity.map({"easy":0,"moderate":1,"hard":2}).values
a = pd.concat([tr, te]); ns = a.groupby("user_id").sport.nunique(); multi = set(ns[ns>1].index)
C = np.load(sys.argv[1])
ku = lambda d: d.user_id.astype(str).values
kus = lambda d: (d.user_id.astype(str) + "|" + d.sport.astype(str)).values
acc = {"argmax": [], "q_user": [], "q_user_sport": []}; ys=[]; preds={k:[] for k in acc}
for f in range(5):
    va = C[f"va{f}"]; P = C[f"P{f}"]; nv = len(va); lab = np.setdiff1d(np.arange(len(tr)), va)
    U = pd.concat([tr.iloc[va], te]); L = tr.iloc[lab]
    mm = tr.user_id.iloc[va].isin(multi).values
    ys.append(yi[va][mm]); preds["argmax"].append(P[:nv].argmax(1)[mm])
    for name, k in [("q_user", ku), ("q_user_sport", kus)]:
        preds[name].append(constrained_predict(P, k(U), k(L), yi[lab])[:nv][mm])
y = np.concatenate(ys)
for k in preds:
    p = np.concatenate(preds[k]); print(k, "multi-sport rows", len(y), "acc", (p==y).mean().round(4), "F1", round(f1_score(y, p, average="macro"),4))
