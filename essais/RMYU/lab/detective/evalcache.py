"""Score quota (user) par pli d'un cache (même partiel). Usage : python lab/detective/evalcache.py <npz>"""
import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from src.validation import load_data
from quota import constrained_predict
tr, te, _ = load_data(); yi = tr.intensity.map({"easy":0,"moderate":1,"hard":2}).values
C = np.load(sys.argv[1]); a, q = [], []
for f in range(5):
    if f"P{f}" not in C.files: break
    va = C[f"va{f}"]; P = C[f"P{f}"]; nv = len(va); lab = np.setdiff1d(np.arange(len(tr)), va)
    a.append(f1_score(yi[va], P[:nv].argmax(1), average="macro"))
    q.append(f1_score(yi[va], constrained_predict(P, np.concatenate([tr.user_id.values[va], te.user_id.values]), tr.user_id.values[lab], yi[lab])[:nv], average="macro"))
print(sys.argv[1], "folds", len(q), "argmax", np.round(a,4).tolist(), "quota", np.round(q,4).tolist(), "mean quota", round(np.mean(q),4))
