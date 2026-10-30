"""Décodage des quotas : affectation -log P (actuel) contre tri par score scalaire E[y] = P(moderate) + 2 P(hard)
dans chaque user (les e plus bas -> easy, les h plus hauts -> hard). Soumission 3 (0,6 person + 0,4 f2)."""
import sys, time; sys.path.append("../.."); sys.path.append(".")
import numpy as np
from sklearn.metrics import f1_score
import blendn as B

def predict_rank(Q, pl, mode):
    pred = Q.argmax(1).copy()
    s = Q[:, 1] + 2 * Q[:, 2] if mode == "ey" else np.log(Q[:, 2] + 1e-9) - np.log(Q[:, 0] + 1e-9)
    for rows, rem in pl:
        o = rows[np.argsort(s[rows], kind="stable")]
        pred[o[:rem[0]]] = 0; pred[o[rem[0]:rem[0] + rem[1]]] = 1; pred[o[rem[0] + rem[1]:]] = 2
    return pred

names = ["person", "f2"]; w = [0.6, 0.4]
for rule in ("frac", "pct"):
    folds = B.prepare(names, rule)
    t0 = time.time(); s, _ = B.score(folds, names, np.array(w)); dt = time.time() - t0
    out = {"assign": s}
    for mode in ("ey", "logit"):
        r = []
        for fd in folds:
            nv = len(fd["va"]); Q = sum(wi * fd["Q"][n] for wi, n in zip(w, names))
            r.append(f1_score(B.yi[fd["va"]], predict_rank(Q, fd["pl"], mode)[:nv], average="macro"))
        out[mode] = np.array(r)
    for k, v in out.items():
        print(f"{rule:5s} {k:7s} {v.mean():.4f} ± {v.std():.4f} plis {np.round(v, 4)}", flush=True)
    print(f"(un score assign : {dt:.1f}s)")
