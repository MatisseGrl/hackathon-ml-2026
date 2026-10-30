"""Objectif du décodage sous quotas (protocole fidèle : comptes réels de la validation, user + sport), mélange égal des 5 caches.
- logP : max somme log P (actuel)            - P : max somme P (exactitude espérée ; macro-F1 ≈ exactitude quand les comptes sont fixés)
- sinkhorn : probas recalées pour que, dans chaque groupe, leur somme par classe égale les quotas, puis max somme P."""
import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np
from sklearn.metrics import f1_score
import blendn as B
from quota import assign

names = ["person", "f2", "neighbor", "person7", "resid2"]
folds = B.prepare(names, "pct", "user_sport", oracle=True)
w = np.full(5, 0.2)

def sinkhorn(P, rem, it=50):
    P = np.clip(P, 1e-9, 1); lam = np.ones(3)
    for _ in range(it):
        A = P * lam; A /= A.sum(1, keepdims=True)
        lam *= np.clip(rem, 1e-3, None) / np.clip(A.sum(0), 1e-9, None)
    A = P * lam; return A / A.sum(1, keepdims=True)

res = {k: [] for k in ["logP", "P", "sinkhorn+P", "sinkhorn+logP"]}
for fd in folds:
    nv = len(fd["va"]); Q = sum(wi * fd["Q"][n] for wi, n in zip(w, names)); yv = B.yi[fd["va"]]
    preds = {k: Q.argmax(1).copy() for k in res}
    for rows, rem in fd["pl"]:
        q = Q[rows]
        preds["logP"][rows] = assign(q, rem)
        preds["P"][rows] = assign(np.exp(q - 1), rem)
        s = sinkhorn(q, rem)
        preds["sinkhorn+P"][rows] = assign(np.exp(s - 1), rem)
        preds["sinkhorn+logP"][rows] = assign(s, rem)
    for k in res:
        res[k].append(f1_score(yv, preds[k][:nv], average="macro"))
base = np.array(res["logP"])
for k, v in res.items():
    v = np.array(v); d = v - base
    print(f"{k:14s} {v.mean():.4f} ± {v.std():.4f} | Δ logP {d.mean():+.4f} plis gagnés {(d > 0).sum()}/5 {np.round(d, 4)}", flush=True)
