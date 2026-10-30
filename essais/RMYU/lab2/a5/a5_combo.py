import sys, os
os.chdir(r"C:\Users\curie\hk\essais\RMYU")
sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
import blendn as B
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
names = ["person", "f2", "neighbor", "person7", "resid2"]
folds = B.prepare(names, "pct", "user_sport", oracle=True)
tr = B.train.reset_index(drop=True); y = B.yi; ref = np.load("lab2/a5/ref_folds.npy")
def nbstats(cols):
    k = tr[cols].astype(str).agg("_".join, axis=1).values; S = []
    for fd in folds:
        va = fd["va"]; lab = np.setdiff1d(np.arange(len(tr)), va)
        g = pd.DataFrame(dict(k=k[lab], y=y[lab] - 1.0)).groupby("k").y.agg(["sum", "size"]); kv = pd.Series(k[va])
        S.append((kv.map(g["sum"]).fillna(0).values, kv.map(g["size"]).fillna(0).values))
    return S
NB = {"ymw": nbstats(["user_id", "year", "month", "weekday"]), "ymwh": nbstats(["user_id", "year", "month", "weekday", "start_hour_utc"])}
def evalc(a, key, beta, lam):
    out = []
    for f, fd in enumerate(folds):
        va = fd["va"]; P = sum(fd["Q"][n] for n in names) / 5
        E = np.stack([fd["Q"][n] @ np.array([0, 1, 2.]) for n in names], 1); sd = E.std(1)
        tau = np.clip((np.maximum(sd, 1e-3) / np.median(sd)) ** (-a), 0.25, 4)
        L = np.log(np.clip(P, 1e-9, 1)) * tau[:, None]
        if beta > 0:
            s_, n_ = NB[key][f]; L = L + beta * (s_ / (n_ + lam))[:, None] * np.array([-1, 0, 1.])
        Pn = np.exp(L - L.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True)
        out.append(f1_score(y[va], B.predict(Pn, fd["pl"]), average="macro"))
    return np.array(out)
C = [(a, k, b, l) for a in [0.0, 0.25, 0.5] for (k, b, l) in [("ymw", 0, 1), ("ymw", 0.25, 1.0), ("ymw", 0.5, 4.0), ("ymw", 0.25, 4.0), ("ymwh", 0.25, 1.0), ("ymwh", 0.5, 4.0)]]
S = np.array(Parallel(n_jobs=2)(delayed(evalc)(*c) for c in C))
for c, sc in zip(C, S):
    print(c, round(sc.mean(), 4), np.round(sc - ref, 4), "won", int((sc > ref).sum()), flush=True)
nest, idx = B.nested(S)
print("nested", round(nest.mean(), 4), "+-", round(nest.std(), 4), np.round(nest - ref, 4), "won", int((nest > ref).sum()), [C[i] for i in idx])
np.savez("lab2/a5/combo_grid.npz", S=S)
