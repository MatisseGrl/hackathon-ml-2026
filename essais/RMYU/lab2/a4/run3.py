import sys, time; sys.path.append("../.."); sys.path.append("."); sys.path.append("lab2/a4")
import numpy as np, pandas as pd
import blendn as B
from sklearn.metrics import f1_score
from trans import level_keys, smooth
names = ["person", "f2", "neighbor", "person7", "resid2"]
folds = B.prepare(names, "pct", "user_sport", oracle=True)
s0, _ = B.score(folds, names, np.full(5, 0.2))
tr, te = B.train, B.test; ntr = len(tr)
keys = level_keys(pd.concat([tr, te], ignore_index=True)); yi = B.yi
base = []
for fd in folds:
    va = fd["va"]; nv = len(va)
    Qv = sum(fd["Q"][n] for n in names) / 5; Qt = sum(fd["R"][n][nv:] for n in names) / 5
    base.append(dict(va=va, Qv=Qv, Qt=Qt, lab=np.setdiff1d(np.arange(ntr), va), fixed=Qv.max(1) > 0.999, pl=fd["pl"]))
res = {}
for useT in [False, True]:
    for wn, wts in {"S": {"slot": 1}, "R": {"route": 1}, "M": {"month": 1}}.items():
        for a in [-0.3, -0.2, -0.1]:
            s = []
            for b in base:
                va = b["va"]
                U_rows, U_V = (np.concatenate([va, ntr + np.arange(len(te))]), np.vstack([b["Qv"], b["Qt"]])) if useT else (va, b["Qv"])
                Q = smooth(b["Qv"], keys, va, U_rows, U_V, b["lab"], yi[b["lab"]], wts, a, 0.0, b["fixed"])
                Q = np.clip(Q, 1e-6, None); Q /= Q.sum(1, keepdims=True)
                s.append(f1_score(yi[va], B.predict(Q, b["pl"]), average="macro"))
            s = np.array(s); res[(useT, wn, a)] = s
            print(f"T={int(useT)} {wn} a={a:+.2f} : {s.mean():.4f} ± {s.std():.4f} d={np.round(s-s0,4)} win={(s>s0).sum()}", flush=True)
sys.exit()
r1 = np.load("lab2/a4/res1.npy", allow_pickle=True).item()["res"]
for useT in [False, True]:
    ks = [k for k in r1 if k[0] == useT]; ks3 = [k for k in res if k[0] == useT]
    M = np.vstack([s0] + [r1[k] for k in ks] + [res[k] for k in ks3]); lab = ["ref"] + ks + ks3
    nest, idx = B.nested(M)
    print(f"T={int(useT)} imbriqué (run1+run3, {len(lab)} cand.) {nest.mean():.4f} ± {nest.std():.4f} d={np.round(nest-s0,4)} win={(nest>s0).sum()} choix", [lab[i] for i in idx], flush=True)
