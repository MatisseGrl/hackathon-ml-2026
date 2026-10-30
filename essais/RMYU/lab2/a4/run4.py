import sys; sys.path.append("../.."); sys.path.append("."); sys.path.append("lab2/a4")
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
WS = {"M": {"month": 1}, "S": {"slot": 1}, "R": {"route": 1}, "MSR": {"month": 1, "slot": 2, "route": 2}}
for useT in [False, True]:
    lab, rows = ["ref"], [s0]
    for wn, wts in WS.items():
        for a in [-0.3, -0.2, -0.1, 0.05, 0.1, 0.15, 0.2, 0.3]:
            s = []
            for b in base:
                va = b["va"]
                U_rows, U_V = (np.concatenate([va, ntr + np.arange(len(te))]), np.vstack([b["Qv"], b["Qt"]])) if useT else (va, b["Qv"])
                Q = smooth(b["Qv"], keys, va, U_rows, U_V, b["lab"], yi[b["lab"]], wts, a, 0.0, b["fixed"])
                Q = np.clip(Q, 1e-6, None); Q /= Q.sum(1, keepdims=True)
                s.append(f1_score(yi[va], B.predict(Q, b["pl"]), average="macro"))
            lab.append((wn, a)); rows.append(np.array(s))
    M = np.vstack(rows); nest, idx = B.nested(M)
    best = int(np.argmax(M.mean(1)))
    print(f"T={int(useT)} {len(lab)} cand. | meilleur grille {lab[best]} {M[best].mean():.4f} | imbriqué {nest.mean():.4f} ± {nest.std():.4f} d={np.round(nest-s0,4)} win={(nest>s0).sum()} choix {[lab[i] for i in idx]}", flush=True)
    np.save(f"lab2/a4/grid_T{int(useT)}.npy", {"lab": lab, "S": M}, allow_pickle=True)
