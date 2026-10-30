import sys, time, itertools; sys.path.append("../.."); sys.path.append("."); sys.path.append("lab2/a4")
import numpy as np, pandas as pd
import blendn as B
from sklearn.metrics import f1_score
from trans import level_keys, smooth
t0 = time.time()
names = ["person", "f2", "neighbor", "person7", "resid2"]
folds = B.prepare(names, "pct", "user_sport", oracle=True)
s0, _ = B.score(folds, names, np.full(5, 0.2))
print(f"ref {s0.mean():.4f} ± {s0.std():.4f} {np.round(s0,4)}  ({time.time()-t0:.0f}s)", flush=True)
tr, te = B.train, B.test; ntr = len(tr)
al = pd.concat([tr, te], ignore_index=True)
keys = level_keys(al)
yi = B.yi
base = []
for fd in folds:
    va = fd["va"]; nv = len(va)
    Qv = sum(fd["Q"][n] for n in names) / 5
    Qt = sum(fd["R"][n][nv:] for n in names) / 5
    lab = np.setdiff1d(np.arange(ntr), va)
    base.append(dict(va=va, Qv=Qv, Qt=Qt, lab=lab, fixed=Qv.max(1) > 0.999, pl=fd["pl"]))
def run(wts, alpha, beta, useT):
    s = []
    for b in base:
        va = b["va"]
        if useT:
            U_rows = np.concatenate([va, ntr + np.arange(len(te))]); U_V = np.vstack([b["Qv"], b["Qt"]])
        else:
            U_rows = va; U_V = b["Qv"]
        Q = smooth(b["Qv"], keys, va, U_rows, U_V, b["lab"], yi[b["lab"]], wts, alpha, beta, b["fixed"])
        s.append(f1_score(yi[va], B.predict(Q, b["pl"]), average="macro"))
    return np.array(s)
chk = run({"month": 1}, 0.0, 0.0, False)
print("check alpha=0 :", np.round(chk, 4), flush=True)
WS = {"M": {"month": 1}, "S": {"slot": 1}, "R": {"route": 1}, "MSR": {"month": 1, "slot": 2, "route": 2}}
res = {}
for useT in [False, True]:
    for wn, wts in WS.items():
        for a in [0.1, 0.2, 0.3, 0.5]:
            s = run(wts, a, 0.0, useT)
            res[(useT, wn, a, 0.0)] = s
            print(f"T={int(useT)} {wn:4s} a={a:.1f} : {s.mean():.4f} ± {s.std():.4f} d={np.round(s-s0,4)} win={(s>s0).sum()}  ({time.time()-t0:.0f}s)", flush=True)
np.save("lab2/a4/res1.npy", {"res": res, "s0": s0}, allow_pickle=True)
for useT in [False, True]:
    ks = [k for k in res if k[0] == useT]
    S = np.vstack([s0] + [res[k] for k in ks])
    nest, idx = B.nested(S)
    print(f"T={int(useT)} imbriqué {nest.mean():.4f} ± {nest.std():.4f} d={np.round(nest-s0,4)} choix", [("ref" if i == 0 else ks[i-1]) for i in idx], flush=True)
