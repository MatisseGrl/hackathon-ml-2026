import sys, time; sys.path.append("../.."); sys.path.append("."); sys.path.append("lab2/a4")
import numpy as np, pandas as pd
import blendn as B
from sklearn.metrics import f1_score
from trans import level_keys, smooth, pseudo_iter
t0 = time.time()
names = ["person", "f2", "neighbor", "person7", "resid2"]
folds = B.prepare(names, "pct", "user_sport", oracle=True)
s0, _ = B.score(folds, names, np.full(5, 0.2))
print(f"ref {s0.mean():.4f} ± {s0.std():.4f} {np.round(s0,4)}", flush=True)
tr, te = B.train, B.test; ntr = len(tr)
al = pd.concat([tr, te], ignore_index=True)
keys = level_keys(al); yi = B.yi
base = []
for fd in folds:
    va = fd["va"]; nv = len(va)
    Qv = sum(fd["Q"][n] for n in names) / 5
    lab = np.setdiff1d(np.arange(ntr), va)
    base.append(dict(va=va, Qv=Qv, lab=lab, fixed=Qv.max(1) > 0.999, pl=fd["pl"]))
WS = {"M": {"month": 1}, "S": {"slot": 1}, "R": {"route": 1}, "MSR": {"month": 1, "slot": 2, "route": 2}}
res = {}
# 2) mixte : lissage vers les labels connus des voisines étiquetées (plis != k)
for wn, wts in WS.items():
    for bt in [0.05, 0.1, 0.2, 0.3]:
        s = []
        for b in base:
            Q = smooth(b["Qv"], keys, b["va"], b["va"], b["Qv"], b["lab"], yi[b["lab"]], wts, 0.0, bt, b["fixed"])
            s.append(f1_score(yi[b["va"]], B.predict(Q, b["pl"]), average="macro"))
        s = np.array(s); res[("beta", wn, bt, 0)] = s
        print(f"beta {wn:4s} b={bt:.2f} : {s.mean():.4f} ± {s.std():.4f} d={np.round(s-s0,4)} win={(s>s0).sum()} ({time.time()-t0:.0f}s)", flush=True)
# 3) pseudo-labels (étiquetées plis != k + pli k pseudo-étiqueté après quotas), 1 à 3 itérations
for wn, wts in WS.items():
    for g in [0.1, 0.2, 0.3]:
        S = np.zeros((4, 5))
        for f, b in enumerate(base):
            hist = pseudo_iter(b["Qv"], keys, b["va"], b["lab"], yi[b["lab"]], wts, g,
                               lambda Q: B.predict(Q, b["pl"]), n_iter=3, fixed=b["fixed"])
            S[:, f] = [f1_score(yi[b["va"]], p, average="macro") for p in hist]
        for it in range(1, 4):
            s = S[it]; res[("pseudo", wn, g, it)] = s
            print(f"pseudo {wn:4s} g={g:.1f} it={it} : {s.mean():.4f} ± {s.std():.4f} d={np.round(s-s0,4)} win={(s>s0).sum()} ({time.time()-t0:.0f}s)", flush=True)
np.save("lab2/a4/res2.npy", {"res": res, "s0": s0}, allow_pickle=True)
for fam in ["beta", "pseudo"]:
    ks = [k for k in res if k[0] == fam]
    M = np.vstack([s0] + [res[k] for k in ks])
    nest, idx = B.nested(M)
    print(f"{fam} imbriqué {nest.mean():.4f} ± {nest.std():.4f} d={np.round(nest-s0,4)} win={(nest>s0).sum()} choix", [("ref" if i == 0 else ks[i-1]) for i in idx], flush=True)
