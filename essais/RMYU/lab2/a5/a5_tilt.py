import sys, os, time
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
import blendn as B
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
t0 = time.time()
names = ["person", "f2", "neighbor", "person7", "resid2"]
folds = B.prepare(names, "pct", "user_sport", oracle=True)
tr = B.train.reset_index(drop=True); y = B.yi
ref = np.load("lab2/a5/ref_folds.npy")
def nbstats(cols):
    k = tr[cols].astype(str).agg("_".join, axis=1).values
    S = []; 
    for f, fd in enumerate(folds):
        va = fd["va"]; lab = np.setdiff1d(np.arange(len(tr)), va)
        g = pd.DataFrame(dict(k=k[lab], y=y[lab] - 1.0)).groupby("k").y.agg(["sum", "size"])
        kv = pd.Series(k[va])
        S.append((kv.map(g["sum"]).fillna(0).values, kv.map(g["size"]).fillna(0).values))
    return S
NB = {"ymwh": nbstats(["user_id", "year", "month", "weekday", "start_hour_utc"]),
      "ym": nbstats(["user_id", "year", "month"]),
      "ymw": nbstats(["user_id", "year", "month", "weekday"])}
Q0 = [sum(fd["Q"][n] for n in names) / 5 for fd in folds]
# residual check: corr within fold between shrunk nb mean and (y - E)
for key in NB:
    r = []
    for f, fd in enumerate(folds):
        s_, n_ = NB[key][f]; m = s_ / (n_ + 1)
        E = Q0[f] @ np.array([0, 1, 2.]); res = y[fd["va"]] - E
        r.append(np.corrcoef(m, res)[0, 1])
    print(key, "corr(nb_mean, y-E) per fold", np.round(r, 3))
def evalc(key, beta, lam):
    out = []
    for f, fd in enumerate(folds):
        s_, n_ = NB[key][f]; m = s_ / (n_ + lam)
        L = np.log(np.clip(Q0[f], 1e-12, 1)) + beta * m[:, None] * np.array([-1, 0, 1.])
        P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        out.append(f1_score(y[fd["va"]], B.predict(P, fd["pl"]), average="macro"))
    return np.array(out)
C = [(k, b, l) for k in NB for b in [0.0, 0.25, 0.5, 1.0, 1.5, 2.5] for l in [1.0, 4.0]]
S = np.array(Parallel(n_jobs=2)(delayed(evalc)(*c) for c in C))
for c, sc in zip(C, S):
    print(c, round(sc.mean(), 4), np.round(sc - ref, 4), int((sc > ref).sum()))
nest, idx = B.nested(S)
print("nested", round(nest.mean(), 4), "+-", round(nest.std(), 4), "delta", np.round(nest - ref, 4), "won", int((nest > ref).sum()), [C[i] for i in idx])
np.savez("lab2/a5/tilt_grid.npz", S=S, C=np.array([str(c) for c in C]))
print("time", round(time.time() - t0))
