import sys, os, time
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
import blendn as B
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
names = ["person", "f2", "neighbor", "person7", "resid2"]
folds = B.prepare(names, "pct", "user_sport", oracle=True)
y = B.yi; ref = np.load("lab2/a5/ref_folds.npy")
T = pd.read_pickle("lab2/a5/T.pkl"); hasnb = (T.nb_ymwh > 0).values
def evalc(kind, a):
    out = []
    for fd in folds:
        va = fd["va"]
        P = sum(fd["Q"][n] for n in names) / 5
        E = np.stack([fd["Q"][n] @ np.array([0, 1, 2.]) for n in names], 1); sd = E.std(1)
        if kind == "std":
            tau = np.clip((np.maximum(sd, 1e-3) / np.median(sd)) ** (-a), 0.25, 4)
        else:
            tau = 1 + a * hasnb[va]
        L = np.log(np.clip(P, 1e-9, 1)) * tau[:, None]
        Pn = np.exp(L - L.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True)
        out.append(f1_score(y[va], B.predict(Pn, fd["pl"]), average="macro"))
    return np.array(out)
C = [("std", 0.0)] + [("std", a) for a in [-0.5, -0.25, 0.25, 0.5, 1.0]] + [("nb", g) for g in [-0.3, 0.3, 0.6, 1.0]]
S = np.array(Parallel(n_jobs=2)(delayed(evalc)(*c) for c in C))
for c, sc in zip(C, S):
    print(c, round(sc.mean(), 4), "+-", round(sc.std(), 4), np.round(sc - ref, 4), "won", int((sc > ref).sum()), flush=True)
nest, idx = B.nested(S)
print("nested", round(nest.mean(), 4), "+-", round(nest.std(), 4), np.round(nest - ref, 4), "won", int((nest > ref).sum()), [C[i] for i in idx])
np.savez("lab2/a5/rowT_grid.npz", S=S)
