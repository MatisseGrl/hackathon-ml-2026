import sys, os, time, itertools
os.chdir(r"C:\Users\curie\hk\essais\RMYU")
sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
import blendn as B
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
t0 = time.time()
names = ["person", "f2", "neighbor", "person7", "resid2"]
folds = B.prepare(names, "pct", "user_sport", oracle=True)
T = pd.read_pickle("lab2/a5/T.pkl"); y = B.yi
ref = np.load("lab2/a5/ref_folds.npy")
rare = ~T.sport.isin(["cycling", "running"]).values
nonb = (T.nb_ymwh == 0).values
SEG = {"rare_sport": rare, "no_nb_ymwh": nonb}
def evalw(segname, w):
    out = []
    for f, fd in enumerate(folds):
        va = fd["va"]; m = SEG[segname][va][:, None]
        Qe = sum(fd["Q"][n] for n in names) / 5
        Qs = sum(wi * fd["Q"][n] for wi, n in zip(w, names))
        P = np.where(m, Qs, Qe)
        out.append(f1_score(y[va], B.predict(P, fd["pl"]), average="macro"))
    return np.array(out)
W = [np.array(c) / 4 for c in itertools.product(range(5), repeat=5) if sum(c) == 4]
W = [np.full(5, .2)] + W
for seg in SEG:
    S = np.array(Parallel(n_jobs=2)(delayed(evalw)(seg, w) for w in W))
    nest, idx = B.nested(S)
    i = np.argmax(S.mean(1))
    print(seg, "n", SEG[seg].sum(), "best", W[i], round(S[i].mean(), 4), "| nested", round(nest.mean(), 4), "+-", round(nest.std(), 4),
          "delta", np.round(nest - ref, 4), "won", int((nest > ref).sum()), [list(W[j]) for j in idx], flush=True)
    # single cache in segment
    for j, n in enumerate(names):
        w = np.zeros(5); w[j] = 1
        print("   only", n, np.round(evalw(seg, w).mean(), 4), flush=True)
# temperature / rank decoding (global)
def evalT(Tp, mode):
    out = []
    for f, fd in enumerate(folds):
        va = fd["va"]
        if mode == "arith":
            P = sum(fd["Q"][n] for n in names) / 5
        else:
            L = sum(np.log(np.clip(fd["Q"][n], 1e-9, 1)) for n in names) / 5; P = np.exp(L); P /= P.sum(1, keepdims=True)
        P = P ** Tp; P /= P.sum(1, keepdims=True)
        out.append(f1_score(y[va], B.predict(P, fd["pl"]), average="macro"))
    return np.array(out)
C = [(t, m) for m in ["arith", "geo"] for t in [0.5, 0.75, 1.0, 1.5, 2.0]]
S = np.array(Parallel(n_jobs=2)(delayed(evalT)(*c) for c in C))
for c, sc in zip(C, S):
    print("temp", c, round(sc.mean(), 4), np.round(sc - ref, 4), int((sc > ref).sum()), flush=True)
nest, idx = B.nested(S)
print("temp nested", round(nest.mean(), 4), np.round(nest - ref, 4), [C[i] for i in idx])
print("time", round(time.time() - t0))
