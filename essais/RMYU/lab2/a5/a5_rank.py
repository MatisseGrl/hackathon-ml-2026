import sys, os, time
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
import blendn as B
from sklearn.metrics import f1_score
names = ["person", "f2", "neighbor", "person7", "resid2"]
folds = B.prepare(names, "pct", "user_sport", oracle=True)
y = B.yi; ref = np.load("lab2/a5/ref_folds.npy")
def rank_decode(s, pl):
    pred = np.zeros(len(s), int)
    for rows, rem in pl:
        o = rows[np.argsort(s[rows], kind="stable")]
        pred[o[:rem[0]]] = 0; pred[o[rem[0]:rem[0] + rem[1]]] = 1; pred[o[rem[0] + rem[1]:]] = 2
    return pred
res = {}
for mode in ["E", "logratio", "p2-p0"]:
    sc = []
    for fd in folds:
        va = fd["va"]; P = sum(fd["Q"][n] for n in names) / 5
        Pc = np.clip(P, 1e-9, 1)
        s = {"E": P @ np.array([0, 1, 2.]), "logratio": np.log(Pc[:, 2] / Pc[:, 0]), "p2-p0": P[:, 2] - P[:, 0]}[mode]
        sc.append(f1_score(y[va], rank_decode(s, fd["pl"]), average="macro"))
    sc = np.array(sc); res[mode] = sc
    print(mode, round(sc.mean(), 4), "+-", round(sc.std(), 4), np.round(sc - ref, 4), "won", int((sc > ref).sum()), flush=True)
# error by true within-group position (distance to the cut) for reference decode
T = pd.read_pickle("lab2/a5/T.pkl")
print("2-level errors share among errors", round(T.err2.sum() / T.err.sum(), 3))
