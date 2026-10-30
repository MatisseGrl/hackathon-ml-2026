import sys, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
import blendn as B
from sklearn.metrics import f1_score
names = ["person", "f2", "neighbor", "person7", "resid2"]
folds = B.prepare(names, "pct", "user_sport", oracle=True)
s, a = B.score(folds, names, np.full(5, 0.2))
print("ref", np.round(s, 4), round(s.mean(), 4), round(s.std(), 4))
tr, te = B.train, B.test
y = B.yi
rows = []
for f, fd in enumerate(folds):
    va = fd["va"]; nv = len(va)
    Q = sum(fd["Q"][n] for n in names) / 5
    pred = B.predict(Q, fd["pl"])[:nv]
    E = np.stack([fd["Q"][n] @ np.array([0, 1, 2.]) for n in names], 1)
    Er = np.stack([fd["R"][n][:nv] @ np.array([0, 1, 2.]) for n in names], 1)
    d = pd.DataFrame(dict(idx=va, fold=f, pred=pred, y=y[va], pmax=Q.max(1), Estd=E.std(1), Eraw_std=Er.std(1),
                          Emean=E.mean(1), p0=Q[:, 0], p1=Q[:, 1], p2=Q[:, 2]))
    for j, n in enumerate(names):
        d["E_" + n] = E[:, j]
    rows.append(d)
D = pd.concat(rows).sort_values("idx").reset_index(drop=True)
D.to_pickle("lab2/a5/oof_ref.pkl")
np.save("lab2/a5/ref_folds.npy", s)
print(D.shape, (D.pred != D.y).mean())
