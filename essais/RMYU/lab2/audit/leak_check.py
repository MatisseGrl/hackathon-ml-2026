import sys; sys.path.append("../../../.."); sys.path.append("../..")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from src.validation import load_data, get_folds
train, test, _ = load_data()
y = train.intensity.map({"easy":0,"moderate":1,"hard":2}).values
nte = len(test)
folds = list(get_folds(y).split(np.zeros(len(y)), y))
for n in ["person","person7","f2","neighbor","resid2"]:
    z = np.load(f"../../cache_{n}.npz")
    Pf = z["Pfull"]
    rows = []
    for f in range(5):
        va = z[f"va{f}"]; P = z[f"P{f}"]
        assert (np.sort(va) == np.sort(folds[f][1])).all(), "va mismatch"
        Pv, Pt = P[:len(va)], P[len(va):]
        assert len(Pt) == nte
        f1 = f1_score(y[va], Pv.argmax(1), average="macro")
        ll = -np.log(np.clip(Pv[np.arange(len(va)), y[va]], 1e-9, 1)).mean()
        rows.append((f, f1, ll, Pv.max(1).mean(), Pt.max(1).mean(),
                     np.abs(Pt - Pf).sum(1).mean()/2))
    r = np.array(rows)
    print(f"{n:9s} F1/pli=" + " ".join(f"{v:.4f}" for v in r[:,1]) + f" | moy {r[:,1].mean():.4f}"
          f" | logloss {r[:,2].mean():.4f} | conf val {r[:,3].mean():.4f} test(pli) {r[:,4].mean():.4f} Pfull {Pf.max(1).mean():.4f}"
          f" | TV(Pt_pli,Pfull) {r[:,5].mean():.3f}")
