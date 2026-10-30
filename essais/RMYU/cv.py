"""CV commune (folds de src/validation.py) + quotas. Usage : python cv.py <nom> '<kwargs Final>'
Sauve les probas hors-fold (val) et test moyennées dans oof_<nom>.npz pour l'ensemble."""
import sys, json, time; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from src.validation import load_data, get_folds
from feats import build, add_resid
from quota import constrained_predict
from final2 import Final2 as Final
name = sys.argv[1]; kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
train, test, _ = load_data()
F = add_resid(build(train, test))
X = F.drop(columns=["part", "workout_id"]).reset_index(drop=True)
ntr = (F.part == "tr").sum(); Xtr, Xte = X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True)
y = train.intensity.reset_index(drop=True); yi = y.map({"easy": 0, "moderate": 1, "hard": 2}).values
key = lambda d: d.user_id.values
oof = np.zeros((ntr, 3)); Pte_cv = np.zeros((len(Xte), 3)); a, q = [], []
t0 = time.time()
for tr_i, va_i in get_folds(y).split(Xtr, y):
    m = Final(**kw).fit(Xtr.iloc[tr_i], y.iloc[tr_i])
    Xu = pd.concat([Xtr.iloc[va_i], Xte]).reset_index(drop=True); nv = len(va_i)
    P = m.predict_proba(Xu); oof[va_i] = P[:nv]; Pte_cv += P[nv:] / 5
    a.append(f1_score(yi[va_i], P[:nv].argmax(1), average="macro"))
    q.append(f1_score(yi[va_i], constrained_predict(P, key(Xu), key(Xtr.iloc[tr_i]), yi[tr_i])[:nv], average="macro"))
    print(f"fold {len(a)}: argmax {a[-1]:.4f} quota {q[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
print(f"{name} {kw} => argmax {np.mean(a):.4f} | quota {np.mean(q):.4f} ± {np.std(q):.4f}", flush=True)
np.savez(f"oof_{name}.npz", oof=oof, quota=np.array(q))
