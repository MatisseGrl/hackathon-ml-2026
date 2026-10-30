"""Calcule et met en cache les probas d'un modèle : pour chaque pli (folds de src/validation.py) P(val+test),
et P(test) du modèle entraîné sur tout le train. Usage : python cache.py <nom> <module:Classe> '<kwargs>'"""
import sys, json, time, importlib; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from src.validation import load_data, get_folds
from feats import build, add_resid
name, spec = sys.argv[1], sys.argv[2]; kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
mod, cls = spec.split(":"); M = getattr(importlib.import_module(mod), cls)
train, test, _ = load_data()
F = add_resid(build(train, test))
X = F.drop(columns=["part", "workout_id"]).reset_index(drop=True)
ntr = (F.part == "tr").sum(); Xtr, Xte = X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True)
y = train.intensity.reset_index(drop=True)
out = {}; t0 = time.time()
for f, (tr_i, va_i) in enumerate(get_folds(y).split(Xtr, y)):
    m = M(**kw).fit(Xtr.iloc[tr_i], y.iloc[tr_i])
    out[f"P{f}"] = m.predict_proba(pd.concat([Xtr.iloc[va_i], Xte]).reset_index(drop=True))
    out[f"va{f}"] = va_i
    print(f"fold {f} done ({time.time()-t0:.0f}s)", flush=True)
out["Pfull"] = M(**kw).fit(Xtr, y).predict_proba(Xte)
np.savez(f"cache_{name}.npz", **out); print("saved", name, f"({time.time()-t0:.0f}s)", flush=True)
