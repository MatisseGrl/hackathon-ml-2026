"""Comme cache.py, mais le modèle reçoit les colonnes brutes du train (sans workout_id ni intensity),
pour les modèles de l'équipe (src/person_stack.py, src/stacking.py). Usage : python cache_raw.py <nom> <module:Classe> '<kwargs>'"""
import sys, json, time, importlib; sys.path.insert(0, "../.."); sys.path.append(".")
import numpy as np, pandas as pd
from src.validation import load_data, get_folds
name, spec = sys.argv[1], sys.argv[2]; kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
mod, cls = spec.split(":"); M = getattr(importlib.import_module(mod), cls)
train, test, _ = load_data()
Xtr = train.drop(columns=["workout_id", "intensity"]).reset_index(drop=True)
Xte = test.drop(columns=["workout_id"]).reset_index(drop=True)
y = train.intensity.reset_index(drop=True)
out = {}; t0 = time.time()
for f, (tr_i, va_i) in enumerate(get_folds(y).split(Xtr, y)):
    m = M(**kw).fit(Xtr.iloc[tr_i], y.iloc[tr_i])
    P = m.predict_proba(pd.concat([Xtr.iloc[va_i], Xte]).reset_index(drop=True))
    order = [list(m.classes_).index(c) for c in ["easy", "moderate", "hard"]]
    out[f"P{f}"] = P[:, order]; out[f"va{f}"] = va_i
    print(f"fold {f} done ({time.time()-t0:.0f}s)", flush=True)
m = M(**kw).fit(Xtr, y)
order = [list(m.classes_).index(c) for c in ["easy", "moderate", "hard"]]
out["Pfull"] = m.predict_proba(Xte)[:, order]
np.savez(f"cache_{name}.npz", **out); print("saved", name, f"({time.time()-t0:.0f}s)", flush=True)
