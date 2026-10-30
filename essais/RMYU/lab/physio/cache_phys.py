"""Comme cache.py mais avec X = add_physio(add_resid(build(train, test))) (colonnes ph_* en plus).
Usage (depuis essais/RMYU) :
  python -u lab/physio/cache_phys.py <nom> <module:Classe> '<kwargs json>'  -> lab/physio/cache_<nom>.npz
Le cache a le même format que cache.py (P{f}, va{f}, Pfull) ; pour post.py, copier/lier dans RMYU/cache_<nom>.npz."""
import sys, json, time, importlib, os; sys.path.append("../.."); sys.path.append("."); sys.path.append("lab/physio")
import numpy as np, pandas as pd
from src.validation import load_data, get_folds
from feats import build, add_resid
from physio import add_physio
name, spec = sys.argv[1], sys.argv[2]; kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
drop = json.loads(sys.argv[4]) if len(sys.argv) > 4 else []
mod, cls = spec.split(":"); M = getattr(importlib.import_module(mod), cls)
train, test, _ = load_data()
pk = "lab/physio/F_physio.pkl"
if os.path.exists(pk):
    F = pd.read_pickle(pk)
else:
    F = add_physio(add_resid(build(train, test))); F.to_pickle(pk)
F = F.drop(columns=[c for c in drop if c in F.columns])
X = F.drop(columns=["part", "workout_id"]).reset_index(drop=True)
print("X", X.shape, flush=True)
ntr = (F.part == "tr").sum(); Xtr, Xte = X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True)
y = train.intensity.reset_index(drop=True)
out = {}; t0 = time.time()
for f, (tr_i, va_i) in enumerate(get_folds(y).split(Xtr, y)):
    m = M(**kw).fit(Xtr.iloc[tr_i], y.iloc[tr_i])
    out[f"P{f}"] = m.predict_proba(pd.concat([Xtr.iloc[va_i], Xte]).reset_index(drop=True))
    out[f"va{f}"] = va_i
    print(f"fold {f} done ({time.time()-t0:.0f}s)", flush=True)
    np.savez(f"lab/physio/cache_{name}_partial.npz", **out)
out["Pfull"] = M(**kw).fit(Xtr, y).predict_proba(Xte)
np.savez(f"lab/physio/cache_{name}.npz", **out); print("saved", name, f"({time.time()-t0:.0f}s)", flush=True)
