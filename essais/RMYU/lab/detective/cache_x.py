"""Comme cache.py mais avec add_my_feats (groupes choisis). Usage (depuis RMYU) :
python -u lab/detective/cache_x.py <nom> <module:Classe> '<kwargs json>' <groupes,séparés|all>
-> lab/detective/cache_<nom>.npz (P{f}, va{f}, Pfull)"""
import sys, json, time, importlib, os; sys.path.append("../.."); sys.path.append("."); sys.path.append("lab/detective")
import numpy as np, pandas as pd
from src.validation import load_data, get_folds
from feats import build, add_resid
from myfeats import add_my_feats
name, spec = sys.argv[1], sys.argv[2]; kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
groups = sys.argv[4] if len(sys.argv) > 4 else "all"
mod, cls = spec.split(":"); M = getattr(importlib.import_module(mod), cls)
train, test, _ = load_data()
cf = "lab/detective/F_base.pkl"
F = pd.read_pickle(cf) if os.path.exists(cf) else add_resid(build(train, test))
F = add_my_feats(F) if groups == "all" else add_my_feats(F, which=tuple(groups.split(",")))
X = F.drop(columns=["part", "workout_id"]).reset_index(drop=True)
ntr = (F.part == "tr").sum(); Xtr, Xte = X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True)
y = train.intensity.reset_index(drop=True)
out = {}; t0 = time.time()
for f, (tr_i, va_i) in enumerate(get_folds(y).split(Xtr, y)):
    m = M(**kw).fit(Xtr.iloc[tr_i], y.iloc[tr_i])
    out[f"P{f}"] = m.predict_proba(pd.concat([Xtr.iloc[va_i], Xte]).reset_index(drop=True))
    out[f"va{f}"] = va_i
    print(f"fold {f} done ({time.time()-t0:.0f}s)", flush=True)
    np.savez(f"lab/detective/cache_{name}_partial.npz", **out)
out["Pfull"] = M(**kw).fit(Xtr, y).predict_proba(Xte)
np.savez(f"lab/detective/cache_{name}.npz", **out); print("saved", name, f"({time.time()-t0:.0f}s)", flush=True)
