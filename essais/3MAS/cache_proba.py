"""Met en cache les probabilités d'un modèle (3MAS), au format de essais/RMYU/cache.py :
pour chaque pli de src/validation.py, P(val + test) du modèle entraîné sur le reste du train,
puis P(test) du modèle entraîné sur tout le train. Permet de tester les post-traitements sans réentraîner.
Usage : python cache_proba.py <nom> <person|final2|plus> '<kwargs json>'"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path += [str(ROOT), str(ROOT / "essais/RMYU"), str(Path(__file__).resolve().parent)]
import numpy as np, pandas as pd
from src.validation import load_data, get_folds

name, which = sys.argv[1], sys.argv[2]
kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
train, test, _ = load_data()
y = train.intensity.reset_index(drop=True)
if which == "final2":                     # Final2 utilise les variables de RMYU
    import os; os.chdir(ROOT / "essais/RMYU")
    from feats import build, add_resid
    from final2 import Final2 as M
    F = add_resid(build(train, test))
    X = F.drop(columns=["part", "workout_id"]).reset_index(drop=True)
    ntr = (F.part == "tr").sum()
    Xtr, Xte = X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True)
    os.chdir(Path(__file__).resolve().parent)
else:
    if which == "plus":
        from person_plus import PersonStackPlus as M
    else:
        from src.person_stack import PersonStack as M
    Xtr, Xte = train.drop(columns=["workout_id", "intensity"]), test.drop(columns=["workout_id"])
out, t0 = {}, time.time()
for f, (tr, va) in enumerate(get_folds(y).split(Xtr, y)):
    m = M(**kw).fit(Xtr.iloc[tr], y.iloc[tr])
    P = m.predict_proba(pd.concat([Xtr.iloc[va], Xte]).reset_index(drop=True))
    out[f"P{f}"] = P[:, [list(m.classes_).index(c) for c in ["easy", "moderate", "hard"]]]
    out[f"va{f}"] = va
    print(f"{name} pli {f} ({time.time() - t0:.0f}s)", flush=True)
m = M(**kw).fit(Xtr, y)
out["Pfull"] = m.predict_proba(Xte)[:, [list(m.classes_).index(c) for c in ["easy", "moderate", "hard"]]]
np.savez(Path(__file__).resolve().parent / f"cache/{name}.npz", **out)
print("fini", name, f"({time.time() - t0:.0f}s)", flush=True)
