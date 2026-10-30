"""Entraîne sur tout le train, prédit le test, applique les quotas par user. Usage : python submit.py <modele> <sortie.csv> '<kwargs>'"""
import sys, json; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from src.validation import load_data
from feats import build, add_resid
from quota import constrained_predict, LAB
which, out = sys.argv[1], sys.argv[2]; kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
train, test, sample = load_data()
F = add_resid(build(train, test))
X = F.drop(columns=["part", "workout_id"]).reset_index(drop=True)
ntr = (F.part == "tr").sum(); Xtr, Xte = X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True)
y = train.intensity.reset_index(drop=True); yi = y.map({"easy": 0, "moderate": 1, "hard": 2}).values
if which == "resid2":
    from resid2 import Resid2 as M
else:
    from final import Final as M
P = M(**kw).fit(Xtr, y).predict_proba(Xte)
np.save(out.replace(".csv", "_proba.npy"), P)
pred = constrained_predict(P, Xte.user_id.values, Xtr.user_id.values, yi)
sub = pd.DataFrame({"workout_id": test.workout_id.values, "intensity": LAB[pred]})
assert len(sub) == len(sample) and (sub.workout_id.values == sample.workout_id.values).all()
sub.to_csv(out, index=False)
print(out, sub.intensity.value_counts(normalize=True).round(3).to_dict(), "| argmax vs quota agree:", (P.argmax(1) == pred).mean().round(3))
