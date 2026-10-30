"""Ajoute Pfull (HGB lr .05 300 it entraîné sur tout le train) à un cache evalf : python lab/detective/add_pfull.py <tag> <prefixes>"""
import sys; sys.path.append("../.."); sys.path.append("."); sys.path.append("lab/detective")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from src.validation import load_data
from myfeats import add_my_feats
tag, which = sys.argv[1], sys.argv[2]
train, test, _ = load_data()
F = pd.read_pickle("lab/detective/F_base.pkl"); base = set(F.columns); F = add_my_feats(F)
new = [c for c in F.columns if c not in base]; keep = which.split(",")
F = F.drop(columns=[c for c in new if not any(c.startswith(k) for k in keep)])
X = F.drop(columns=["part", "workout_id"]).reset_index(drop=True); ntr = (F.part == "tr").sum()
yi = train.intensity.map({"easy": 0, "moderate": 1, "hard": 2}).values
m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, random_state=42).fit(X.iloc[:ntr], yi)
C = dict(np.load(f"lab/detective/cache_det_{tag}.npz")); C["Pfull"] = m.predict_proba(X.iloc[ntr:])
np.savez(f"lab/detective/cache_det_{tag}.npz", **C); print("Pfull added", tag, C["Pfull"].shape)
