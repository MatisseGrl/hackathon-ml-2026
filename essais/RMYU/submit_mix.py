"""Soumission à partir de caches et de poids : moyenne pondérée des Pfull, recopie des doublons exacts, quotas exacts
par user + sport (règle easy = N//3, moderate = 2N//3 - N//3, hard = reste), puis contrôle par check_sub.py.
Usage (depuis essais/RMYU) : python submit_mix.py ../../submissions/<fichier>.csv person:0.2 f2:0.2 neighbor:0.2 person7:0.2 resid2:0.2"""
import sys, subprocess; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
import blendn as B
from post import adjust
from quota import LAB

out = sys.argv[1]
spec = [a.split(":") for a in sys.argv[2:]]
names, w = [n for n, _ in spec], np.array([float(x) for _, x in spec])
assert abs(w.sum() - 1) < 1e-9, w.sum()
P = sum(wi * np.load(f"cache_{n}.npz")["Pfull"] for wi, n in zip(w, names))
Q = adjust(P, B.kf_te, B.kp_te, B.kf_tr, B.kp_tr, B.yi, w4=0, wfull=1.0)
gtr, gte = B.group_keys("user_sport")
pred = B.predict(Q, B.plan(gte, gtr, B.yi, "pct"))
pd.DataFrame({"workout_id": B.test.workout_id.values, "intensity": LAB[pred]}).to_csv(out, index=False)
s, _ = B.score(B.prepare(names, "pct", "user_sport", oracle=True), names, w)
print(f"écrit {out} ; validation (protocole fidèle) {s.mean():.4f} ± {s.std():.4f} plis {np.round(s, 4)}", flush=True)
subprocess.run([sys.executable, "check_sub.py", out])
