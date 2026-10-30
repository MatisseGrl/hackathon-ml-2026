"""Soumission : moyenne ÉGALE des 5 modèles en cache (PersonStack graines 42 et 7, Final2, NeighborStack, Resid2),
puis recopie des doublons exacts, puis quotas exacts par user + sport (règle : easy = N//3, moderate = 2N//3 - N//3,
hard = le reste ; seule règle cohérente avec tous les comptes du train).
Usage (depuis essais/RMYU) : python submit_blend5.py ../../submissions/<fichier>.csv"""
import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
import blendn as B
from post import adjust
from quota import LAB

names = ["person", "f2", "neighbor", "person7", "resid2"]
w = np.full(len(names), 1 / len(names))
P = sum(wi * np.load(f"cache_{n}.npz")["Pfull"] for wi, n in zip(w, names))
Q = adjust(P, B.kf_te, B.kp_te, B.kf_tr, B.kp_tr, B.yi, w4=0, wfull=1.0)
gtr, gte = B.group_keys("user_sport")
# contrôle : les comptes restants de la règle tombent juste dans chaque groupe (pas de correction nécessaire)
known = pd.crosstab(pd.Series(gtr), pd.Series(B.yi)).reindex(columns=[0, 1, 2], fill_value=0)
ntr, nte = pd.Series(gtr).value_counts(), pd.Series(gte).value_counts()
bad = [g for g in nte.index if ((B.targets(ntr.get(g, 0) + nte[g], "pct") - (known.loc[g].values if g in known.index else 0)) < 0).any()
       or (B.targets(ntr.get(g, 0) + nte[g], "pct") - (known.loc[g].values if g in known.index else 0)).sum() != nte[g]]
print("groupes du test :", len(nte), "; groupes où la règle ne tombe pas juste :", len(bad))
pred = B.predict(Q, B.plan(gte, gtr, B.yi, "pct"))
sub = pd.DataFrame({"workout_id": B.test.workout_id.values, "intensity": LAB[pred]})
ref = pd.read_csv("../../data/sample_submission.csv")
assert len(sub) == 7669 and list(sub.columns) == ["workout_id", "intensity"]
assert (sub.workout_id.values == ref.workout_id.values).all()
out = sys.argv[1]
sub.to_csv(out, index=False)
print("écrit", out, sub.intensity.value_counts().to_dict())
for f in ["sub_RMYU_blend_person06_final2_dup_quota.csv", "sub_RMYU_blend_person06_final2_dup_quota_numpy.csv"]:
    o = pd.read_csv(f"../../submissions/{f}")
    print(f"accord avec {f} : {(o.intensity.values == sub.intensity.values).mean():.4f}")
