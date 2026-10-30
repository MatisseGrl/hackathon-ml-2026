"""Graines supplémentaires (règle fixée AVANT de voir les résultats) : moyenne égale de tous les modèles disponibles,
adoptée seulement si elle ne perd pas contre la soumission n° 5 (moyenne égale des 5) : moyenne >= et au moins 3 plis sur 5.
Variante secondaire : poids de familles de la n° 5 (PersonStack 0,4 ; Final2 0,2 ; NeighborStack 0,2 ; Resid2 0,2), graines moyennées dans chaque famille.
Trois protocoles : comptes réels d'un pli (≈15 % de chaque groupe), 2 plis réunis (≈30 %, proche des 25 % du test), quotas validation + test.
Usage (depuis essais/RMYU) : python -u seeds_eval.py person11 neighbor7 [f2s7]"""
import sys, itertools; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
import blendn as B
from post import adjust

new = sys.argv[1:]
fam = {"person": ["person", "person7"], "f2": ["f2"], "neighbor": ["neighbor"], "resid2": ["resid2"]}
for n in new:
    fam[[k for k in fam if n.startswith(k)][0]].append(n)
base5 = ["person", "f2", "neighbor", "person7", "resid2"]
allr = sorted(set(sum(fam.values(), [])), key=lambda x: (x not in base5, x))
fw = {"person": 0.4, "f2": 0.2, "neighbor": 0.2, "resid2": 0.2}
cands = {"n°5 (égal 5)": {n: 0.2 for n in base5},
         f"égal {len(allr)} (règle fixée)": {n: 1 / len(allr) for n in allr},
         "familles n°5, graines moyennées": {n: fw[k] / len(v) for k, v in fam.items() for n in v}}


def pair_folds(names):
    """2 plis réunis : lignes de validation des plis a et b (chacune prédite par son propre modèle), comptes réels par user+sport."""
    C = {n: np.load(f"cache_{n}.npz") for n in names}
    va = [C[names[0]][f"va{f}"] for f in range(5)]
    gtr, _ = B.group_keys("user_sport")
    out = []
    for a, b in itertools.combinations(range(5), 2):
        v = np.concatenate([va[a], va[b]]); lab = np.setdiff1d(np.arange(len(B.train)), v)
        Q = {n: adjust(np.concatenate([C[n][f"P{a}"][:len(va[a])], C[n][f"P{b}"][:len(va[b])]]),
                       B.kf_tr[v], B.kp_tr[v], B.kf_tr[lab], B.kp_tr[lab], B.yi[lab], w4=0, wfull=1.0) for n in names}
        g = pd.Series(gtr[v])
        pl = [(np.asarray(r), np.bincount(B.yi[v][r], minlength=3)) for r in g.groupby(g).indices.values()]
        out.append(dict(va=v, Q=Q, pl=pl))
    return out


def score_pairs(folds, w):
    return np.array([f1_score(B.yi[fd["va"]], B.predict(sum(wi * fd["Q"][n] for n, wi in w.items()), fd["pl"]), average="macro")
                     for fd in folds])


F1 = B.prepare(allr, "pct", "user_sport", oracle=True)
FU = B.prepare(allr, "pct", "user_sport", oracle=False)
FP = pair_folds(allr)
res = {}
for k, w in cands.items():
    names, ww = list(w), np.array(list(w.values()))
    s1, _ = B.score(F1, names, ww); su, _ = B.score(FU, names, ww); sp = score_pairs(FP, w)
    res[k] = (s1, su, sp)
r0 = res["n°5 (égal 5)"]
for k, (s1, su, sp) in res.items():
    d1, du, dp = s1 - r0[0], su - r0[1], sp - r0[2]
    print(f"{k:34s} | 1 pli {s1.mean():.4f} Δ {d1.mean():+.4f} ({(d1 > 0).sum()}/5) | 2 plis {sp.mean():.4f} Δ {dp.mean():+.4f} ({(dp > 0).sum()}/10)"
          f" | val+test {su.mean():.4f} Δ {du.mean():+.4f} ({(du > 0).sum()}/5)", flush=True)
print("Δ par pli (1 pli) :", {k: np.round(v[0] - r0[0], 4).tolist() for k, v in res.items() if k != "n°5 (égal 5)"})
