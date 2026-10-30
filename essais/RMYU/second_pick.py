"""Second choix de soumission finale : Kaggle garde le MEILLEUR des deux scores privés si l'équipe coche deux soumissions.
Pour chaque candidat X, gain attendu E[max(F1_5, F1_X)] - E[F1_5] sur des « moitiés privées » simulées (3 835 lignes tirées
au hasard dans 2 plis de validation réunis, chaque ligne prédite par le modèle de son pli, quotas avec les comptes réels).
Usage (depuis essais/RMYU) : python -u second_pick.py"""
import sys, itertools; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
import blendn as B
from post import adjust

C = {n: np.load(f"cache_{n}.npz") for n in ["person", "person7", "person11", "f2", "f2s7", "neighbor", "neighbor7", "resid2"]}
va = [C["person"][f"va{f}"] for f in range(5)]
cands = {"n°5": ({"person": .2, "f2": .2, "neighbor": .2, "person7": .2, "resid2": .2}, "user_sport"),
         "n°3": ({"person": .6, "f2": .4}, "user"),
         "n°4 (≈ person seul)": ({"person": 1.0}, "user_sport"),
         "familles 8": ({"person": .4 / 3, "person7": .4 / 3, "person11": .4 / 3, "f2": .1, "f2s7": .1, "neighbor": .1, "neighbor7": .1, "resid2": .2}, "user_sport"),
         "égal 8": ({n: 1 / 8 for n in C}, "user_sport")}


def f1m(y, p):
    cm = np.bincount(3 * y + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm); return np.mean(2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1))


rng = np.random.default_rng(0)
D = {k: [] for k in cands}
dis = {k: [] for k in cands}
for a, b in itertools.combinations(range(5), 2):
    v = np.concatenate([va[a], va[b]]); lab = np.setdiff1d(np.arange(len(B.train)), v); yv = B.yi[v]
    pred = {}
    for k, (w, by) in cands.items():
        P = sum(wi * np.concatenate([C[n][f"P{a}"][:len(va[a])], C[n][f"P{b}"][:len(va[b])]]) for n, wi in w.items())
        Q = adjust(P, B.kf_tr[v], B.kp_tr[v], B.kf_tr[lab], B.kp_tr[lab], B.yi[lab], w4=0, wfull=1.0)
        g = pd.Series(B.group_keys(by)[0][v])
        pl = [(np.asarray(r), np.bincount(yv[r], minlength=3)) for r in g.groupby(g).indices.values()]
        pred[k] = B.predict(Q, pl)
    for k in cands:
        dis[k].append((pred[k] != pred["n°5"]).mean())
    for _ in range(300):
        s = rng.choice(len(v), 3835, replace=False)
        f5 = f1m(yv[s], pred["n°5"][s])
        for k in cands:
            D[k].append((f1m(yv[s], pred[k][s]), f5))
print(f"{'candidat':22s} désaccord  Δ moyen   écart-type  P(X > n°5)  gain attendu du max")
for k in cands:
    x = np.array(D[k]); d = x[:, 0] - x[:, 1]
    print(f"{k:22s} {np.mean(dis[k]):8.3f}  {d.mean():+.4f}   {d.std():.4f}      {(d > 0).mean():.2f}       {np.maximum(d, 0).mean():+.4f}", flush=True)
