"""Test de la règle d'arrondi par vraisemblance : pour chaque user avec N%3 != 0, log P(comptes restants = q | probas)
pour chaque répartition possible de l'unité en trop. Usage : python lab/decode/roundlik.py <cache.npz> [Pkey]"""
import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from src.validation import load_data

train, test, _ = load_data()
yi = train.intensity.map({"easy": 0, "moderate": 1, "hard": 2}).values
C = np.load(sys.argv[1])
PAT = {1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)], 2: [(1, 1, 0), (1, 0, 1), (0, 1, 1)]}


def logZ(P, qs):
    m = len(P); A = max(q[0] for q in qs) + 1; B = max(q[1] for q in qs) + 1
    f = np.zeros((A, B)); f[0, 0] = 1; ls = 0.0
    for i in range(m):
        g = f * P[i, 2]; g[1:] += f[:-1] * P[i, 0]; g[:, 1:] += f[:, :-1] * P[i, 1]
        s = g.sum(); f = g / s; ls += np.log(s)
    return [np.log(max(f[q[0], q[1]], 1e-300)) + ls for q in qs]


def analyse(P, gu, gl, yl, tag):
    P = np.clip(P, 1e-6, 1); P = P / P.sum(1, keepdims=True)
    known = pd.crosstab(pd.Series(gl), pd.Series(yl)).reindex(columns=[0, 1, 2], fill_value=0)
    nl = pd.Series(gl).value_counts(); gus = pd.Series(gu)
    tot = {1: np.zeros(3), 2: np.zeros(3)}; win = {1: np.zeros(3), 2: np.zeros(3)}; ninf = {1: np.zeros(3), 2: np.zeros(3)}
    for g, rows in gus.groupby(gus).groups.items():
        rows = np.asarray(rows); N = len(rows) + nl.get(g, 0); k, r = divmod(N, 3)
        if r == 0:
            continue
        kc = known.loc[g].values if g in known.index else np.zeros(3, int)
        qs = [k + np.array(e) - kc for e in PAT[r]]
        ok = [(q >= 0).all() for q in qs]
        ll = np.full(3, -np.inf)
        val = logZ(P[rows], [q for q, o in zip(qs, ok) if o])
        ll[np.array(ok)] = val
        ninf[r] += ~np.array(ok)
        llf = np.where(np.isfinite(ll), ll, np.nan)
        lse = np.log(np.nansum(np.exp(llf - np.nanmax(llf)))) + np.nanmax(llf)
        post = np.where(np.isfinite(ll), np.exp(ll - lse), 0)
        tot[r] += np.log(np.clip(post, 1e-12, 1)) * 0 + post   # somme des postérieurs (prior uniforme)
        win[r][np.argmax(ll)] += 1
    for r in (1, 2):
        print(f"{tag} N%3={r} patterns {PAT[r]} : somme posterieurs {np.round(tot[r],1)} | argmax {win[r]} | infaisables(train) {ninf[r]}")


key = sys.argv[2] if len(sys.argv) > 2 else "Pfull"
if key == "Pfull":
    analyse(C["Pfull"], test.user_id.values, train.user_id.values, yi, "kaggle")
else:
    for f in range(5):
        va = C[f"va{f}"]; lab = np.setdiff1d(np.arange(len(train)), va)
        # pour tester sans tricher on utilise les vrais labels des lignes de validation ? NON : seulement les probas
        gu = np.concatenate([train.user_id.values[va], test.user_id.values])
        analyse(C[f"P{f}"], gu, train.user_id.values[lab], yi[lab], f"fold{f}")
