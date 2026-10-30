"""Affectation sous contrainte : dans chaque groupe (user[, sport]), les 3 classes font chacune 1/3 de (train+test).
Les comptes restants = N/3 - comptes connus du train ; on affecte les lignes non étiquetées pour maximiser sum log P."""
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

LAB = np.array(["easy", "moderate", "hard"])


def remaining_counts(n_total, known_counts, m):
    target = np.full(3, n_total / 3.0)
    rem = np.clip(target - known_counts, 0, None)
    base = np.floor(rem).astype(int)
    left = m - base.sum()
    frac = rem - base
    order = np.argsort(-frac) if left >= 0 else np.argsort(frac)
    i = 0
    while left != 0:
        c = order[i % 3]
        if left > 0:
            base[c] += 1; left -= 1
        elif base[c] > 0:
            base[c] -= 1; left += 1
        i += 1
    return base


def assign(P, counts):
    cols = np.concatenate([np.full(k, c) for c, k in enumerate(counts)])
    cost = -np.log(np.clip(P[:, cols], 1e-9, 1))
    r, c = linear_sum_assignment(cost)
    out = np.empty(len(P), dtype=int); out[r] = cols[c]
    return out


def constrained_predict(P, groups_unl, groups_lab, y_lab):
    """P: probas des lignes non étiquetées ; groups_*: clé de groupe ; y_lab: labels connus (0/1/2)."""
    pred = P.argmax(1).copy()
    gl = pd.Series(groups_lab)
    known = pd.crosstab(gl, pd.Series(y_lab)).reindex(columns=[0, 1, 2], fill_value=0)
    nl = gl.value_counts()
    gu = pd.Series(groups_unl)
    for g, rows in gu.groupby(gu).groups.items():
        rows = np.asarray(rows)
        kc = known.loc[g].values if g in known.index else np.zeros(3)
        n_total = len(rows) + nl.get(g, 0)
        pred[rows] = assign(P[rows], remaining_counts(n_total, kc, len(rows)))
    return pred
