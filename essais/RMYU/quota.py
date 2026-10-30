"""Affectation sous contrainte : dans chaque groupe (user[, sport]), les 3 classes font chacune 1/3 de (train+test).
Les comptes restants = N/3 - comptes connus du train ; on affecte les lignes non étiquetées pour maximiser sum log P.
Uniquement NumPy / pandas (pas de scipy) : voir assign()."""
import numpy as np, pandas as pd

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


def assign(P, counts, eps=1e-12):
    """Affectation optimale (min sum -log P) avec exactement counts[k] lignes dans la classe k, en NumPy pur.
    1) départ réalisable : on remplit les classes par ordre de « surplus » de probabilité ;
    2) amélioration : tant qu'un échange entre 2 classes, ou une rotation entre les 3, diminue le coût, on l'applique.
    Avec 3 classes, une affectation sans échange ni rotation améliorante est optimale (plus de cycle négatif)."""
    C = -np.log(np.clip(P, 1e-9, 1))
    n = len(P)
    out = np.full(n, -1)
    free = np.ones(n, bool)
    for k in np.argsort(counts):                       # classes les plus petites d'abord
        idx = np.where(free)[0]
        others = [j for j in range(3) if j != k]
        gain = C[idx][:, others].min(1) - C[idx, k]    # combien la ligne « préfère » k
        take = idx[np.argsort(-gain)[:counts[k]]]
        out[take] = k; free[take] = False
    for _ in range(10 * n + 10):
        # d[a, b] = coût minimal pour faire passer une ligne de la classe a à la classe b
        d = np.full((3, 3), np.inf); arg = np.full((3, 3), -1)
        for a in range(3):
            ia = np.where(out == a)[0]
            if len(ia) == 0:
                continue
            for b in range(3):
                if a != b:
                    delta = C[ia, b] - C[ia, a]
                    j = delta.argmin(); d[a, b] = delta[j]; arg[a, b] = ia[j]
        best, move = -eps, None
        for a in range(3):
            for b in range(a + 1, 3):
                if d[a, b] + d[b, a] < best:
                    best, move = d[a, b] + d[b, a], [(a, b), (b, a)]
        for cyc in [(0, 1, 2), (0, 2, 1)]:
            s = d[cyc[0], cyc[1]] + d[cyc[1], cyc[2]] + d[cyc[2], cyc[0]]
            if s < best:
                best, move = s, [(cyc[0], cyc[1]), (cyc[1], cyc[2]), (cyc[2], cyc[0])]
        if move is None:
            break
        rows = [arg[a, b] for a, b in move]
        for (a, b), r in zip(move, rows):
            out[r] = b
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
