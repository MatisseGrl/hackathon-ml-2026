"""Quotas par user ou par user + sport : cohérence avec les comptes du train (aucun compte au-dessus de sa cible)
et log-vraisemblance hypergéométrique, pour chaque règle d'arrondi. Puis score de validation de la soumission 3."""
import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from math import lgamma
import blendn as B
lc = lambda n, k: lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
for by in ("user", "user_sport"):
    gtr, gte = B.group_keys(by)
    K = pd.crosstab(pd.Series(gtr), pd.Series(B.yi)).reindex(columns=[0, 1, 2], fill_value=0)
    N = pd.Series(gtr).value_counts().add(pd.Series(gte).value_counts(), fill_value=0).reindex(K.index).astype(int)
    for rule in ("frac", "qcut", "pct"):
        T = np.array([B.targets(n, rule) for n in N.values])
        over = (K.values > T).any(1).sum()
        ll = sum(lc(T[i, c], K.values[i, c]) for i in range(len(T)) for c in range(3) if (K.values[i] <= T[i]).all())
        print(f"{by:10s} {rule:4s} groupes {len(K)} ; groupes incohérents (train > cible) {over} ; log-vrais. (groupes cohérents) {ll:.1f}", flush=True)
names = ["person", "f2"]
for by in ("user", "user_sport"):
    for rule in ("frac", "pct"):
        s, _ = B.score(B.prepare(names, rule, by), names, np.array([0.6, 0.4]))
        print(f"sub3 quotas {by:10s} {rule:4s} {s.mean():.4f} ± {s.std():.4f} plis {np.round(s, 4)}", flush=True)
