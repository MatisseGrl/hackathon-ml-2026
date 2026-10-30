"""Mélanges fixés A PRIORI (pas de recherche de poids, donc pas de biais de sélection), doublons exacts + quotas.
Comparaison pli par pli à la soumission 3 (0,6 person + 0,4 f2)."""
import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np
import blendn as B
names = ["person", "f2", "neighbor", "person7", "resid2"]
cands = {"sub3 0.6p+0.4f2": [0.6, 0.4, 0, 0, 0],
         "graines 0.3p+0.3p7+0.4f2": [0.3, 0.4, 0, 0.3, 0],
         "0.3p+0.3p7+0.2f2+0.2r2": [0.3, 0.2, 0, 0.3, 0.2],
         "égal 5": [0.2] * 5,
         "0.25p+0.25p7+0.2f2+0.15n+0.15r2": [0.25, 0.2, 0.15, 0.25, 0.15],
         "0.3p+0.3p7+0.2f2+0.2n": [0.3, 0.2, 0.2, 0.3, 0]}
for rule in ("frac", "pct"):
    folds = B.prepare(names, rule)
    base, _ = B.score(folds, names, np.array(cands["sub3 0.6p+0.4f2"]))
    for k, w in cands.items():
        s, a = B.score(folds, names, np.array(w))
        d = s - base
        print(f"{rule:4s} {k:34s} {s.mean():.4f} ± {s.std():.4f} | argmax {a.mean():.4f} | Δ {d.mean():+.4f} plis gagnés {(d > 0).sum()}/5 {np.round(d, 4)}", flush=True)
