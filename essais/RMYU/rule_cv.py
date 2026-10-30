"""Score de validation de la soumission 3 (0,6 person + 0,4 f2, doublons exacts + quotas) selon la règle d'arrondi."""
import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np
import blendn as B
names = ["person", "f2"]
for rule in ("frac", "qcut", "pct"):
    s, a = B.score(B.prepare(names, rule), names, np.array([0.6, 0.4]))
    print(f"{rule:5s} dup+quota {s.mean():.4f} ± {s.std():.4f} plis {np.round(s, 4)}", flush=True)
