"""Protocole fidèle à la soumission : quotas sur la validation seule avec ses comptes réels par groupe (user + sport).
Caches seuls, soumission 3, mélanges fixés a priori ; puis grille 5 caches + sélection imbriquée."""
import sys, time; sys.path.append("../.."); sys.path.append(".")
import numpy as np
from joblib import Parallel, delayed
import blendn as B
names = ["person", "f2", "neighbor", "person7", "resid2"]
t0 = time.time()
folds = B.prepare(names, "pct", "user_sport", oracle=True)
cands = {"person": [1, 0, 0, 0, 0], "f2": [0, 1, 0, 0, 0], "neighbor": [0, 0, 1, 0, 0], "person7": [0, 0, 0, 1, 0], "resid2": [0, 0, 0, 0, 1],
         "sub3 0.6p+0.4f2": [0.6, 0.4, 0, 0, 0], "graines 0.3p+0.3p7+0.4f2": [0.3, 0.4, 0, 0.3, 0],
         "0.3p+0.3p7+0.2f2+0.2r2": [0.3, 0.2, 0, 0.3, 0.2], "égal 5": [0.2] * 5,
         "0.25p+0.25p7+0.2f2+0.15n+0.15r2": [0.25, 0.2, 0.15, 0.25, 0.15]}
base, _ = B.score(folds, names, np.array(cands["sub3 0.6p+0.4f2"]))
for k, w in cands.items():
    s, a = B.score(folds, names, np.array(w)); d = s - base
    print(f"{k:34s} oracle {s.mean():.4f} ± {s.std():.4f} | argmax {a.mean():.4f} | Δ sub3 {d.mean():+.4f} plis gagnés {(d > 0).sum()}/5 {np.round(d, 4)}", flush=True)
W = list(B.grid(5, 0.1))
res = Parallel(n_jobs=10)(delayed(B.score)(folds, names, w) for w in W)
S = np.array([r[0] for r in res]); m = S.mean(1)
for i in np.argsort(-m)[:8]:
    print(" ".join(f"{n}={w:.1f}" for n, w in zip(names, W[i])), f"{m[i]:.4f} ± {S[i].std():.4f} plis {np.round(S[i], 4)}", flush=True)
nest, idx = B.nested(S)
print(f"imbriqué : {nest.mean():.4f} ± {nest.std():.4f} plis {np.round(nest, 4)} Δ sub3 {(nest - base).mean():+.4f} ({((nest - base) > 0).sum()}/5) ; choix",
      [" ".join(f"{w:.1f}" for w in W[i]) for i in idx], f"({time.time()-t0:.0f}s)", flush=True)
np.savez("grid_oracle_us_pct.npz", W=np.array(W), S=S)
