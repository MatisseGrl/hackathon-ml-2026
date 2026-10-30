"""Variante : même variables de paires (lab2/a2/feats_pair.npz produit par a2_pair.py), 2e niveau = petit HGB.
Lancer depuis essais/RMYU : python -u lab2/a2/a2_lvl2hgb.py"""
import sys, time, os
sys.path.append("../.."); sys.path.append(".")
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import blendn as B
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
ntr = len(B.train); y = B.yi
Fz = np.load("lab2/a2/feats_pair.npz"); F_tr, F_te = Fz["F_tr"], Fz["F_te"]
names = ["person", "f2", "neighbor", "person7", "resid2"]
C = {n: np.load(f"cache_{n}.npz") for n in names}
va = [C["person"][f"va{f}"] for f in range(5)]
fold = np.full(ntr, -1)
for f in range(5): fold[va[f]] = f
Pm_tr = np.zeros((ntr, 3)); Pm_te = []
for f in range(5):
    nv = len(va[f]); P = sum(C[n][f"P{f}"] for n in names) / 5
    Pm_tr[va[f]] = P[:nv]; Pm_te.append(P[nv:])
def feat(Pm, F): return np.column_stack([np.log(np.clip(Pm, 1e-6, 1)), F])
tag = "a2_pairhgb"; fn = f"cache_{tag}.npz"; assert not os.path.exists(fn)
out = {}
for k in range(5):
    tr = np.where(fold != k)[0]
    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=50,
                                       l2_regularization=1.0, random_state=0).fit(feat(Pm_tr[tr], F_tr[tr]), y[tr])
    out[f"va{k}"] = va[k]
    out[f"P{k}"] = np.vstack([m.predict_proba(feat(Pm_tr[va[k]], F_tr[va[k]])), m.predict_proba(feat(Pm_te[k], F_te[k]))])
out["Pfull"] = np.mean([out[f"P{k}"][len(va[k]):] for k in range(5)], axis=0)
np.savez(fn, **out); log("sauvé", fn)
alln = names + [tag]
folds = B.prepare(alln, "pct", "user_sport", oracle=True)
ref, _ = B.score(folds, alln, np.array([0.2] * 5 + [0]))
log(f"REF {ref.mean():.4f} ± {ref.std():.4f} plis {np.round(ref, 4)}")
betas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]; S = []
for bt in betas:
    w = np.array([(1 - bt) / 5] * 5 + [bt]); s, _ = B.score(folds, alln, w); S.append(s); d = s - ref
    log(f"{tag} beta={bt:.1f} {s.mean():.4f} ± {s.std():.4f} delta {d.mean():+.4f} plis gagnés {(d > 0).sum()}/5 delta plis {np.round(d, 4)}")
S = np.array(S); nest, idx = B.nested(S); d = nest - ref
log(f"{tag} IMBRIQUÉ {nest.mean():.4f} ± {nest.std():.4f} delta {d.mean():+.4f} plis gagnés {(d > 0).sum()}/5 "
    f"delta {np.round(d, 4)} betas choisis {[betas[i] for i in idx]}")
log("fini")
