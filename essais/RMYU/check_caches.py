"""Scores individuels des caches (argmax, doublons+quotas), contrôle de la soumission 3, et test de fuite :
une fuite des labels de validation rendrait les probas plus confiantes sur la validation que sur le test."""
import sys, time; sys.path.append("../.."); sys.path.append(".")
import numpy as np
import blendn as B
names = ["person", "f2", "neighbor", "person7", "resid2"]
t0 = time.time()
folds = B.prepare(names)
print(f"préparation {time.time()-t0:.0f}s")
for i, n in enumerate(names):
    w = np.eye(len(names))[i]
    s, a = B.score(folds, names, w)
    C = np.load(f"cache_{n}.npz")
    conf_va = np.mean([C[f"P{f}"][:len(C[f'va{f}'])].max(1).mean() for f in range(5)])
    conf_te = np.mean([C[f"P{f}"][len(C[f'va{f}']):].max(1).mean() for f in range(5)])
    conf_full = C["Pfull"].max(1).mean()
    print(f"{n:9s} argmax {a.mean():.4f} | dup+quota {s.mean():.4f} ± {s.std():.4f} | conf val {conf_va:.3f} test(pli) {conf_te:.3f} test(full) {conf_full:.3f}", flush=True)
s, a = B.score(folds, names, np.array([0.6, 0.4, 0, 0, 0]))
print(f"soumission 3 (0.6 person + 0.4 f2) : dup+quota {s.mean():.4f} ± {s.std():.4f} plis {np.round(s,4)} | argmax {a.mean():.4f}  ({time.time()-t0:.0f}s)")
