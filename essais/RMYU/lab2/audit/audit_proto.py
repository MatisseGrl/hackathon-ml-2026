"""Audit protocole oracle : (2) règle pct user+sport ; (4) fraction de groupe val vs test ;
(3)/(4) oracle sur 1 pli (~15 % de chaque groupe) et sur 2 plis réunis (~30 %, proche du test ~25 %).
Lancer depuis essais/RMYU. N'écrit aucun fichier."""
import sys, time; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
import blendn as B
from post import adjust
t0 = time.time()
gtr, gte = B.group_keys("user_sport")
K = pd.crosstab(pd.Series(gtr), pd.Series(B.yi)).reindex(columns=[0, 1, 2], fill_value=0)
N = pd.Series(gtr).value_counts().add(pd.Series(gte).value_counts(), fill_value=0).reindex(K.index).astype(int)
T = np.array([B.targets(n, "pct") for n in N.values])
print("groupes train", len(K), "; train>cible (pct)", int((K.values > T).any(1).sum()),
      "; groupes test sans train", len(set(gte) - set(gtr)), flush=True)
rem = T - K.values
nte = pd.Series(gte).value_counts().reindex(K.index, fill_value=0).values
print("somme des restes = taille test du groupe :", bool((rem.sum(1) == nte).all()), flush=True)
names = ["person", "person7", "f2", "neighbor", "resid2"]
C = {n: np.load(f"cache_{n}.npz") for n in names}
va = [C["person"][f"va{f}"] for f in range(5)]
def sizes(g):
    s = pd.Series(g).value_counts(); r = pd.Series(g).map(s).values
    return f"part lignes en groupe de taille 1 : {np.mean(r == 1):.3f} ; <=3 : {np.mean(r <= 3):.3f} ; taille médiane par ligne {np.median(r):.0f}"
print("test     ", sizes(gte)); print("val pli 0", sizes(gtr[va[0]])); print("val 2 plis", sizes(gtr[np.concatenate([va[0], va[1]])]), flush=True)
blends = {"person": [1, 0, 0, 0, 0], "sub3 .6p+.4f2": [.6, 0, .4, 0, 0], "egal5": [.2] * 5,
          "egal4 sans r2": [.25, .25, .25, .25, 0], ".3p+.3p7+.4f2": [.3, .3, .4, 0, 0], "egal3 p,p7,f2": [1/3, 1/3, 1/3, 0, 0]}
def run(fset):
    rows = np.concatenate([va[f] for f in fset])
    lab = np.setdiff1d(np.arange(len(B.train)), rows)
    Q = {n: adjust(np.concatenate([C[n][f"P{f}"][:len(va[f])] for f in fset]), B.kf_tr[rows], B.kp_tr[rows],
                   B.kf_tr[lab], B.kp_tr[lab], B.yi[lab], w4=0, wfull=1.0) for n in names}
    g = pd.Series(gtr[rows])
    pl = [(np.asarray(r), np.bincount(B.yi[rows][r], minlength=3)) for r in g.groupby(g).indices.values()]
    return {k: f1_score(B.yi[rows], B.predict(sum(wi * Q[n] for wi, n in zip(w, names)), pl), average="macro") for k, w in blends.items()}
for lab_, sets in [("1 pli", [(f,) for f in range(5)]), ("2 plis", [(f, (f + 1) % 5) for f in range(5)])]:
    R = pd.DataFrame([run(s) for s in sets])
    print(f"--- oracle {lab_} ({time.time()-t0:.0f}s)")
    for k in R.columns:
        d = R[k] - R["sub3 .6p+.4f2"]; dp = R[k] - R["person"]
        print(f"{k:16s} {R[k].mean():.4f} ± {R[k].std(ddof=0):.4f} | Δsub3 {d.mean():+.4f} ({(d > 0).sum()}/5) {np.round(d.values, 4)} | Δperson {dp.mean():+.4f} ({(dp > 0).sum()}/5)", flush=True)
print(f"fini {time.time()-t0:.0f}s")
