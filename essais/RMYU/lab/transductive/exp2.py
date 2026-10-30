"""Coût d'affectation sous quotas : -log P (référence) vs -P (maximise l'espérance du nombre de bons) vs -P^a."""
import sys, numpy as np, trans as T
from sklearn.metrics import f1_score
C = dict(np.load(sys.argv[1]))
def ev(tf):
    out = []
    for f in range(5):
        va = C[f"va{f}"]; P = C[f"P{f}"]; nv = len(va)
        lab = np.setdiff1d(np.arange(T.ntr), va)
        g = np.concatenate([va, T.ntr + np.arange(T.nte)])
        Q = T.adjust(P, T.kf_all[g], T.kp_all[g], T.kf_all[lab], T.kp_all[lab], T.yi[lab], w4=0.5)
        pred = T.constrained_predict(tf(Q), T.u[g], T.u[lab], T.yi[lab])
        out.append((f1_score(T.yi[va], pred[:nv], average="macro"), 0))
    return np.array(out)
T.rep("cost -log P (ref)", ev(lambda Q: Q))
for a in [3.0, 1.0, 0.5, 0.25]:
    T.rep(f"cost -(Q^{a})/a  [lin]", ev(lambda Q, a=a: np.exp((Q ** a - 1) / a)))
