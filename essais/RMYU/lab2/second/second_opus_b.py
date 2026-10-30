import sys, time
sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
import blendn as B
from sklearn.metrics import f1_score
t0 = time.time()
ALL = ["person","person7","person11","f2","f2s7","neighbor","neighbor7","resid2"]
folds = B.prepare(ALL, "pct", "user_sport", oracle=True)
gu_tr = B.train.user_id.values
cands = {
 "n5": ({"person":.2,"f2":.2,"neighbor":.2,"person7":.2,"resid2":.2}, "user_sport"),
 "a_n3": ({"person":.6,"f2":.4}, "user"),
 "b_n4": ({"person":1.0}, "user_sport"),
 "c_fam8": ({"person":.4/3,"person7":.4/3,"person11":.4/3,"f2":.1,"f2s7":.1,"neighbor":.1,"neighbor7":.1,"resid2":.2}, "user_sport"),
 "d_eq8": ({n:1/8 for n in ALL}, "user_sport"),
}
N = len(B.yi); y = B.yi
OOF = {k: np.full(N, -1) for k in cands}; FOLD = np.full(N, -1)
for f, fd in enumerate(folds):
    va = fd["va"]; nv = len(va); FOLD[va] = f
    g = pd.Series(gu_tr[va]); pl_user = [(np.asarray(r), np.bincount(y[va][r], minlength=3)) for r in g.groupby(g).indices.values()]
    for k, (w, by) in cands.items():
        Q = sum(wi * fd["Q"][n] for n, wi in w.items())
        OOF[k][va] = B.predict(Q, fd["pl"] if by == "user_sport" else pl_user)[:nv]
np.savez("lab2/second/oof_second_opus_b.npz", FOLD=FOLD, y=y, **OOF)
print("N", N, "fold sizes", np.bincount(FOLD), flush=True)
print("full OOF macroF1:", {k: round(f1_score(y, v, average="macro"), 4) for k, v in OOF.items()}, flush=True)
print("per-fold d vs n5:", {k: np.round([f1_score(y[FOLD==f], OOF[k][FOLD==f], average="macro") - f1_score(y[FOLD==f], OOF["n5"][FOLD==f], average="macro") for f in range(5)], 4).tolist() for k in cands if k != "n5"}, flush=True)
def mf1(Y, P):
    s = 0
    for c in range(3):
        tp = ((P == c) & (Y == c)).sum(1); d = (P == c).sum(1) + (Y == c).sum(1)
        s = s + np.where(d > 0, 2 * tp / np.maximum(d, 1), 0)
    return s / 3
rng = np.random.default_rng(0); n = 3835; Bn = 2000
for mode in ["within_fold", "pooled"]:
    if mode == "pooled":
        I = np.array([rng.choice(N, n, replace=False) for _ in range(Bn)])
    else:
        I = np.array([rng.choice(np.where(FOLD == b % 5)[0], n, replace=False) for b in range(Bn)])
    Y = y[I]; F5 = mf1(Y, OOF["n5"][I])
    print(f"== {mode} draws={len(I)} size={I.shape[1]} meanF1_5={F5.mean():.4f} sdF1_5={F5.std():.4f}", flush=True)
    for k in cands:
        if k == "n5": continue
        FX = mf1(Y, OOF[k][I]); d = FX - F5
        print(f"{k}: dis={(OOF[k] != OOF['n5']).mean():.4f} mean_d={d.mean():+.5f} sd_d={d.std():.5f} P(X>5)={(d>0).mean():.3f} gain={np.maximum(d,0).mean():.5f} meanF1_X={FX.mean():.4f}", flush=True)
print("fin", round(time.time()-t0), flush=True)
