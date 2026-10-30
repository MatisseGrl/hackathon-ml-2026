"""a1 « même parcours » : variables sans cible par parcours (user, sport, km, altitude, dénivelé) + 2e niveau HGB
sur log des probas du mélange égal des 5 caches (hors pli). Lancer depuis essais/RMYU :
OMP_NUM_THREADS=2 python -u lab2/a1/a1_route.py
Produit cache_a1_route.npz (avec variables parcours) et cache_a1_ctrl.npz (témoin : 2e niveau sans variables parcours)."""
import sys, os, time
sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
import blendn as B
from sklearn.ensemble import HistGradientBoostingClassifier

t0 = time.time()
tr, te = B.train, B.test
ntr, nte = len(tr), len(te)
D = pd.concat([tr.drop(columns=["intensity"]), te[tr.drop(columns=["intensity"]).columns]], ignore_index=True)
alt = D.altitude_mean_m.where((D.altitude_mean_m > -500) & (D.altitude_mean_m < 9000))
gain = D.altitude_gain_m.where((D.altitude_gain_m >= 0) & (D.altitude_gain_m < 10000))
km = D.distance_km.round()
us = D.user_id.astype(str) + "_" + D.sport.astype(str)
keys = {"r1": us + "_" + km.astype(str) + "_" + (alt / 25).round().astype(str) + "_" + (gain / 25).round().astype(str),
        "r2": us + "_" + km.astype(str) + "_" + (alt / 25).round().astype(str),
        "r3": us + "_" + km.astype(str)}
V = ["speed_kmh", "max_speed_kmh", "duration_min", "speed_std_kmh"]
F = pd.DataFrame(index=D.index)
g = D.groupby(us)
zus = (D.speed_kmh - g.speed_kmh.transform("mean")) / g.speed_kmh.transform("std")
for name, k in keys.items():
    gg = D.groupby(k)
    n = gg.speed_kmh.transform("size")
    F[f"{name}_n"] = n
    ok = n >= 5
    for v in V:
        m, s = gg[v].transform("mean"), gg[v].transform("std")
        F[f"{name}_z_{v}"] = ((D[v] - m) / s).where(ok & (s > 0))
        F[f"{name}_rk_{v}"] = gg[v].rank(pct=True).where(ok)
    z = F[f"{name}_z_speed_kmh"]
    msk = (z.notna() & zus.notna()).values
    b = np.polyfit(zus.values[msk], z.values[msk], 1)
    F[f"{name}_res"] = z - (b[0] * zus + b[1])
    print(name, "couverture train n>=5 :", round(float(ok[:ntr].mean()), 3), "pente", np.round(b, 3), flush=True)
Ftr, Fte = F.iloc[:ntr].values, F.iloc[ntr:].values
print("variables", F.shape, f"{time.time()-t0:.0f}s", flush=True)

names = ["person", "f2", "neighbor", "person7", "resid2"]
C = {n: np.load(f"cache_{n}.npz") for n in names}
va = [C["person"][f"va{f}"] for f in range(5)]
O = np.zeros((ntr, 3)); T = []
for f in range(5):
    nv = len(va[f])
    Pm = sum(C[n][f"P{f}"] for n in names) / 5
    O[va[f]] = Pm[:nv]; T.append(Pm[nv:])
yi = B.yi

from sklearn.metrics import f1_score
# inclinaison ordinale du mélange égal : p_c * exp(gamma * c * résidu_parcours), puis doublons + quotas (même protocole)
folds = B.prepare(names, "pct", "user_sport", oracle=True)
cands = [("ref", None, 0.0)] + [(c, c, gm) for c in ["r1_res", "r2_res", "r3_res"] for gm in [-0.3, -0.15, -0.05, 0.05, 0.15, 0.3]]
S = []
for lbl, col, gm in cands:
    s = []
    for f, fd in enumerate(folds):
        nv = len(va[f]); lab = np.setdiff1d(np.arange(ntr), va[f])
        P = sum(C[n][f"P{f}"] for n in names) / 5
        if col is not None:
            x = np.nan_to_num(np.clip(np.r_[F[col].values[:ntr][va[f]], F[col].values[ntr:]], -4, 4))
            P = P * np.exp(gm * np.outer(x, [0, 1, 2])); P /= P.sum(1, keepdims=True)
        ku_f = np.concatenate([B.kf_tr[va[f]], B.kf_te]); ku_p = np.concatenate([B.kp_tr[va[f]], B.kp_te])
        Q = B.adjust(P, ku_f, ku_p, B.kf_tr[lab], B.kp_tr[lab], yi[lab], w4=0, wfull=1.0)[:nv]
        s.append(f1_score(yi[va[f]], B.predict(Q, fd["pl"])[:nv], average="macro"))
    S.append(s); s = np.array(s)
    print(f"{lbl} gamma={gm:+.2f}: {s.mean():.4f} plis {np.round(s - np.array(S[0]), 4)}", flush=True)
S = np.array(S); nest, idx = B.nested(S); d = nest - S[0]
print(f"imbriqué (réf incluse) : {nest.mean():.4f} ± {nest.std():.4f} Δ {d.mean():+.4f} plis {np.round(d, 4)} ; choix",
      [cands[i][0] + f"{cands[i][2]:+.2f}" for i in idx], flush=True)
S2 = S[1:]; nest2, idx2 = B.nested(S2); d2 = nest2 - S[0]
print(f"imbriqué (sans réf) : {nest2.mean():.4f} Δ {d2.mean():+.4f} plis {np.round(d2, 4)} gagnés {int((d2 > 0).sum())}/5", flush=True)
print(f"fini {time.time()-t0:.0f}s", flush=True)
