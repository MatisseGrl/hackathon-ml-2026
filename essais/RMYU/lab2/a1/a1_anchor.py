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

from sklearn.linear_model import LogisticRegression
# ancres AVEC labels, strictement par plis : pli évalué k -> labels des plis != k (validation k et test) ;
# ligne d'entraînement du pli j -> labels des plis hors {j, k}. La ligne elle-même n'est jamais dans ses ancres.
fold_of = np.full(ntr, -1)
for f in range(5):
    fold_of[va[f]] = f
sp = D.speed_kmh.values
sd_us = g.speed_kmh.transform("std").values
levels = {"r2": keys["r2"].values, "r3": keys["r3"].values}


def anchors(allowed):
    cols = {}
    for name, kv in levels.items():
        df = pd.DataFrame({"k": kv[:ntr][allowed], "y": yi[allowed], "s": sp[:ntr][allowed]})
        ser = pd.Series(kv)
        cols[f"{name}_nlab"] = ser.map(df.groupby("k").size()).fillna(0).values
        cols[f"{name}_ylab"] = ser.map(df.groupby("k").y.mean()).values
        for c in range(3):
            mc = df[df.y == c].groupby("k").s.mean()
            cols[f"{name}_d{c}"] = (sp - ser.map(mc).values) / sd_us
    return pd.DataFrame(cols)


AF = {}
for k in range(5):
    fk = anchors(fold_of != k)
    Xa = fk.values.copy()
    for j in range(5):
        if j != k:
            Xa[va[j]] = anchors((fold_of != k) & (fold_of != j)).values[va[j]]
    AF[k] = (Xa[:ntr], Xa[ntr:])
acols = list(fk.columns)
print("ancres", acols, f"{time.time()-t0:.0f}s", flush=True)


def run(kind, tag, fill=False):
    path = f"cache_a1_{tag}.npz"
    assert not os.path.exists(path), path
    out, Pf = {}, []
    for k in range(5):
        lab = np.setdiff1d(np.arange(ntr), va[k])
        Atr, Ate = AF[k]
        if fill:  # pour la régression logistique : NaN -> 0 + indicateurs
            Atr = np.hstack([np.nan_to_num(np.clip(Atr, -4, 4)), np.isnan(Atr).astype(float)])
            Ate = np.hstack([np.nan_to_num(np.clip(Ate, -4, 4)), np.isnan(Ate).astype(float)])
            Atr[:, [0, 5]] = np.log1p(AF[k][0][:, [0, 5]]); Ate[:, [0, 5]] = np.log1p(AF[k][1][:, [0, 5]])
        Xtr = np.hstack([np.log(np.clip(O, 1e-6, 1)), Atr])
        Xte = np.hstack([np.log(np.clip(T[k], 1e-6, 1)), Ate])
        if kind == "hgb":
            m = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.05, max_iter=200,
                                               l2_regularization=1.0, early_stopping=False, random_state=0)
        else:
            m = LogisticRegression(C=1.0, max_iter=3000)
        m.fit(Xtr[lab], yi[lab])
        pv, pt = m.predict_proba(Xtr[va[k]]), m.predict_proba(Xte)
        out[f"va{k}"] = va[k]; out[f"P{k}"] = np.vstack([pv, pt]); Pf.append(pt)
    out["Pfull"] = np.mean(Pf, 0)
    np.savez(path, **out)
    print("sauvé", path, f"{time.time()-t0:.0f}s", flush=True)


run("hgb", "anchor")
run("lr", "anchorlr", fill=True)

for tag in ["anchor", "anchorlr"]:
    nm = names + [f"a1_{tag}"]
    folds = B.prepare(nm, "pct", "user_sport", oracle=True)
    ref, _ = B.score(folds, nm, np.r_[np.full(5, 0.2), 0.0])
    S = []
    print(f"\n[{tag}] référence {ref.mean():.4f}", flush=True)
    for beta in [0.3, 0.5, 0.7, 1.0]:
        s, a = B.score(folds, nm, np.r_[np.full(5, (1 - beta) / 5), beta])
        S.append(s); d = s - ref
        print(f"  beta={beta}: {s.mean():.4f} ± {s.std():.4f} | Δ {d.mean():+.4f} plis {np.round(d, 4)} gagnés {int((d > 0).sum())}/5"
              f" | argmax {a.mean():.4f}", flush=True)
    nest, idx = B.nested(np.array(S))
    d = nest - ref
    print(f"  imbriqué : {nest.mean():.4f} ± {nest.std():.4f} | Δ {d.mean():+.4f} plis {np.round(d, 4)}"
          f" gagnés {int((d > 0).sum())}/5 ; choix beta {[[0.3, 0.5, 0.7, 1.0][i] for i in idx]}", flush=True)
print(f"fini {time.time()-t0:.0f}s", flush=True)
