"""a2 v3 (K=20 ancres, stats ancres du même mois, contexte z de l'ancre dans le comparateur, 2e niveau HGB). a2 : comparateur par paires (HGB sur différences de variables) + ancres étiquetées proches dans le temps,
puis 2e niveau logistique sur [log mélange égal des 5 caches, u, moyennes P(séance > ancre) par classe d'ancre].
Lancer depuis essais/RMYU : python -u lab2/a2/a2_pair.py"""
import sys, time, os
sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import blendn as B
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)

train, test = B.train, B.test
ntr, nte = len(train), len(test); N = ntr + nte
cols0 = list(test.columns)
df = pd.concat([train[cols0], test[cols0]], ignore_index=True)
y = np.full(N, -1); y[:ntr] = B.yi
gtr, gte = B.group_keys("user_sport")
G = pd.factorize(pd.Series(np.concatenate([np.asarray(gtr).astype(str), np.asarray(gte).astype(str)])))[0]
alt = df.altitude_mean_m.where(df.altitude_mean_m.between(-500, 9000))
climb = (df.altitude_gain_m.clip(0, 5000) / df.distance_km.clip(lower=0.1)).clip(0, 200)
X = pd.DataFrame({"speed": df.speed_kmh, "vmax": df.max_speed_kmh, "vstd": df.speed_std_kmh,
                  "dur": df.duration_min, "dist": df.distance_km, "climb": climb})
Zs = []
for c in X.columns:
    m = X[c].groupby(G).transform("mean"); s = X[c].groupby(G).transform("std").fillna(1).replace(0, 1)
    Zs.append(((X[c] - m) / s).values); Zs.append(X[c].groupby(G).rank(pct=True).values)
Z = np.column_stack(Zs)
T = (12 * df.year + df.month + df.weekday / 7).values.astype(float)
HOUR = df.start_hour_utc.values.astype(float); DIST = df.distance_km.values; ALTv = alt.values
SP = pd.factorize(df.sport)[0]
C0 = np.load("cache_person.npz"); va = [C0[f"va{f}"] for f in range(5)]
fold = np.full(N, -1)
for f in range(5): fold[va[f]] = f
gidx = pd.Series(np.arange(N)).groupby(G).indices
log("groupes", len(gidx), "max", max(len(v) for v in gidx.values()))
rng = np.random.default_rng(0)

def pf(a, b):
    return np.column_stack([Z[a] - Z[b], np.abs(T[a] - T[b]), T[a] - T[b], np.abs(HOUR[a] - HOUR[b]),
                            np.abs(DIST[a] - DIST[b]), np.abs(ALTv[a] - ALTv[b]), Z[b], SP[a]])
NCAT = 2 * Z.shape[1] + 5

def make_pairs(mask, K=10):
    A, Bb = [], []
    for g, rows in gidx.items():
        L = rows[mask[rows]]
        if len(L) < 2: continue
        D = np.abs(T[L][:, None] - T[L][None, :]) + rng.random((len(L), len(L))) * 1e-3
        D[y[L][:, None] == y[L][None, :]] = np.inf
        k = min(K, len(L) - 1)
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        ok = np.isfinite(np.take_along_axis(D, idx, 1))
        A.append(np.repeat(L, k).reshape(len(L), k)[ok]); Bb.append(L[idx][ok])
    a = np.concatenate(A); b = np.concatenate(Bb)
    return np.concatenate([a, b]), np.concatenate([b, a])

def anchor_feats(m, qmask, cmask, K=20):
    out = np.full((N, 9), np.nan)
    QA, AA = [], []
    for g, rows in gidx.items():
        Q = rows[qmask[rows]]; Cn = rows[cmask[rows]]
        if len(Q) == 0 or len(Cn) == 0: continue
        D = np.abs(T[Q][:, None] - T[Cn][None, :]) + rng.random((len(Q), len(Cn))) * 1e-3
        k = min(K, len(Cn))
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        QA.append(np.repeat(Q, k)); AA.append(Cn[idx].ravel())
    q = np.concatenate(QA); a = np.concatenate(AA)
    p = m.predict_proba(pf(q, a))[:, 1]
    d = pd.DataFrame({"q": q, "v": y[a] + 2 * p - 1, "p": p, "c": y[a]})
    u = d.groupby("q").v.mean(); out[u.index.values, 0] = u.values
    for c in range(3):
        mc = d[d.c == c].groupby("q").p.mean(); out[mc.index.values, 1 + c] = mc.values
    sz = d.groupby("q").size(); out[sz.index.values, 4] = sz.values
    dn = d[np.floor(T[q]) == np.floor(T[a])]
    un = dn.groupby("q").v.mean(); out[un.index.values, 5] = un.values
    for c in range(3):
        mc = dn[dn.c == c].groupby("q").p.mean(); out[mc.index.values, 6 + c] = mc.values
    lab = (q < ntr) & (y[q] != y[a])
    acc = np.mean((p[lab] > 0.5) == (y[q][lab] > y[a][lab]))
    return out, acc

F_tr = np.full((ntr, 9), np.nan); F_te = []
for j in range(5):
    trm = (fold >= 0) & (fold != j)
    a, b = make_pairs(trm)
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=0,
                                       categorical_features=[NCAT])
    m.fit(pf(a, b), (y[a] > y[b]).astype(int))
    out, acc = anchor_feats(m, (fold == j) | (fold == -1), trm)
    F_tr[va[j]] = out[va[j]]; F_te.append(out[ntr:])
    log(f"passe {j}: {len(a)} paires, exactitude comparateur (pli tenu, paires de classes diff.) {acc:.4f}; "
        f"corr(u, y) pli {np.corrcoef(out[va[j], 0][~np.isnan(out[va[j], 0])], y[va[j]][~np.isnan(out[va[j], 0])])[0,1]:.4f}")
np.savez("lab2/a2/feats_pair3.npz", F_tr=F_tr, F_te=np.array(F_te))

names = ["person", "f2", "neighbor", "person7", "resid2"]
C = {n: np.load(f"cache_{n}.npz") for n in names}
Pm_tr = np.zeros((ntr, 3)); Pm_te = []
for f in range(5):
    nv = len(va[f]); P = sum(C[n][f"P{f}"] for n in names) / 5
    Pm_tr[va[f]] = P[:nv]; Pm_te.append(P[nv:])

from sklearn.ensemble import HistGradientBoostingClassifier as HGB
tag = "a2_pair3hgb"; fn = f"cache_{tag}.npz"; assert not os.path.exists(fn)
def feat(Pm, F): return np.column_stack([np.log(np.clip(Pm, 1e-6, 1)), F])
out = {}
for k in range(5):
    tr = np.where((fold[:ntr] >= 0) & (fold[:ntr] != k))[0]
    m2 = HGB(max_iter=150, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=50, l2_regularization=1.0,
             random_state=0).fit(feat(Pm_tr[tr], F_tr[tr]), y[tr])
    out[f"va{k}"] = va[k]
    out[f"P{k}"] = np.vstack([m2.predict_proba(feat(Pm_tr[va[k]], F_tr[va[k]])), m2.predict_proba(feat(Pm_te[k], F_te[k]))])
out["Pfull"] = np.mean([out[f"P{k}"][len(va[k]):] for k in range(5)], axis=0)
np.savez(fn, **out); log("sauvé", fn)
alln = names + [tag]
folds = B.prepare(alln, "pct", "user_sport", oracle=True)
ref, _ = B.score(folds, alln, np.array([0.2] * 5 + [0]))
log(f"REF {ref.mean():.4f} ± {ref.std():.4f} plis {np.round(ref, 4)}")
betas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]; S = []
for bt in betas:
    w = np.array([(1 - bt) / 5] * 5 + [bt]); sc_, _ = B.score(folds, alln, w); S.append(sc_); d = sc_ - ref
    log(f"{tag} beta={bt:.1f} {sc_.mean():.4f} ± {sc_.std():.4f} delta {d.mean():+.4f} plis gagnés {(d > 0).sum()}/5 delta plis {np.round(d, 4)}")
S = np.array(S); nest, idx = B.nested(S); d = nest - ref
log(f"{tag} IMBRIQUÉ {nest.mean():.4f} ± {nest.std():.4f} delta {d.mean():+.4f} plis gagnés {(d > 0).sum()}/5 "
    f"delta {np.round(d, 4)} betas choisis {[betas[i] for i in idx]}")
log("fini")
