"""a3 : seuils locaux dans l'espace du score (lancer depuis essais/RMYU : python -u lab2/a3/a3.py).
Score E = P(mod) + 2 P(hard) du mélange égal des 5 caches (hors pli). Pour chaque séance à prédire, ancres = séances
étiquetées du même user+sport hors pli(s) interdits, pondérées par exp(-|dt|/h), h in {1,3,12 mois, global}.
Variables : densité, label moyen pondéré, percentile local du score, part des ancres de chaque classe au-dessus,
écart aux seuils locaux (milieux des moyennes pondérées des scores par classe). 2e niveau par pli."""
import sys, time
sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
import blendn as B
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

T0 = time.time()
names = ["person", "f2", "neighbor", "person7", "resid2"]
C = {n: np.load(f"cache_{n}.npz") for n in names}
va = [C["person"][f"va{f}"] for f in range(5)]
tr, te = B.train, B.test
ntr, nte = len(tr), len(te)
fold = np.full(ntr, -1)
for f in range(5):
    fold[va[f]] = f
Ptr = np.zeros((ntr, 3))
for f in range(5):
    Ptr[va[f]] = np.mean([C[n][f"P{f}"][:len(va[f])] for n in names], 0)
Pte = [np.mean([C[n][f"P{f}"][len(va[f]):] for n in names], 0) for f in range(5)]
sc = lambda P: P[:, 1] + 2 * P[:, 2]
s_tr = sc(Ptr); s_te = [sc(p) for p in Pte]
y = B.yi
gtr, gte = B.group_keys("user_sport")
tnum = lambda d: (12 * d.year + d.month + d.weekday / 7).values.astype(float)
ttr, tte = tnum(tr), tnum(te)
Itr = pd.Series(np.arange(ntr)).groupby(np.asarray(gtr)).indices
Ite = pd.Series(np.arange(nte)).groupby(np.asarray(gte)).indices
print("groupes", len(Itr), "taille max", max(len(v) for v in Itr.values()),
      "somme n*a", sum((len(v) + len(Ite.get(g, []))) * len(v) for g, v in Itr.items()), flush=True)
HS = [1.0, 3.0, 12.0, None]
EX = int(sys.argv[sys.argv.index("--excl") + 1]) if "--excl" in sys.argv else 0
SUF = f"_ex{EX}" if EX else ""
LAB = "--labonly" in sys.argv
SUF += "_lab" if LAB else ""
COLS = [j for j in range(8 * len(HS)) if (j % 8 in (0, 1) or not LAB)]
mtr, mte = (12 * tr.year + tr.month).values, (12 * te.year + te.month).values
DFLT = np.tile([0, 1, 0.5, 0.5, 0.5, 0.5, 0, 0], len(HS)).astype(float)


def feats(k):
    Ftr = np.full((ntr, len(DFLT)), np.nan); Fte = np.full((nte, len(DFLT)), np.nan)
    for g, R in Itr.items():
        T = Ite.get(g, np.array([], int))
        rs = np.concatenate([s_tr[R], s_te[k][T]]); rt = np.concatenate([ttr[R], tte[T]])
        rf = np.concatenate([fold[R], np.full(len(T), -1)])
        sA, yA, fA, tA = s_tr[R], y[R].astype(float), fold[R], ttr[R]
        # ligne du pli k ou test : ancres hors pli k ; ligne d'entraînement du pli j : ancres hors {j, k}
        M = ((fA[None, :] != k) & (fA[None, :] != rf[:, None]))
        rmo = np.concatenate([mtr[R], mte[T]])
        M = (M & (np.abs(rmo[:, None] - mtr[R][None, :]) >= EX)).astype(float)
        nf = np.where((rf == k) | (rf == -1), 4.0, 3.0)
        dt = np.abs(rt[:, None] - tA[None, :])
        above = (sA[None, :] > rs[:, None]).astype(float)
        out = []
        with np.errstate(invalid="ignore", divide="ignore"):
            for h in HS:
                K = M * np.exp(-dt / h) if h else M
                W = K.sum(1)
                ybar = K @ yA / W
                pct = (K * (1 - above)).sum(1) / W
                ab, mu = [], []
                for c in range(3):
                    Kc = K * (yA == c)[None, :]
                    Wc = Kc.sum(1)
                    ab.append((Kc * above).sum(1) / Wc); mu.append(Kc @ sA / Wc)
                out += [np.log1p(W / nf), ybar, pct, ab[0], ab[1], ab[2],
                        rs - (mu[0] + mu[1]) / 2, rs - (mu[1] + mu[2]) / 2]
        F = np.column_stack(out)
        Ftr[R] = F[:len(R)]; Fte[T] = F[len(R):]
    for F in (Ftr, Fte):
        i = np.where(~np.isfinite(F)); F[i] = DFLT[i[1]]
    return Ftr, Fte


lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
hgb = lambda: HistGradientBoostingClassifier(max_leaf_nodes=15, l2_regularization=1.0, learning_rate=0.05,
                                             max_iter=250, early_stopping=False, random_state=0)
VAR = {"seuil": (lr, True)} if LAB else {"seuil": (lr, True), "temoin": (lr, False)}
if "--hgb" in sys.argv:
    VAR = {"hgb": (hgb, True), "hgbtem": (hgb, False)}
res = {v: {} for v in VAR}
base_tr = np.column_stack([np.log(np.clip(Ptr, 1e-6, 1)), s_tr])
for k in range(5):
    Ftr, Fte = feats(k)
    base_te = np.column_stack([np.log(np.clip(Pte[k], 1e-6, 1)), s_te[k]])
    trn = fold != k
    for v, (mk, use) in VAR.items():
        Xtr = np.hstack([base_tr, Ftr[:, COLS]]) if use else base_tr
        Xte = np.hstack([base_te, Fte[:, COLS]]) if use else base_te
        m = mk().fit(Xtr[trn], y[trn])
        res[v][f"P{k}"] = np.vstack([m.predict_proba(Xtr[va[k]]), m.predict_proba(Xte)])
        res[v][f"va{k}"] = va[k]
    print(f"pli {k} fini {time.time()-T0:.0f}s", flush=True)
for v in VAR:
    res[v]["Pfull"] = np.mean([res[v][f"P{k}"][len(va[k]):] for k in range(5)], 0)
    fn = f"cache_a3_{v}{SUF}.npz"
    import os
    assert not os.path.exists(fn), fn
    np.savez(fn, **res[v])
mine = [f"a3_{v}{SUF}" for v in VAR]
alln = names + mine
folds = B.prepare(alln, "pct", "user_sport", oracle=True)
ref, _ = B.score(folds, alln, np.r_[np.full(5, .2), np.zeros(len(mine))])
print(f"REF {ref.mean():.4f} ± {ref.std():.4f} plis {np.round(ref, 4)}", flush=True)
betas = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
for i, mn in enumerate(mine):
    S = []
    for b in betas:
        w = np.r_[np.full(5, (1 - b) / 5), np.zeros(len(mine))]; w[5 + i] = b
        S.append(B.score(folds, alln, w)[0])
    S = np.array(S)
    for b, s in zip(betas, S):
        d = s - ref
        print(f"{mn} beta={b:.1f} {s.mean():.4f} ± {s.std():.4f} delta {d.mean():+.4f} gagnés {int((d > 0).sum())}/5 "
              f"plis {np.round(d, 4)}", flush=True)
    nest, idx = B.nested(S)
    d = nest - ref
    print(f"{mn} IMBRIQUE {nest.mean():.4f} ± {nest.std():.4f} delta {d.mean():+.4f} gagnés {int((d > 0).sum())}/5 "
          f"plis {np.round(d, 4)} betas {[betas[j] for j in idx]}", flush=True)
print(f"fini {time.time()-T0:.0f}s", flush=True)
