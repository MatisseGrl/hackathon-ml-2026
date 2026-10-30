"""Variante « stacker » : régression logistique multinomiale (explicable : coefficients) sur
 log P (après doublons) + pour chaque groupe g : log moyenne des probas des voisins NON étiquetés, log(1+n),
 log moyenne (lissée) des labels des voisins étiquetés.
Entraînée en validation croisée « croisée » : pour le pli f, apprise sur les lignes de validation des 4 autres plis
(probas hors-fold, labels connus) puis appliquée au pli f. Pour la soumission : apprise sur les 5 plis, appliquée au test.
Usage : python -u stack.py <cache.npz> [--pool test] [--submit out.csv]"""
import sys, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import trans as T

GROUPS = ("dh", "d", "ms", "m", "m1")


def feats(Q, gidx_u, gidx_l, y_l, pool_mask, groups=GROUPS, use_lab=True, unl=True):
    cols = [np.log(np.clip(Q, 1e-4, 1))]
    for g in groups:
        K, offs = T.KEYS[g]
        ku = K[gidx_u]
        s, c = T.nb(ku, ku[pool_mask], Q[pool_mask], offs)
        s[pool_mask] -= Q[pool_mask]; c[pool_mask] -= 1
        m = (s + Q) / (c + 1)[:, None]                 # moyenne avec soi-même comme a priori
        if unl:
            cols += [np.log(np.clip(m, 1e-4, 1)) - cols[0], np.log1p(c)[:, None]]
        if use_lab:
            sl, cl = T.nb(ku, K[gidx_l], np.eye(3)[y_l], offs)
            ml = (sl + 1.0) / (cl + 3.0)[:, None]
            cols += [np.log(ml), np.log1p(cl)[:, None]]
    return np.column_stack(cols)


def fold_data(C, f, pool_test_only, **kw):
    va = C[f"va{f}"]; P = C[f"P{f}"]; nv = len(va)
    lab = np.setdiff1d(np.arange(T.ntr), va)
    gidx_u = np.concatenate([va, T.ntr + np.arange(T.nte)])
    ku_f, ku_p = T.kf_all[gidx_u], T.kp_all[gidx_u]
    Q0 = T.adjust(P, ku_f, ku_p, T.kf_all[lab], T.kp_all[lab], T.yi[lab], w4=0.5)
    pool = np.r_[np.zeros(nv, bool), np.ones(T.nte, bool)] if pool_test_only else np.ones(len(gidx_u), bool)
    X = feats(Q0, gidx_u, lab, T.yi[lab], pool, **kw)
    return X, va, lab, gidx_u, ku_f, ku_p


def finish(Pr, ku_f, ku_p, lab, gidx_u):
    Q = T.adjust(Pr, ku_f, ku_p, T.kf_all[lab], T.kp_all[lab], T.yi[lab], w4=0, wfull=1.0)
    return T.constrained_predict(Q, T.u[gidx_u], T.u[lab], T.yi[lab])


def run(C, pool_test_only=False, Cr=0.1, **kw):
    D = [fold_data(C, f, pool_test_only, **kw) for f in range(5)]
    res = []
    for f in range(5):
        Xtr = np.vstack([D[g][0][:len(D[g][1])] for g in range(5) if g != f])
        ytr = np.concatenate([T.yi[D[g][1]] for g in range(5) if g != f])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        lr = LogisticRegression(C=Cr, max_iter=2000).fit((Xtr - mu) / sd, ytr)
        X, va, lab, gidx_u, ku_f, ku_p = D[f]
        Pr = lr.predict_proba((X - mu) / sd)
        pred = finish(Pr, ku_f, ku_p, lab, gidx_u)
        res.append((f1_score(T.yi[va], pred[:len(va)], average="macro"),
                    f1_score(T.yi[va], Pr[:len(va)].argmax(1), average="macro")))
    return np.array(res), D


def submit(C, path, pool_test_only=False, Cr=0.1, **kw):
    D = [fold_data(C, f, pool_test_only, **kw) for f in range(5)]
    Xtr = np.vstack([d[0][:len(d[1])] for d in D]); ytr = np.concatenate([T.yi[d[1]] for d in D])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    lr = LogisticRegression(C=Cr, max_iter=2000).fit((Xtr - mu) / sd, ytr)
    gidx_u = T.ntr + np.arange(T.nte); lab = np.arange(T.ntr)
    ku_f, ku_p = T.kf_all[gidx_u], T.kp_all[gidx_u]
    Q0 = T.adjust(C["Pfull"], ku_f, ku_p, T.kf_all[lab], T.kp_all[lab], T.yi, w4=0.5)
    X = feats(Q0, gidx_u, lab, T.yi, np.ones(T.nte, bool), **kw)
    Pr = lr.predict_proba((X - mu) / sd)
    pred = finish(Pr, ku_f, ku_p, lab, gidx_u)
    sub = pd.DataFrame({"workout_id": T.test.workout_id.values, "intensity": T.LAB[pred]})
    assert (sub.workout_id.values == T.sample.workout_id.values).all()
    sub.to_csv(path, index=False); np.save(path.replace(".csv", "_P.npy"), Pr)
    print("écrit", path, sub.intensity.value_counts().to_dict(), "coef", np.round(lr.coef_, 2).tolist(), flush=True)


if __name__ == "__main__":
    C = dict(np.load(sys.argv[1]))
    for pt in [False, True]:
        for kw in [dict(), dict(use_lab=False), dict(groups=("dh", "m1"))]:
            r, _ = run(C, pool_test_only=pt, **kw)
            T.rep(f"stack pool_test_only={pt} {kw}", r)
    if "--submit" in sys.argv:
        submit(C, sys.argv[sys.argv.index("--submit") + 1])
