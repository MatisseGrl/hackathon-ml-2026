"""Lissage transductif des probas entre séances NON étiquetées (val du pli + test) d'un même user,
puis doublons (post.adjust) + quotas (quota.constrained_predict).

Pour chaque séance non étiquetée i et chaque groupe g (ex. même user+année+mois+jour de semaine+heure) :
  m_g(i) = moyenne des probas Q des AUTRES séances non étiquetées du groupe (n_g voisins)
  Q'(i)  = (P(i) + sum_g lam_g * w_g(i) * m_g(i)) / (1 + sum_g lam_g * w_g(i)),  w_g = n_g / (n_g + alpha)
Optionnel : idem avec les labels one-hot des séances étiquetées du groupe (lamL), itérations (propagation).
Usage : python -u trans.py <cache.npz> [--grid] [--submit out.csv]
"""
import sys, os, itertools, time
HERE = os.path.dirname(os.path.abspath(__file__))
RM = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [RM, os.path.join(RM, "..", "..")]
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from src.validation import load_data
from quota import constrained_predict, LAB
from post import adjust, keyof, FULL, PART

train, test, sample = load_data()
ntr, nte = len(train), len(test)
ALL = pd.concat([train.drop(columns="intensity"), test], ignore_index=True)
yi = train.intensity.map({"easy": 0, "moderate": 1, "hard": 2}).values
ym = (ALL.year * 12 + ALL.month).values.astype(np.int64)
u = ALL.user_id.values.astype(np.int64)


def fac(cols):
    return pd.MultiIndex.from_frame(ALL[cols]).factorize()[0].astype(np.int64) if len(cols) > 1 else \
        pd.factorize(ALL[cols[0]])[0].astype(np.int64)


# groupes : (clé entière par ligne de ALL, décalages autorisés)
KEYS = {
    "dh": (fac(["user_id", "year", "month", "weekday", "start_hour_utc"]), (0,)),
    "d": (fac(["user_id", "year", "month", "weekday"]), (0,)),
    "ms": (fac(["user_id", "sport", "year", "month"]), (0,)),
    "m": (u * 100000 + ym, (0,)),
    "m1": (u * 100000 + ym, (-1, 0, 1)),
}
kf_all, kp_all = keyof(ALL, FULL), keyof(ALL, PART)


def nb(keys_q, keys_p, V, offsets, self_mask=None):
    """sommes/compte de V (lignes du pool, clés keys_p) pour chaque requête (clés keys_q), décalages offsets.
    self_mask : booléen (len q) -> la requête est elle-même dans le pool à la même position (on la retire)."""
    df = pd.DataFrame(V); df["k"] = keys_p
    g = df.groupby("k"); S = g.sum(); C = g.size().reindex(S.index).values.astype(float)
    s = np.zeros((len(keys_q), V.shape[1])); c = np.zeros(len(keys_q))
    for o in offsets:
        idx = S.index.get_indexer(keys_q + o); m = idx >= 0
        s[m] += S.values[idx[m]]; c[m] += C[idx[m]]
    if self_mask is not None:
        s[self_mask] -= V[self_mask]; c[self_mask] -= 1
    return s, c


def smooth(P, gidx_u, gidx_l, y_l, groups, lam, alpha=1.0, lamL=0.0, iters=1, pool_mask=None, mode="lin", V=None, pseudo=0.0):
    """P : probas des lignes non étiquetées (indices globaux gidx_u dans ALL) ; pool_mask : lesquelles servent de voisins."""
    if pool_mask is None:
        pool_mask = np.ones(len(gidx_u), bool)
    Q = P.copy() if V is None else V
    for _ in range(iters):
        num = np.log(np.clip(P, 1e-6, 1)) if mode == "log" else P.copy()
        den = np.ones(len(P))
        for gname in groups:
            K, offs = KEYS[gname]
            ku = K[gidx_u]
            s, c = nb(ku, ku[pool_mask], Q[pool_mask], offs, None)
            s[pool_mask] -= Q[pool_mask]; c[pool_mask] -= 1
            mean = s / np.maximum(c, 1)[:, None]
            w = lam * c / (c + alpha)
            if mode == "log":
                num += w[:, None] * np.log(np.clip(mean, 1e-3, 1)) * (c > 0)[:, None]
            else:
                num += w[:, None] * mean
                den += w
            if lamL > 0:
                kl = K[gidx_l]
                sl, cl = nb(ku, kl, np.eye(3)[y_l], offs)
                wl = lamL * cl / (cl + alpha)
                ml = sl / np.maximum(cl, 1)[:, None]
                if mode == "log":
                    num += wl[:, None] * np.log(np.clip(ml, 1e-3, 1)) * (cl > 0)[:, None]
                else:
                    num += wl[:, None] * ml; den += wl
        if mode == "log":
            Z = np.exp(num - num.max(1, keepdims=True)); Q = Z / Z.sum(1, keepdims=True)
        else:
            Q = num / den[:, None]
    return Q


def two_pass(Q0, gidx_u, gidx_l, y_l, cfg, pool_mask=None):
    """pseudo>0 : 1re passe quotas -> pseudo-labels one-hot ; les voisins portent (1-pseudo)*Q0 + pseudo*onehot."""
    cfg = dict(cfg); beta = cfg.pop("pseudo", 0.0)
    if beta > 0:
        pr = constrained_predict(Q0, u[gidx_u], u[gidx_l], y_l)
        cfg["V"] = (1 - beta) * Q0 + beta * np.eye(3)[pr]
    return smooth(Q0, gidx_u, gidx_l, y_l, pool_mask=pool_mask, **cfg)


def evaluate(C, cfg, folds=range(5), dup=dict(w4=0.5), test_only_pool=False):
    out = []
    for f in folds:
        va = C[f"va{f}"]; P = C[f"P{f}"]; nv = len(va)
        lab = np.setdiff1d(np.arange(ntr), va)
        gidx_u = np.concatenate([va, ntr + np.arange(nte)])
        ku_f, ku_p = kf_all[gidx_u], kp_all[gidx_u]
        Q0 = adjust(P, ku_f, ku_p, kf_all[lab], kp_all[lab], yi[lab], **dup) if dup is not None else P
        pool = np.r_[np.zeros(nv, bool), np.ones(nte, bool)] if test_only_pool else None
        Q = two_pass(Q0, gidx_u, lab, yi[lab], cfg, pool) if cfg else Q0
        if dup is not None:  # re-imposer les doublons exacts étiquetés
            Q = adjust(Q, ku_f, ku_p, kf_all[lab], kp_all[lab], yi[lab], w4=0, wfull=1.0)
        gu = u[gidx_u]
        pred = constrained_predict(Q, gu, u[lab], yi[lab])
        out.append((f1_score(yi[va], pred[:nv], average="macro"), f1_score(yi[va], Q[:nv].argmax(1), average="macro")))
    return np.array(out)


def submit(C, cfg, path, dup=dict(w4=0.5)):
    gidx_u = ntr + np.arange(nte); lab = np.arange(ntr)
    ku_f, ku_p = kf_all[gidx_u], kp_all[gidx_u]
    Q0 = adjust(C["Pfull"], ku_f, ku_p, kf_all[lab], kp_all[lab], yi, **dup)
    Q = two_pass(Q0, gidx_u, lab, yi, cfg) if cfg else Q0
    Q = adjust(Q, ku_f, ku_p, kf_all[lab], kp_all[lab], yi, w4=0, wfull=1.0)
    pred = constrained_predict(Q, u[gidx_u], u[lab], yi)
    sub = pd.DataFrame({"workout_id": test.workout_id.values, "intensity": LAB[pred]})
    assert len(sub) == len(sample) and (sub.workout_id.values == sample.workout_id.values).all()
    sub.to_csv(path, index=False); print("écrit", path, sub.intensity.value_counts().to_dict(), flush=True)
    np.save(path.replace(".csv", "_Q.npy"), Q)


def rep(name, r):
    print(f"{name:55s} quota {r[:,0].mean():.4f} ± {r[:,0].std():.4f}  argmax {r[:,1].mean():.4f}  "
          f"folds {np.round(r[:,0],4).tolist()}", flush=True)
    return r[:, 0].mean()


if __name__ == "__main__":
    C = dict(np.load(sys.argv[1]))
    t0 = time.time()
    if "--grid" in sys.argv:
        rep("ref dup(w4=.5)+quota", evaluate(C, None))
        rep("ref quota seul", evaluate(C, None, dup=None))
        res = {}
        for groups in [("dh",), ("d",), ("ms",), ("m",), ("m1",), ("dh", "m"), ("d", "m1")]:
            for lam in [0.25, 0.5, 1.0]:
                cfg = dict(groups=groups, lam=lam, alpha=1.0)
                res[(groups, lam)] = rep(str(cfg), evaluate(C, cfg))
        print(f"({time.time()-t0:.0f}s)")
    if "--submit" in sys.argv:
        import json
        cfg = json.loads(sys.argv[sys.argv.index("--cfg") + 1]) if "--cfg" in sys.argv else None
        if cfg and "groups" in cfg: cfg["groups"] = tuple(cfg["groups"])
        submit(C, cfg, sys.argv[sys.argv.index("--submit") + 1])
