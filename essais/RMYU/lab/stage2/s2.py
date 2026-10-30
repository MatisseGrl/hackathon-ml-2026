"""Étages 2 alternatifs sur les matrices dumpées (dump.py). Écrit cache_stage2_<nom>.npz (format cache.py) dans RMYU.
Usage (depuis RMYU) : OMP_NUM_THREADS=4 python -u lab/stage2/s2.py <nom> <variante> <seeds_dump ex. 42,7> [seeds_s2 ex. 42,43,44]"""
import sys, time; sys.path.append("../.."); sys.path.append(".")
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR


def gauss(s, sig=0.5):
    """score ordinal -> pseudo-probas (affectation sous quotas = tri du score)."""
    L = -(s[:, None] - np.arange(3)[None, :]) ** 2 / (2 * sig ** 2)
    P = np.exp(L - L.max(1, keepdims=True)); return P / P.sum(1, keepdims=True)


def model(v, s):
    if v == "hgb":      # = étage 2 de Final2 par défaut
        return HGB(random_state=s, learning_rate=0.05, max_iter=300, min_samples_leaf=20, l2_regularization=0.0)
    if v == "hgb_reg":  # piste réglage : lr .03, 500 it, leaf 40, l2 1
        return HGB(random_state=s, learning_rate=0.03, max_iter=500, min_samples_leaf=40, l2_regularization=1.0)
    if v == "hgb_noes":  # sans early stopping (tout le train), lent mais fixe
        return HGB(random_state=s, learning_rate=0.03, max_iter=350, min_samples_leaf=40, l2_regularization=1.0,
                   early_stopping=False)
    if v == "hgb_mf":   # sous-échantillonnage de variables (sklearn>=1.4)
        return HGB(random_state=s, learning_rate=0.03, max_iter=500, min_samples_leaf=40, l2_regularization=1.0,
                   max_features=0.5)
    if v == "et":
        return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=4, random_state=s)
    if v == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.2, n_jobs=4, random_state=s)
    raise ValueError(v)


def fitpred(v, s, Z, y, Q):
    if v == "hgbr":   # régression ordinale (y en 0/1/2)
        return gauss(HGBR(random_state=s, learning_rate=0.03, max_iter=500, min_samples_leaf=40,
                          l2_regularization=1.0).fit(Z, y).predict(Q))
    if v in ("et", "rf"):
        Z = np.nan_to_num(Z, nan=-999.0); Q = np.nan_to_num(Q, nan=-999.0)
    return model(v, s).fit(Z, y).predict_proba(Q)


if __name__ == "__main__":
    name, v = sys.argv[1], sys.argv[2]
    dseeds = [int(x) for x in sys.argv[3].split(",")]
    sseeds = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [42]
    D = [np.load(f"lab/stage2/dump_s{d}.npz") for d in dseeds]
    out = {}; per = {d: {} for d in dseeds}; t0 = time.time()
    for f in list(range(5)) + ["F"]:
        kz, ky, kq = (f"Ztr{f}", f"ytr{f}", f"Zq{f}") if f != "F" else ("ZtrF", "yF", "ZqF")
        Ps = {ds: np.mean([fitpred(v, s, d[kz], d[ky], d[kq]) for s in sseeds], axis=0) for ds, d in zip(dseeds, D)}
        P = np.mean(list(Ps.values()), axis=0)
        for tgt, PP in [(out, P)] + [(per[ds], Ps[ds]) for ds in dseeds]:
            if f == "F":
                tgt["Pfull"] = PP
            else:
                tgt[f"P{f}"] = PP; tgt[f"va{f}"] = D[0][f"va{f}"]
        print(f"{name} fold {f} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(f"cache_stage2_{name}.npz", **out)
    if len(dseeds) > 1:
        for ds in dseeds:
            np.savez(f"cache_stage2_{name}_d{ds}.npz", **per[ds])
    sys.path.append("lab/stage2")
    from ev import ev
    for n in [name] + ([f"{name}_d{ds}" for ds in dseeds] if len(dseeds) > 1 else []):
        r = ev(np.load(f"cache_stage2_{n}.npz"))
        print("EVAL", n, " | ".join(f"{k} {m:.4f}±{s:.4f}" for k, (m, s, _) in r.items()), flush=True)
