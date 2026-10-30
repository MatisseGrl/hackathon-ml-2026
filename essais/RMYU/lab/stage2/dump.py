"""Dump des matrices de l'étage 2 de Final2 (A, E, C) par pli, pour tester vite beaucoup d'étages 2.
Usage (depuis RMYU) : OMP_NUM_THREADS=2 python -u lab/stage2/dump.py <seed> [kwargs json]
-> lab/stage2/dump_s<seed>.npz : Ztr{f}, ytr{f}, Zq{f} (val du pli puis test), va{f} ; ZtrF, yF, ZqF (tout le train -> test)."""
import sys, json, time; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.validation import load_data, get_folds
from feats import build, add_resid
from final2 import Final2, YMAP
from resid2 import _num


class F2Dump(Final2):
    def fit(self, X, y):
        X = X.reset_index(drop=True); yi = pd.Series(y).map(YMAP).values
        self.scale_ = X[list(self.rcols)].std().values + 1e-6
        A = _num(X)
        folds = list(StratifiedKFold(self.inner, shuffle=True, random_state=self.seed).split(A, yi))
        E = np.zeros(len(X))
        for a, b in folds:
            E[b] = self._h(self.iters, self.lr).fit(A[a], yi[a]).predict_proba(A[b]) @ [0, 1, 2]
        self.s1_ = self._h(self.iters, self.lr).fit(A, yi)
        r = yi - E
        C = None
        for a, b in folds:
            cb = self._ctx(X.iloc[b], X.iloc[a].reset_index(drop=True), yi[a], r[a])
            if C is None:
                C = np.zeros((len(X), cb.shape[1]))
            C[b] = cb
        self.Xl_, self.yl_, self.rl_ = X, yi, r
        self.Ztr_ = np.column_stack([A, E, C]).astype(np.float32); self.ytr_ = yi
        return self

    def qfeats(self, X):
        X = X.reset_index(drop=True); A = _num(X)
        E = self.s1_.predict_proba(A) @ [0, 1, 2]
        return np.column_stack([A, E, self._ctx(X, self.Xl_, self.yl_, self.rl_)]).astype(np.float32)


if __name__ == "__main__":
    seed = int(sys.argv[1]); kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    train, test, _ = load_data()
    F = add_resid(build(train, test))
    X = F.drop(columns=["part", "workout_id"]).reset_index(drop=True)
    ntr = (F.part == "tr").sum(); Xtr, Xte = X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True)
    y = train.intensity.reset_index(drop=True)
    out = {}; t0 = time.time()
    for f, (tr_i, va_i) in enumerate(get_folds(y).split(Xtr, y)):
        m = F2Dump(seed=seed, **kw).fit(Xtr.iloc[tr_i], y.iloc[tr_i])
        out[f"Ztr{f}"], out[f"ytr{f}"] = m.Ztr_, m.ytr_
        out[f"Zq{f}"] = m.qfeats(pd.concat([Xtr.iloc[va_i], Xte]).reset_index(drop=True)); out[f"va{f}"] = va_i
        print(f"fold {f} done ({time.time()-t0:.0f}s)", flush=True)
    m = F2Dump(seed=seed, **kw).fit(Xtr, y)
    out["ZtrF"], out["yF"], out["ZqF"] = m.Ztr_, m.ytr_, m.qfeats(Xte)
    tag = kw.get("tag", "") if False else ""
    np.savez(f"lab/stage2/dump_s{seed}.npz", **out); print("saved", seed, f"({time.time()-t0:.0f}s)", flush=True)
