"""Comme resid2, plus : résidus moyens des k plus proches voisins (même user, espace speed/dist/temps),
et un 2e passage où les résidus sont recalculés avec le modèle de l'étage 2 (hors-fold)."""
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from knnlab import KnnLabel
from resid2 import _num, win_feats, YMAP

KS = (5, 20)


def knn_resid(Xq, Xl, rl, cols, scale, self_mask=False):
    out = np.zeros((len(Xq), len(KS)))
    uq, ul = Xq.user_id.values, Xl.user_id.values
    Q = Xq[list(cols)].fillna(0).values / scale; L = Xl[list(cols)].fillna(0).values / scale
    for u in np.unique(uq):
        iq = np.where(uq == u)[0]; il = np.where(ul == u)[0]
        k = min(max(KS) + self_mask, len(il))
        if k <= self_mask:
            continue
        _, ind = NearestNeighbors(n_neighbors=k).fit(L[il]).kneighbors(Q[iq])
        ind = ind[:, int(self_mask):]
        for j, kk in enumerate(KS):
            out[iq, j] = rl[il][ind[:, :kk]].mean(1)
    return out


class Resid3(BaseEstimator, ClassifierMixin):
    def __init__(self, passes=2, inner=5, cols=("speed_kmh_z_us", "distance_km_z_us", "gain_per_km_z_us", "t"), final=None):
        self.passes, self.inner, self.cols, self.final = passes, inner, cols, final

    def _clf(self):
        return HistGradientBoostingClassifier(random_state=42, **(self.final or dict(learning_rate=0.05, max_iter=300)))

    def _ctx(self, Xq, Xl, rl, self_mask):
        return np.column_stack([win_feats(Xq, Xl, rl, self_mask=self_mask),
                                knn_resid(Xq, Xl, rl, self.cols, self.scale_, self_mask=self_mask)])

    def fit(self, X, y):
        X = X.reset_index(drop=True); yi = pd.Series(y).map(YMAP).values
        self.scale_ = X[list(self.cols)].std().values + 1e-6
        self.knn_ = KnnLabel(cols=("resid_spd", "ldist", "t")).fit(X, y)
        base = np.column_stack([self.knn_._feats(X, loo=1)])
        folds = list(StratifiedKFold(self.inner, shuffle=True, random_state=0).split(base, yi))
        self.Xl_ = X
        self.stages_ = []
        Z = base
        r = None
        for p in range(self.passes + 1):
            E = np.zeros(len(X))
            for a, b in folds:
                E[b] = clone(self._clf()).fit(Z[a], yi[a]).predict_proba(Z[b]) @ [0, 1, 2]
            m = clone(self._clf()).fit(Z, yi)
            self.stages_.append((m, r))
            r = yi - E
            Z = np.column_stack([base, E, self._ctx(X, X, r, True)])
        self.final_ = clone(self._clf()).fit(Z, yi)
        self.r_last_ = r
        self.classes_ = np.array(["easy", "moderate", "hard"])
        return self

    def predict_proba(self, X):
        X = X.reset_index(drop=True)
        base = self.knn_._feats(X, loo=0)
        Z = base
        for i, (m, _) in enumerate(self.stages_):
            E = m.predict_proba(Z) @ [0, 1, 2]
            r_next = self.stages_[i + 1][1] if i + 1 < len(self.stages_) else self.r_last_
            Z = np.column_stack([base, E, self._ctx(X, self.Xl_, r_next, False)])
        return self.final_.predict_proba(Z)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
