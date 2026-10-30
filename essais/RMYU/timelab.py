"""Variables de voisinage temporel : label moyen des séances étiquetées du même user dans une fenêtre de mois.
Calculées uniquement avec les labels du train du fold (en excluant la ligne elle-même pendant le fit)."""
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from knnlab import KnnLabel

YMAP = {"easy": 0, "moderate": 1, "hard": 2}
WINDOWS = (0, 1, 3, 6, 12)


def time_feats(Xq, Xl, yl, loo, prior=5.0):
    """Pour chaque ligne de Xq : somme/compte des labels de Xl du même user à |Δmois| <= w."""
    out = np.zeros((len(Xq), 3 * len(WINDOWS)))
    mq = (Xq.year * 12 + Xq.month).values; ml = (Xl.year * 12 + Xl.month).values
    sz_l = Xl["speed_kmh_z_us"].fillna(0).values
    uq, ul = Xq.user_id.values, Xl.user_id.values
    for u in np.unique(uq):
        iq = np.where(uq == u)[0]; il = np.where(ul == u)[0]
        if len(il) == 0:
            continue
        D = np.abs(mq[iq][:, None] - ml[il][None, :])
        Y = yl[il]; S = sz_l[il]
        for j, w in enumerate(WINDOWS):
            M = (D <= w).astype(float)
            cnt = M.sum(1); sy = M @ Y; ss = M @ S
            if loo:  # la ligne elle-même est dans Xl à la même position
                cnt -= 1; sy -= yl[il][np.searchsorted(il, iq)] if False else 0
            out[iq, 3 * j] = cnt
            out[iq, 3 * j + 1] = sy
            out[iq, 3 * j + 2] = ss
    return out


class TimeLabel(BaseEstimator, ClassifierMixin):
    def __init__(self, base=None, prior=5.0, knn_cols=("resid_spd", "ldist", "t")):
        self.base, self.prior, self.knn_cols = base, prior, knn_cols

    def _final(self, X, T, K):
        feats = []
        for j in range(len(WINDOWS)):
            cnt, sy, ss = T[:, 3 * j], T[:, 3 * j + 1], T[:, 3 * j + 2]
            feats += [cnt, (sy + self.prior) / (cnt + self.prior),  # moyenne lissée vers 1
                      (ss) / (cnt + 1)]
        return np.column_stack([K] + feats)

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        yi = pd.Series(y).map(YMAP).values
        self.Xl_, self.yl_ = X, yi
        self.knn_ = KnnLabel(cols=self.knn_cols).fit(X, y)  # stocke les voisins
        K = self.knn_._feats(X, loo=1)
        T = time_feats(X, X, yi, loo=False)
        # leave-one-out : retirer la ligne elle-même (Δmois = 0 <= w pour toutes les fenêtres)
        sz = X["speed_kmh_z_us"].fillna(0).values
        for j in range(len(WINDOWS)):
            T[:, 3 * j] -= 1; T[:, 3 * j + 1] -= yi; T[:, 3 * j + 2] -= sz
        self.m_ = clone(self.base or HistGradientBoostingClassifier(random_state=42, learning_rate=0.05, max_iter=300))
        self.m_.fit(self._final(X, T, K), yi)
        self.classes_ = np.array(["easy", "moderate", "hard"])
        return self

    def predict_proba(self, X):
        X = X.reset_index(drop=True)
        K = self.knn_._feats(X, loo=0)
        T = time_feats(X, self.Xl_, self.yl_, loo=False)
        return self.m_.predict_proba(self._final(X, T, K))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
