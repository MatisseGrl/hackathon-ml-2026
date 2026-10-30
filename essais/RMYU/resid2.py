"""Modèle en 2 étages.
1) HGB sur les variables (sans label voisin) -> E[y] ; prédictions hors-fold (CV interne) pour le train.
2) Résidu r = y - E[y]. Pour chaque séance : moyenne des résidus des séances étiquetées du même user
   dans des fenêtres de temps (= « le user est-il en ce moment plus/moins essoufflé que prévu ? »).
3) HGB final sur variables + E[y] + voisins kNN + résidus temporels."""
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from knnlab import KnnLabel

YMAP = {"easy": 0, "moderate": 1, "hard": 2}
WINDOWS = (0, 1, 2, 4, 8, 16)


def _num(X):
    return np.column_stack([X.select_dtypes(exclude="category").values.astype(float),
                            X.select_dtypes("category").apply(lambda s: s.cat.codes).values])


def win_feats(Xq, Xl, rl, self_mask=False, prior=3.0):
    out = np.zeros((len(Xq), 2 * len(WINDOWS)))
    mq = (Xq.year * 12 + Xq.month).values; ml = (Xl.year * 12 + Xl.month).values
    uq, ul = Xq.user_id.values, Xl.user_id.values
    for u in np.unique(uq):
        iq = np.where(uq == u)[0]; il = np.where(ul == u)[0]
        if len(il) == 0:
            continue
        D = np.abs(mq[iq][:, None] - ml[il][None, :])
        if self_mask:
            D[np.arange(len(iq)), np.arange(len(iq))] = 10 ** 6  # iq == il quand Xq is Xl
        R = rl[il]
        for j, w in enumerate(WINDOWS):
            M = (D <= w)
            cnt = M.sum(1); s = M.astype(float) @ R
            out[iq, 2 * j] = cnt
            out[iq, 2 * j + 1] = s / (cnt + prior)
    return out


class Resid2(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.05, iters=300, inner=5, knn_cols=("resid_spd", "ldist", "t"), final_params=None):
        self.lr, self.iters, self.inner, self.knn_cols, self.final_params = lr, iters, inner, knn_cols, final_params

    def _base(self):
        return HistGradientBoostingClassifier(random_state=42, learning_rate=self.lr, max_iter=self.iters)

    def fit(self, X, y):
        X = X.reset_index(drop=True); yi = pd.Series(y).map(YMAP).values
        A = _num(X)
        E = np.zeros(len(X))
        for a, b in StratifiedKFold(self.inner, shuffle=True, random_state=0).split(A, yi):
            E[b] = clone(self._base()).fit(A[a], yi[a]).predict_proba(A[b]) @ [0, 1, 2]
        self.b_ = clone(self._base()).fit(A, yi)
        self.Xl_, self.rl_ = X, yi - E
        self.knn_ = KnnLabel(cols=self.knn_cols).fit(X, y)
        Z = np.column_stack([self.knn_._feats(X, loo=1), E, win_feats(X, X, self.rl_, self_mask=True)])
        self.m_ = HistGradientBoostingClassifier(random_state=42, **(self.final_params or dict(learning_rate=0.05, max_iter=300))).fit(Z, yi)
        self.classes_ = np.array(["easy", "moderate", "hard"])
        return self

    def predict_proba(self, X):
        X = X.reset_index(drop=True)
        E = self.b_.predict_proba(_num(X)) @ [0, 1, 2]
        Z = np.column_stack([self.knn_._feats(X, loo=0), E, win_feats(X, self.Xl_, self.rl_)])
        return self.m_.predict_proba(Z)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
