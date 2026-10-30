"""Classifieur en 2 étages : variables de voisinage (labels des séances similaires du même user/sport) + HGB.
Les labels voisins viennent UNIQUEMENT des lignes d'entraînement du fold (leave-one-out sur le train)."""
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors

YMAP = {"easy": 0, "moderate": 1, "hard": 2}


class KnnLabel(BaseEstimator, ClassifierMixin):
    def __init__(self, cols=("speed_kmh_z_us", "distance_km_z_us", "gain_per_km_z_us", "t"), k=(5, 15), base=None):
        self.cols, self.k, self.base = cols, k, base

    def _feats(self, X, loo):
        out = np.full((len(X), 2 * len(self.k)), np.nan)
        keys = list(zip(X.user_id, X.sport))
        for key, rows in pd.Series(range(len(X))).groupby(pd.Series(keys)):
            if key not in self.store_:
                continue
            Z, yv = self.store_[key]
            Q = X.iloc[rows.values][list(self.cols)].fillna(0).values / self.scale_
            kmax = min(max(self.k) + loo, len(Z))
            if kmax <= loo:
                continue
            _, ind = NearestNeighbors(n_neighbors=kmax).fit(Z).kneighbors(Q)
            ind = ind[:, loo:]
            for j, k in enumerate(self.k):
                lab = yv[ind[:, :k]]
                out[rows.values, 2 * j] = lab.mean(1)
                out[rows.values, 2 * j + 1] = (lab == 2).mean(1) - (lab == 0).mean(1)
        return np.column_stack([X.select_dtypes(exclude="category").values.astype(float), X.select_dtypes("category").apply(lambda s: s.cat.codes).values, out])

    def fit(self, X, y):
        y = pd.Series(y).map(YMAP).values
        self.scale_ = X[list(self.cols)].std().values + 1e-6
        self.store_ = {}
        for key, rows in pd.Series(range(len(X))).groupby(pd.Series(list(zip(X.user_id, X.sport)))):
            self.store_[key] = (X.iloc[rows.values][list(self.cols)].fillna(0).values / self.scale_, y[rows.values])
        self.m_ = clone(self.base or HistGradientBoostingClassifier(random_state=42, learning_rate=0.05, max_iter=300))
        self.m_.fit(self._feats(X, loo=1), y)
        self.classes_ = np.array(["easy", "moderate", "hard"])
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(self._feats(X, loo=0))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
