"""Modèle consolidé RMYU.
Étage 1 : HGB sur les variables relatives (+ voisins k-NN étiquetés) -> niveau attendu E (hors-fold sur le train).
Contexte : pour chaque séance, moyenne des labels et des résidus (y - E) des séances étiquetées du même user
           dans des groupes exacts (même jour/heure, même mois…) et des fenêtres de ±k mois, + k-NN de résidus.
           Sur le train : leave-one-out (la séance ne voit jamais son propre label).
Étage 2 : HGB sur tout ça. Puis (hors classe) affectation sous quotas par user (quota.py)."""
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from knnlab import KnnLabel
from resid2 import win_feats
from resid3 import knn_resid

YMAP = {"easy": 0, "moderate": 1, "hard": 2}
GROUPS = [["user_id", "ym", "weekday", "start_hour_utc"], ["user_id", "ym", "weekday"], ["user_id", "ym"],
          ["user_id", "sport", "ym"], ["user_id", "year"], ["user_id", "ym", "start_hour_utc"],
          ["user_id", "year", "weekday", "start_hour_utc"]]


def _key(X, cols):
    d = X.assign(ym=X.year * 12 + X.month)
    k = d[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + "|" + d[c].astype(str)
    return k.values


def group_feats(Xq, Xl, vals, loo, prior=2.0):
    """vals: matrice (n_l, m) ; renvoie compte + moyenne lissée par groupe."""
    out = []
    for cols in GROUPS:
        kq, kl = _key(Xq, cols), _key(Xl, cols)
        S = pd.DataFrame(vals, index=kl).groupby(level=0).sum()
        C = pd.Series(1, index=kl).groupby(level=0).sum()
        s = S.reindex(kq).fillna(0).values; c = C.reindex(kq).fillna(0).values
        if loo:
            s = s - vals; c = c - 1
        out += [c[:, None], s / (c[:, None] + prior)]
    return np.column_stack(out)


class Final(BaseEstimator, ClassifierMixin):
    def __init__(self, inner=5, lr=0.05, iters=300, final_iters=400, leaf=20, l2=0.0, seed=42,
                 rcols=("speed_kmh_z_us", "distance_km_z_us", "gain_per_km_z_us", "t")):
        self.inner, self.lr, self.iters, self.final_iters, self.leaf, self.l2, self.seed, self.rcols = \
            inner, lr, iters, final_iters, leaf, l2, seed, rcols

    def _h(self, it):
        return HistGradientBoostingClassifier(random_state=self.seed, learning_rate=self.lr, max_iter=it,
                                              min_samples_leaf=self.leaf, l2_regularization=self.l2)

    def _ctx(self, Xq, loo):
        V = np.column_stack([self.yl_ - 1.0, self.rl_])  # label centré et résidu
        return np.column_stack([group_feats(Xq, self.Xl_, V, loo),
                                win_feats(Xq, self.Xl_, self.rl_, self_mask=loo),
                                knn_resid(Xq, self.Xl_, self.rl_, self.rcols, self.scale_, self_mask=loo)])

    def fit(self, X, y):
        X = X.reset_index(drop=True); yi = pd.Series(y).map(YMAP).values
        self.knn_ = KnnLabel(cols=("resid_spd", "ldist", "t")).fit(X, y)
        B = self.knn_._feats(X, loo=1)
        E = np.zeros(len(X))
        for a, b in StratifiedKFold(self.inner, shuffle=True, random_state=self.seed).split(B, yi):
            E[b] = self._h(self.iters).fit(B[a], yi[a]).predict_proba(B[b]) @ [0, 1, 2]
        self.s1_ = self._h(self.iters).fit(B, yi)
        self.Xl_, self.yl_, self.rl_ = X, yi.astype(float), yi - E
        self.scale_ = X[list(self.rcols)].std().values + 1e-6
        Z = np.column_stack([B, E, self._ctx(X, loo=True)])
        self.s2_ = self._h(self.final_iters).fit(Z, yi)
        self.classes_ = np.array(["easy", "moderate", "hard"])
        return self

    def predict_proba(self, X):
        X = X.reset_index(drop=True)
        B = self.knn_._feats(X, loo=0)
        E = self.s1_.predict_proba(B) @ [0, 1, 2]
        return self.s2_.predict_proba(np.column_stack([B, E, self._ctx(X, loo=False)]))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
