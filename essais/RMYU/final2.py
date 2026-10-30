"""Modèle consolidé RMYU, version sans fuite.
Étage 1 : HGB sur les variables (aucun label voisin) -> niveau attendu E, hors-fold sur le train.
Contexte (tout ce qui utilise des labels) : pour une séance du train, calculé UNIQUEMENT à partir des séances
  des autres plis internes (découpage en K) ; pour une séance à prédire, à partir de tout le train.
  - k-NN étiquetés (même user/sport, espace vitesse résiduelle / distance / temps)
  - moyennes des labels et des résidus (y - E) dans des groupes exacts (même jour+heure, même mois…)
  - moyennes des résidus dans des fenêtres de ±k mois, et des k-NN de résidus
Étage 2 : HGB sur variables + E + contexte. Les quotas par user sont appliqués après (quota.py)."""
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from knnlab import KnnLabel
from resid2 import win_feats, _num
from resid3 import knn_resid
from final import group_feats

YMAP = {"easy": 0, "moderate": 1, "hard": 2}


class Final2(BaseEstimator, ClassifierMixin):
    def __init__(self, inner=5, lr=0.05, iters=300, final_iters=300, final_lr=0.05, leaf=20, l2=0.0, seed=42,
                 rcols=("speed_kmh_z_us", "distance_km_z_us", "gain_per_km_z_us", "t"),
                 kcols=("resid_spd", "ldist", "t"), use=("knn", "grp", "win", "knnr")):
        self.inner, self.lr, self.iters, self.final_iters, self.final_lr, self.leaf, self.l2, self.seed = \
            inner, lr, iters, final_iters, final_lr, leaf, l2, seed
        self.rcols, self.kcols, self.use = rcols, kcols, use

    def _h(self, it, lr):
        return HistGradientBoostingClassifier(random_state=self.seed, learning_rate=lr, max_iter=it,
                                              min_samples_leaf=self.leaf, l2_regularization=self.l2)

    def _ctx(self, Xq, Xl, yl, rl):
        parts = []
        if "knn" in self.use:
            k = KnnLabel(cols=self.kcols).fit(Xl, pd.Series(np.array(["easy", "moderate", "hard"])[yl]))
            parts.append(k._feats(Xq, loo=0)[:, -4:])          # seulement les 4 colonnes de voisinage
        V = np.column_stack([yl - 1.0, rl])
        if "grp" in self.use:
            parts.append(group_feats(Xq, Xl, V, loo=False))
        if "win" in self.use:
            parts.append(win_feats(Xq, Xl, rl))
        if "knnr" in self.use:
            parts.append(knn_resid(Xq, Xl, rl, self.rcols, self.scale_))
        return np.column_stack(parts)

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
        self.s2_ = self._h(self.final_iters, self.final_lr).fit(np.column_stack([A, E, C]), yi)
        self.classes_ = np.array(["easy", "moderate", "hard"])
        return self

    def predict_proba(self, X):
        X = X.reset_index(drop=True); A = _num(X)
        E = self.s1_.predict_proba(A) @ [0, 1, 2]
        return self.s2_.predict_proba(np.column_stack([A, E, self._ctx(X, self.Xl_, self.yl_, self.rl_)]))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
