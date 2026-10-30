"""Empilement « par personne » : NeighborStack + ce que disent les séances semblables de la même personne.

Trois blocs s'ajoutent aux variables de NeighborStack :
- estimations locales : moyenne des niveaux des séances du même utilisateur, pondérée par la ressemblance et la proximité
  dans le temps (noyaux), et voisins les plus proches ;
- ridge par groupe : chaque couple utilisateur + sport a sa propre pente, car la vitesse ne compte pas pareil pour tous ;
- jumeaux : niveau moyen des séances du train strictement identiques (même utilisateur, mêmes mesures).

2e étage : moyenne des probabilités d'un HistGradientBoosting et d'un ExtraTrees, avec un poids de 1,1 sur `moderate`,
la classe la plus mal prédite.

Pas de fuite : comme dans NeighborStack, une séance d'entraînement ne voit pas son propre label (découpage interne en 5)
et une séance à prédire ne voit que les labels du train.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

from src.features import UserStats, clean
from src.stacking import LEVEL, _expected_level, _neighbors, _tables, _with_month_index

SHAPE = ["speed_kmh", "duration_min", "distance_km", "altitude_gain_m", "max_speed_kmh", "speed_std_kmh"]
ROUTE = SHAPE + ["altitude_loss_m", "altitude_mean_m"]     # l'altitude aide à reconnaître un même parcours
LOG_COLS = ["duration_min", "distance_km", "altitude_gain_m", "altitude_loss_m"]


def _prepare(X, cols):
    d = clean(X.reset_index(drop=True))
    d["t"] = d["year"] * 12 + d["month"] + 0.0
    for c in LOG_COLS:
        if c in cols:
            d[c] = np.log1p(d[c])
    return d


def _groups(R, Q, keys, cols):
    """Pour chaque groupe : valeurs standardisées (sur ref), écarts de temps et distances entre requêtes et ref."""
    ref_groups = R.groupby(keys).indices
    for key, qi in Q.groupby(keys).indices.items():
        ri = ref_groups.get(key)
        if ri is None or len(ri) < 3:
            continue
        A, B = R[cols].to_numpy()[ri], Q[cols].to_numpy()[qi]
        mean, std = np.nanmean(A, 0), np.nanstd(A, 0) + 1e-6
        A, B = np.nan_to_num((A - mean) / std), np.nan_to_num((B - mean) / std)
        ta, tb = R["t"].to_numpy()[ri], Q["t"].to_numpy()[qi]
        dt = np.abs(tb[:, None] - ta[None, :])
        dx = ((B[:, None, :] - A[None, :, :]) ** 2).sum(2)
        yield qi, ri, A, B, ta, tb, dt, dx


def _local(ref, level, query):
    """Moyennes à noyaux et ridge par couple utilisateur + sport."""
    R, Q = _prepare(ref, SHAPE), _prepare(query, SHAPE)
    out = {}

    def put(name, idx, values):
        out.setdefault(name, np.full(len(Q), np.nan))[idx] = values

    for qi, ri, A, B, ta, tb, dt, dx in _groups(R, Q, ["user_id", "sport"], SHAPE):
        L = level[ri]
        for h in (1, 3, 12):
            w = np.exp(-dt / h) + 1e-9
            put(f"t{h}", qi, (w * L).sum(1) / w.sum(1))
        for bw in (0.5, 1.5):
            w = np.exp(-dx / (2 * bw ** 2)) + 1e-9
            put(f"x{bw}", qi, (w * L).sum(1) / w.sum(1))
            w2 = w * np.exp(-dt / 6) + 1e-12
            put(f"xt{bw}", qi, (w2 * L).sum(1) / w2.sum(1))
        T, Tb = (ta - ta.mean()) / (ta.std() + 1e-6), (tb - ta.mean()) / (ta.std() + 1e-6)
        F, G = np.c_[A, T, T ** 2], np.c_[B, Tb, Tb ** 2]
        for alpha in (3.0, 30.0):
            beta = np.linalg.solve(F.T @ F + alpha * np.eye(F.shape[1]), F.T @ (L - L.mean()))
            put(f"ridge{int(alpha)}", qi, L.mean() + G @ beta)
        beta = np.linalg.solve(A.T @ A + 10 * np.eye(A.shape[1]), A.T @ (L - L.mean()))
        put("slope_speed", qi, np.full(len(qi), beta[0]))
        put("group_n", qi, np.full(len(qi), len(ri)))
    return pd.DataFrame(out, index=range(len(Q)))


def _close(ref, level, query):
    """Voisins serrés (même parcours, même période), par couple utilisateur + sport puis par utilisateur."""
    R, Q = _prepare(ref, ROUTE), _prepare(query, ROUTE)
    out = {}

    def put(name, idx, values):
        out.setdefault(name, np.full(len(Q), np.nan))[idx] = values

    for keys, tag in ((["user_id", "sport"], "us"), (["user_id"], "u")):
        for qi, ri, A, B, ta, tb, dt, dx in _groups(R, Q, keys, ROUTE):
            L = level[ri]
            for bw in (0.15, 0.3, 0.6):
                for h in (1, 3, 1e6):
                    w = np.exp(-dx / (2 * bw * bw)) * np.exp(-dt / h) + 1e-12
                    put(f"{tag}_b{bw}_h{h:g}", qi, (w * L).sum(1) / w.sum(1))
                    if bw == 0.3:
                        put(f"{tag}_w{h:g}", qi, np.log(w.sum(1)))
            order = np.argsort(dx + (dt / 6) ** 2, axis=1)
            for k in (1, 3, 7):
                put(f"{tag}_nn{k}", qi, L[order[:, :min(k, len(ri))]].mean(1))
            put(f"{tag}_nnd", qi, np.sqrt(np.take_along_axis(dx, order[:, :1], 1)[:, 0]))
    return pd.DataFrame(out, index=range(len(Q)))


def _twins(ref, level, query):
    """Niveau moyen et nombre de séances du train strictement identiques."""
    def keys(d):
        return list(map(tuple, d.reset_index(drop=True).to_numpy()))
    table = pd.DataFrame({"k": keys(ref), "v": level}).groupby("k")["v"].agg(["mean", "count"])
    k = pd.Series(keys(query[ref.columns]))
    return pd.DataFrame({"twin": table["mean"].reindex(k).to_numpy(), "twin_n": table["count"].reindex(k).fillna(0).to_numpy()})


def _person(ref, level, query):
    return pd.concat([_local(ref, level, query), _twins(ref, level, query), _close(ref, level, query)], axis=1)


class PersonStack(BaseEstimator):
    def __init__(self, moderate_weight=1.1, use_trees=True, random_state=42):
        self.moderate_weight = moderate_weight
        self.use_trees = use_trees      # False : HGB seul, plus simple à expliquer
        self.random_state = random_state

    def _first_model(self):
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=self.random_state)

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        d = _with_month_index(X)
        y = pd.Series(y).reset_index(drop=True)
        level = y.map(LEVEL).to_numpy(float)
        self.user_stats_ = UserStats().fit(X)
        U = self.user_stats_.transform(X)

        # 1er étage : score attendu hors fold, pour que le résidu mesure une vraie surprise
        expected = np.zeros(len(X))
        for tr, va in KFold(5, shuffle=True, random_state=1).split(X):
            expected[va] = _expected_level(self._first_model().fit(U.iloc[tr], y.iloc[tr]), U.iloc[va])
        self.first_ = self._first_model().fit(U, y)
        residual = level - expected
        self.label_tables_, self.residual_tables_ = _tables(d, level), _tables(d, residual)
        self.ref_, self.level_ = X, level

        # Voisins des séances d'entraînement : calculés sans leur propre label (découpage interne)
        neighbors, person = [], []
        for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
            n = _neighbors(d.iloc[va], _tables(d.iloc[tr], level[tr]), _tables(d.iloc[tr], residual[tr]))
            n.index = va
            neighbors.append(n)
            p = _person(X.iloc[tr], level[tr], X.iloc[va])
            p.index = va
            person.append(p)
        T = pd.concat([U, pd.Series(expected, name="expected_level"), pd.concat(neighbors).sort_index(),
                       pd.concat(person).sort_index()], axis=1)

        # 2e étage : deux familles d'ensembles, dont on moyennera les probabilités
        self.columns_ = list(T.columns)
        self.hgb_ = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40,
                                                   l2_regularization=5.0, random_state=self.random_state).fit(T, y)
        if self.use_trees:
            self.imputer_ = SimpleImputer(strategy="median").fit(T)
            self.trees_ = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=-1,
                                               random_state=self.random_state).fit(self.imputer_.transform(T), y)
        self.classes_ = self.hgb_.classes_
        return self

    def _features(self, X):
        X = X.reset_index(drop=True)
        U = self.user_stats_.transform(X)
        expected = pd.Series(_expected_level(self.first_, U), name="expected_level")
        neighbors = _neighbors(_with_month_index(X), self.label_tables_, self.residual_tables_)
        person = _person(self.ref_, self.level_, X)
        return pd.concat([U, expected, neighbors, person], axis=1)[self.columns_]

    def predict_proba(self, X):
        T = self._features(X)
        if not self.use_trees:
            return self.hgb_.predict_proba(T)
        return (self.hgb_.predict_proba(T) + self.trees_.predict_proba(self.imputer_.transform(T))) / 2

    def predict(self, X):
        weights = np.where(self.classes_ == "moderate", self.moderate_weight, 1.0)
        return self.classes_[(self.predict_proba(X) * weights).argmax(1)]
