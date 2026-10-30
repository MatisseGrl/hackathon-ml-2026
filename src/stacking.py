"""Empilement en deux étages : on ajoute au modèle ce que disent les séances voisines de la même personne.

Idée : à vitesse égale, une séance est plus ou moins dure selon la forme du moment. On l'estime avec les séances
voisines du même utilisateur (même mois, même semaine, même heure, fenêtres de ±1 et ±3 mois).

1er étage : HistGradientBoosting sur UserStats. Il donne un score attendu s (0 = easy, 2 = hard).
Pour chaque séance du train : résidu = vrai niveau - s (« plus dure que prévu ? »), avec s calculé hors fold.
2e étage : HistGradientBoosting sur UserStats + s + moyennes des résidus et des niveaux des voisins.

Pas de fuite : `fit` ne voit que le train du fold. Une séance d'entraînement ne voit pas son propre label :
ses voisins viennent d'un découpage interne en 5. Une séance à prédire ne voit que les labels du train.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from src.features import UserStats

LEVEL = {"easy": 0, "moderate": 1, "hard": 2}
GROUPS = {
    "usym": ["user_id", "sport", "year", "month"],
    "usy": ["user_id", "sport", "year"],
    "uym": ["user_id", "year", "month"],
    "us": ["user_id", "sport"],
    "uymw": ["user_id", "year", "month", "weekday"],
    "usymw": ["user_id", "sport", "year", "month", "weekday"],
    "uymwh": ["user_id", "year", "month", "weekday", "start_hour_utc"],
    "usymwh": ["user_id", "sport", "year", "month", "weekday", "start_hour_utc"],
    "ust": ["user_id", "sport", "month_index"],
    "ut": ["user_id", "month_index"],
}
EXACT = ["usym", "usy", "uym", "us", "uymw", "usymw", "uymwh", "usymwh"]
WINDOWS = {"w1us": ("ust", 1), "w3us": ("ust", 3), "w1u": ("ut", 1), "w3u": ("ut", 3)}   # nom : (groupe, ± mois)


def _with_month_index(X):
    d = X.reset_index(drop=True).copy()
    d["month_index"] = d["year"] * 12 + d["month"]
    return d


def _keys(d, cols):
    return pd.Series(list(map(tuple, d[cols].to_numpy())))


def _tables(d, values):
    """Somme et effectif de `values` par groupe de voisins."""
    return {g: pd.DataFrame({"k": _keys(d, GROUPS[g]), "v": values}).groupby("k")["v"].agg(["sum", "count"])
            for g in EXACT + sorted({g for g, _ in WINDOWS.values()})}


def _encode(d, tables, prefix):
    """Moyenne et nombre de voisins de chaque séance, d'après des tables calculées sur d'autres séances."""
    out = {}
    for g in EXACT:
        k = _keys(d, GROUPS[g])
        total, count = tables[g]["sum"].reindex(k).to_numpy(), tables[g]["count"].reindex(k).fillna(0).to_numpy()
        out[f"{prefix}_{g}"] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        out[f"{prefix}_{g}_n"] = count
    for name, (g, width) in WINDOWS.items():
        total, count = np.zeros(len(d)), np.zeros(len(d))
        for delta in range(-width, width + 1):
            shifted = d[GROUPS[g]].copy()
            shifted["month_index"] += delta
            k = _keys(shifted, GROUPS[g])
            total += tables[g]["sum"].reindex(k).fillna(0).to_numpy()
            count += tables[g]["count"].reindex(k).fillna(0).to_numpy()
        out[f"{prefix}_{name}"] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        out[f"{prefix}_{name}_n"] = count
    return pd.DataFrame(out)


def _neighbors(d, label_tables, residual_tables):
    return pd.concat([_encode(d, residual_tables, "res"), _encode(d, label_tables, "lab")], axis=1)


def _expected_level(model, U):
    proba = model.predict_proba(U)
    classes = list(model.classes_)
    return proba[:, classes.index("moderate")] + 2 * proba[:, classes.index("hard")]


class NeighborStack(BaseEstimator):
    def __init__(self, max_iter=300, learning_rate=0.05, random_state=42):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _hgb(self):
        return HistGradientBoostingClassifier(max_iter=self.max_iter, learning_rate=self.learning_rate,
                                              random_state=self.random_state)

    def fit(self, X, y):
        d = _with_month_index(X)
        y = pd.Series(y).reset_index(drop=True)
        level = y.map(LEVEL).to_numpy(float)
        self.user_stats_ = UserStats().fit(d)
        U = self.user_stats_.transform(d)

        # 1er étage : score attendu hors fold, pour que le résidu mesure une vraie surprise
        expected = np.zeros(len(d))
        for tr, va in KFold(5, shuffle=True, random_state=1).split(d):
            expected[va] = _expected_level(self._hgb().fit(U.iloc[tr], y.iloc[tr]), U.iloc[va])
        self.first_ = self._hgb().fit(U, y)
        residual = level - expected
        self.label_tables_, self.residual_tables_ = _tables(d, level), _tables(d, residual)

        # Voisins des séances d'entraînement : calculés sans leur propre label (découpage interne)
        frames = []
        for tr, va in KFold(5, shuffle=True, random_state=0).split(d):
            part = _neighbors(d.iloc[va], _tables(d.iloc[tr], level[tr]), _tables(d.iloc[tr], residual[tr]))
            part.index = va
            frames.append(part)
        T = pd.concat([U, pd.Series(expected, name="expected_level"), pd.concat(frames).sort_index()], axis=1)
        self.second_ = self._hgb().fit(T, y)
        self.classes_ = self.second_.classes_
        return self

    def _features(self, X):
        d = _with_month_index(X)
        U = self.user_stats_.transform(d)
        expected = pd.Series(_expected_level(self.first_, U), name="expected_level")
        return pd.concat([U, expected, _neighbors(d, self.label_tables_, self.residual_tables_)], axis=1)

    def predict_proba(self, X):
        return self.second_.predict_proba(self._features(X))

    def predict(self, X):
        return self.second_.predict(self._features(X))
