"""Modèle 3MAS : macro-F1 ≈ 0,60 en validation (src/validation.py).

Idée en 3 étapes (explicable sur papier) :
1. Variables relatives à la personne (src/features.py, UserStats) : rang, ratio et z-score de la séance
   par rapport aux autres séances du même utilisateur dans le même sport.
   Ici les habitudes sont calculées sur train + test réunis (sans la cible) : plus de séances = habitudes mieux estimées.
2. Résidus de vitesse : pour chaque utilisateur-sport, une régression linéaire (ridge) prédit log(vitesse)
   à partir de la durée, de la pente, de la saison et de l'année. Résidu > 0 = « plus rapide que d'habitude
   pour ce type de parcours », donc séance probablement plus dure.
3. Deux modèles d'arbres qui connaissent l'utilisateur (user_id), moyennés (vote « soft ») :
   - HistGradientBoosting avec user_id en variable catégorielle,
   - ExtraTrees avec user_id en one-hot.
   Connaître l'utilisateur permet aux arbres d'apprendre des règles propres à chaque personne.

Pas de fuite : aucune statistique n'utilise `intensity` ; tous les user_id du test sont dans le train.
Lancer depuis la racine du dépôt :  python essais/3MAS/modele_06.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from src.features import UserStats, clean
from src.validation import SEED, evaluate, load_data, make_submission

KEYS = ["user_id", "sport"]
TARGETS = ["speed_kmh", "max_speed_kmh", "speed_std_kmh"]


def design(d):
    """Variables explicatives de la régression « vitesse attendue » d'une séance."""
    return np.column_stack([
        np.ones(len(d)),
        np.log(d["duration_min"].clip(1)),
        np.log1p(d["climb_per_km"].clip(0, 200).fillna(0)),
        np.log1p(d["altitude_loss_m"].fillna(0) / d["distance_km"]),
        np.cos(2 * np.pi * d["month"] / 12),
        np.sin(2 * np.pi * d["month"] / 12),
        d["year"] - 2014,
    ])


class PersonFeatures(BaseEstimator, TransformerMixin):
    """UserStats + résidus de vitesse par utilisateur-sport + identifiant utilisateur.

    pool : séances (train + test, sans cible) sur lesquelles on apprend les habitudes. None = celles de fit.
    user : "cat" (un entier, pour HistGB) ou "onehot" (une colonne par utilisateur, pour ExtraTrees).
    """

    def __init__(self, pool=None, user="cat", ridge=3.0, min_count=15):
        self.pool = pool
        self.user = user
        self.ridge = ridge
        self.min_count = min_count

    def fit(self, X, y=None):
        ref = X if self.pool is None else self.pool
        self.stats_ = UserStats().fit(ref)
        self.users_ = sorted(ref["user_id"].unique())
        self.coef_ = {}
        for key, g in clean(ref).groupby(KEYS):
            if len(g) < self.min_count:
                continue
            A = design(g)
            L = self.ridge * np.eye(A.shape[1])
            L[0, 0] = 0                       # on ne pénalise pas la constante
            self.coef_[key] = {}
            for t in TARGETS:
                v = np.log(g[t].clip(0.1)).to_numpy()
                beta = np.linalg.solve(A.T @ A + L, A.T @ v)
                self.coef_[key][t] = (beta, max(np.std(v - A @ beta), 1e-3))
        return self

    def transform(self, X):
        F = self.stats_.transform(X)
        d = clean(X).reset_index(drop=True)
        res = {f"res_{t}": np.full(len(d), np.nan) for t in TARGETS}
        for key, idx in d.groupby(KEYS).indices.items():
            c = self.coef_.get(key)
            if c is None:
                continue
            A = design(d.iloc[idx])
            for t in TARGETS:
                beta, sd = c[t]
                res[f"res_{t}"][idx] = (np.log(d[t].iloc[idx].clip(0.1)).to_numpy() - A @ beta) / sd
        F = pd.concat([F, pd.DataFrame(res)], axis=1)
        code = d["user_id"].map({u: i for i, u in enumerate(self.users_)}).fillna(-1).astype(int)
        if self.user == "cat":
            F["user_cat"] = code.to_numpy()
        else:
            U = np.zeros((len(d), len(self.users_)))
            ok = code.to_numpy() >= 0
            U[np.flatnonzero(ok), code.to_numpy()[ok]] = 1
            F = pd.concat([F, pd.DataFrame(U, columns=[f"user_{u}" for u in self.users_])], axis=1)
        return F


def build_model(pool):
    hgb = make_pipeline(
        PersonFeatures(pool, user="cat"),
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=10,
                                       categorical_features=["user_cat"], random_state=SEED),
    )
    et = make_pipeline(
        PersonFeatures(pool, user="onehot"),
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features=0.5, n_jobs=-1, random_state=SEED),
    )
    return VotingClassifier([("hgb", hgb), ("et", et)], voting="soft")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    train, test, _ = load_data()
    X = train.drop(columns=["workout_id", "intensity"])
    y = train["intensity"]
    X_test = test.drop(columns=["workout_id"])
    pool = pd.concat([X, X_test], ignore_index=True)     # sans la cible : pas de fuite

    model = build_model(pool)
    res = evaluate(model, X, y)
    if "--submit" in sys.argv:                          # écrit le CSV, ne soumet rien sur Kaggle
        make_submission(model, X, y, X_test, test["workout_id"], Path(__file__).parent / "submission_3MAS.csv")
