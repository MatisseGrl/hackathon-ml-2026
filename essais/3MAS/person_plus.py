"""PersonStackPlus (3MAS) : PersonStack de l'équipe + ce qui a marché dans modele_06.py.

On garde tout le 1er étage de PersonStack (UserStats, voisins, estimations par personne, jumeaux) et on ajoute
au 2e étage :
- `user_cat` : l'identifiant de la personne en variable catégorielle pour HistGB (règles propres à chacun) ;
- les résidus de vitesse de modele_06 (vitesse observée - vitesse attendue pour ce parcours, par utilisateur-sport),
  appris ici sur le train seulement (pas sur le test, comme l'équipe l'a décidé).
Pas de fuite : les blocs ajoutés n'utilisent pas `intensity`.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

from src.person_stack import PersonStack, _person
from src.stacking import LEVEL, _expected_level, _neighbors, _tables, _with_month_index
from src.features import UserStats
from modele_06 import PersonFeatures


class PersonStackPlus(PersonStack):
    def __init__(self, moderate_weight=1.1, random_state=42, user_cat=True, resid=True,
                 hgb_params=None, et_params=None, et_weight=0.5, n_seeds=1):
        super().__init__(moderate_weight, random_state)
        self.user_cat = user_cat
        self.resid = resid
        self.hgb_params = hgb_params
        self.et_params = et_params
        self.et_weight = et_weight
        self.n_seeds = n_seeds

    def _extra(self, X):
        X = X.reset_index(drop=True)
        cols = {}
        if self.resid:
            F = self.pf_.transform(X)
            cols.update({c: F[c].to_numpy() for c in F.columns if c.startswith("res_")})
        if self.user_cat:
            cols["user_cat"] = X["user_id"].map(self.users_).fillna(-1).astype(int).to_numpy()
        return pd.DataFrame(cols)

    def build(self, X, y):
        """1er étage (identique à PersonStack.fit) ; renvoie la table du 2e étage."""
        X = X.reset_index(drop=True)
        d = _with_month_index(X)
        y = pd.Series(y).reset_index(drop=True)
        level = y.map(LEVEL).to_numpy(float)
        self.user_stats_ = UserStats().fit(X)
        self.pf_ = PersonFeatures(user="cat").fit(X)
        self.users_ = {u: i for i, u in enumerate(sorted(X["user_id"].unique()))}
        U = self.user_stats_.transform(X)
        expected = np.zeros(len(X))
        for tr, va in KFold(5, shuffle=True, random_state=1).split(X):
            expected[va] = _expected_level(self._first_model().fit(U.iloc[tr], y.iloc[tr]), U.iloc[va])
        self.first_ = self._first_model().fit(U, y)
        residual = level - expected
        self.label_tables_, self.residual_tables_ = _tables(d, level), _tables(d, residual)
        self.ref_, self.level_ = X, level
        neighbors, person = [], []
        for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
            n = _neighbors(d.iloc[va], _tables(d.iloc[tr], level[tr]), _tables(d.iloc[tr], residual[tr]))
            n.index = va
            neighbors.append(n)
            p = _person(X.iloc[tr], level[tr], X.iloc[va])
            p.index = va
            person.append(p)
        T = pd.concat([U, pd.Series(expected, name="expected_level"), pd.concat(neighbors).sort_index(),
                       pd.concat(person).sort_index(), self._extra(X)], axis=1)
        self.columns_ = list(T.columns)
        return T, y

    def fit_second(self, T, y):
        """2e étage : n_seeds paires (HistGB, ExtraTrees) avec des graines différentes, moyennées."""
        hp = dict(max_iter=500, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=5.0)
        hp.update(self.hgb_params or {})
        ep = dict(n_estimators=400, min_samples_leaf=3, max_features=0.3)
        ep.update(self.et_params or {})
        cat = ["user_cat"] if "user_cat" in T.columns else None
        self.imputer_ = SimpleImputer(strategy="median").fit(T)
        Ti = self.imputer_.transform(T)
        self.pairs_ = []
        for k in range(self.n_seeds):
            seed = self.random_state + k
            hgb = HistGradientBoostingClassifier(**hp, categorical_features=cat, random_state=seed).fit(T, y)
            et = ExtraTreesClassifier(**ep, n_jobs=-1, random_state=seed).fit(Ti, y)
            self.pairs_.append((hgb, et))
        self.hgb_, self.trees_ = self.pairs_[0]
        self.classes_ = self.hgb_.classes_
        return self

    def fit(self, X, y):
        return self.fit_second(*self.build(X, y))

    def _features(self, X):
        X = X.reset_index(drop=True)
        U = self.user_stats_.transform(X)
        expected = pd.Series(_expected_level(self.first_, U), name="expected_level")
        neighbors = _neighbors(_with_month_index(X), self.label_tables_, self.residual_tables_)
        person = _person(self.ref_, self.level_, X)
        return pd.concat([U, expected, neighbors, person, self._extra(X)], axis=1)[self.columns_]

    def proba_from(self, T):
        w, Ti = self.et_weight, self.imputer_.transform(T)
        return np.mean([(1 - w) * h.predict_proba(T) + w * e.predict_proba(Ti) for h, e in self.pairs_], axis=0)

    def predict_proba(self, X):
        return self.proba_from(self._features(X))


BEST = dict(hgb_params=dict(max_iter=400, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=10.0),
            et_params=dict(n_estimators=800, min_samples_leaf=2, max_features=0.2), n_seeds=5)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from src.validation import evaluate, load_data
    train, test, _ = load_data()
    X, y = train.drop(columns=["workout_id", "intensity"]), train["intensity"]
    evaluate(PersonStackPlus(**BEST), X, y)
