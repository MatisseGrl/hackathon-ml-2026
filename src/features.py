"""Variables « relatives à la personne » : on compare chaque séance aux habitudes du même utilisateur dans le même sport.

UserStats est un transformer scikit-learn : `fit` apprend les habitudes (médiane, écart-type, valeurs triées) sur les
séances qu'on lui donne, `transform` construit les variables. Dans un Pipeline avec `evaluate`, `fit` ne voit donc que
le train de chaque fold : pas de fuite. Aucune statistique n'utilise la cible `intensity`.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

KEYS = ["user_id", "sport"]
RAW = ["duration_min", "distance_km", "speed_kmh", "max_speed_kmh", "speed_std_kmh",
       "altitude_mean_m", "altitude_gain_m", "altitude_loss_m", "n_points",
       "month", "weekday", "start_hour_utc", "year"]
REL = ["speed_kmh", "duration_min", "distance_km", "max_speed_kmh", "speed_std_kmh",
       "altitude_gain_m", "climb_per_km", "peak_ratio", "speed_cv"]


def clean(df):
    """Nettoyage identique pour le train et le test."""
    d = df.copy()
    d.loc[d["altitude_mean_m"] < -500, "altitude_mean_m"] = np.nan   # -9999 = capteur absent
    d["distance_km"] = d["distance_km"].clip(lower=0.05)
    d["speed_kmh"] = d["speed_kmh"].clip(lower=0.1)
    d["climb_per_km"] = d["altitude_gain_m"] / d["distance_km"]
    d["peak_ratio"] = d["max_speed_kmh"] / d["speed_kmh"]
    d["speed_cv"] = d["speed_std_kmh"] / d["speed_kmh"]
    return d


class UserStats(BaseEstimator, TransformerMixin):
    """Pour chaque variable de REL : valeur / médiane perso (rel), écart en écarts-types (z), rang parmi ses séances (rank)."""

    def __init__(self, min_count=5):
        self.min_count = min_count

    def fit(self, X, y=None):
        d = clean(X)
        self.sports_ = sorted(d["sport"].unique())
        self.genders_ = sorted(d["gender"].astype(str).unique())
        self.group_ = {}
        for key, g in d.groupby(KEYS):
            self.group_[key] = {c: np.sort(g[c].dropna().to_numpy()) for c in REL}
        self.sport_ = {s: {c: np.sort(g[c].dropna().to_numpy()) for c in REL} for s, g in d.groupby("sport")}
        return self

    def _ref(self, key):
        g = self.group_.get(key)
        if g is not None and len(g["speed_kmh"]) >= self.min_count:
            return g
        return self.sport_.get(key[1])          # utilisateur trop peu connu : on se rabat sur le sport

    def transform(self, X):
        d = clean(X).reset_index(drop=True)
        out = d[RAW + ["climb_per_km", "peak_ratio", "speed_cv"]].copy()
        out["sport"] = d["sport"].map({s: i for i, s in enumerate(self.sports_)}).fillna(-1)
        out["gender"] = d["gender"].astype(str).map({s: i for i, s in enumerate(self.genders_)}).fillna(-1)
        new = {f"{p}_{c}": np.full(len(d), np.nan) for c in REL for p in ("rel", "z", "rank")}
        new["n_user_sport"] = np.zeros(len(d))
        for key, idx in d.groupby(KEYS).indices.items():
            ref = self._ref(key)
            if ref is None:
                continue
            new["n_user_sport"][idx] = len(ref["speed_kmh"])
            for c in REL:
                ref_c = ref[c]
                if len(ref_c) == 0:
                    continue
                v = d[c].to_numpy()[idx]
                med, sd = np.median(ref_c), ref_c.std()
                with np.errstate(divide="ignore", invalid="ignore"):
                    new[f"rel_{c}"][idx] = np.where(med != 0, v / med, np.nan)
                    new[f"z_{c}"][idx] = np.where(sd > 0, (v - med) / sd, 0.0)
                new[f"rank_{c}"][idx] = (np.searchsorted(ref_c, v, side="left")
                                         + np.searchsorted(ref_c, v, side="right")) / (2 * len(ref_c))
        return pd.concat([out, pd.DataFrame(new)], axis=1)
