"""Nettoyage des séances : sans cible, donc sans fuite, applicable à train et test."""
import numpy as np
import pandas as pd

ALT_MIN, ALT_MAX = -100, 4000        # altitude moyenne plausible (m)
GAIN_MAX = 3000                      # dénivelé positif/négatif plausible sur une séance (m)


def clean(df):
    """Renvoie une copie : valeurs aberrantes -> NaN (l'imputeur de make_preprocessor les remplace),
    plus des indicateurs de ce qui a été corrigé. Aucune ligne n'est supprimée (le test doit rester complet)."""
    d = df.copy()
    bad_alt = ~d["altitude_mean_m"].between(ALT_MIN, ALT_MAX)
    bad_gain = (d["altitude_gain_m"] > GAIN_MAX) | (d["altitude_loss_m"] > GAIN_MAX)
    d["altitude_suspecte"] = (bad_alt | bad_gain).astype(int)
    d.loc[bad_alt, "altitude_mean_m"] = np.nan
    d.loc[bad_gain, ["altitude_gain_m", "altitude_loss_m"]] = np.nan
    d["altitude_loss_m"] = d["altitude_loss_m"].abs()          # -0.0
    d["n_points_plein"] = (d["n_points"] >= 500).astype(int)   # plafonné à 500 : indicateur de séance tronquée
    d["max_speed_kmh"] = d[["max_speed_kmh", "speed_kmh"]].max(axis=1)  # vitesse max >= vitesse moyenne
    return d
