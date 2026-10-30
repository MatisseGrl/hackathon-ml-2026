"""Diagnostic : les gens refont-ils les mêmes parcours, et la vitesse relative AU MÊME PARCOURS prédit-elle mieux le label
que la vitesse relative à toutes les séances de la personne ? (variables calculées sur train + test sans la cible ;
corrélations mesurées sur le train seulement, sans rien ajuster)"""
import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from src.validation import load_data
train, test, _ = load_data()
y = train.intensity.map({"easy": 0, "moderate": 1, "hard": 2}).values
D = pd.concat([train.drop(columns="intensity"), test], ignore_index=True)
ntr = len(train)
def z(s, keys):
    g = D.groupby(keys)[s]
    return (D[s] - g.transform("mean")) / g.transform("std").replace(0, np.nan), g.transform("size")
D["dist_r"] = (D.distance_km / 1.0).round()          # parcours ~ même distance (1 km), même altitude moyenne (25 m), même dénivelé (25 m)
D["alt_r"] = (D.altitude_mean_m / 25).round(); D["gain_r"] = (D.altitude_gain_m / 25).round()
zu, nu = z("speed_kmh", ["user_id", "sport"])
for name, keys in [("route dist", ["user_id", "sport", "dist_r"]), ("route dist+alt", ["user_id", "sport", "dist_r", "alt_r"]),
                   ("route dist+alt+gain", ["user_id", "sport", "dist_r", "alt_r", "gain_r"])]:
    zr, nr = z("speed_kmh", keys)
    m = (nr.values[:ntr] >= 5) & np.isfinite(zr.values[:ntr])
    cu = np.corrcoef(zu.values[:ntr][m], y[m])[0, 1]; cr = np.corrcoef(zr.values[:ntr][m], y[m])[0, 1]
    # partie de la vitesse-parcours non expliquée par la vitesse-user : apporte-t-elle du signal en plus ?
    resid = zr.values[:ntr][m] - np.polyval(np.polyfit(zu.values[:ntr][m], zr.values[:ntr][m], 1), zu.values[:ntr][m])
    print(f"{name:20s} séances du train dans un parcours d'au moins 5 séances : {m.mean():.1%} ; corr(label, z vitesse user+sport) {cu:.3f} ;"
          f" corr(label, z vitesse parcours) {cr:.3f} ; corr(label, part propre au parcours) {np.corrcoef(resid, y[m])[0, 1]:.3f}", flush=True)
