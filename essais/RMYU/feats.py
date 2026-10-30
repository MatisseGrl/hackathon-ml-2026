import numpy as np, pandas as pd
from cleaning import clean

BASE = ["duration_min","distance_km","speed_kmh","max_speed_kmh","speed_std_kmh","altitude_mean_m","altitude_gain_m","altitude_loss_m","n_points"]

def build(train, test):
    """Features relatives au user/sport, stats sans cible calculées sur train+test."""
    a = pd.concat([train.drop(columns="intensity"), test], keys=["tr","te"])
    a = clean(a).reset_index(level=0, names="part").reset_index(drop=True)
    a["gain_per_km"] = a.altitude_gain_m / a.distance_km
    a["vspeed"] = a.altitude_gain_m / a.duration_min
    a["spd_ratio"] = a.max_speed_kmh / a.speed_kmh
    cols = BASE + ["gain_per_km","vspeed","spd_ratio"]
    out = a.copy()
    for keys, tag in [(["user_id","sport"],"us"), (["user_id"],"u"), (["sport"],"s")]:
        g = a.groupby(keys)
        for c in cols:
            out[f"{c}_rk_{tag}"] = g[c].rank(pct=True)
            if tag != "s":
                out[f"{c}_z_{tag}"] = (a[c]-g[c].transform("median"))/(g[c].transform("std")+1e-6)
    out["n_us"] = a.groupby(["user_id","sport"])["user_id"].transform("size")
    out["t"] = a.year + (a.month-1)/12
    # forme récente : rang de vitesse dans le user-sport-année
    g = a.groupby(["user_id","sport","year"])
    for c in ["speed_kmh","duration_min","distance_km"]:
        out[f"{c}_rk_usy"] = g[c].rank(pct=True)
    out["sport"] = out["sport"].astype("category"); out["gender"]=out["gender"].astype("category")
    return out


def add_resid(F):
    """Vitesse « attendue » pour ce user/sport vu la distance, le dénivelé et l'année (régression par groupe, sans cible)."""
    F = F.copy()
    F["lspd"] = np.log1p(F.speed_kmh); F["ldist"] = np.log1p(F.distance_km); F["ldur"]=np.log1p(F.duration_min)
    F["gpk"] = F.gain_per_km.fillna(F.gain_per_km.median()).clip(0, 100)
    F["resid_spd"] = np.nan; F["resid_maxspd"] = np.nan
    for _, idx in F.groupby(["user_id","sport"], observed=True).groups.items():
        g = F.loc[idx]
        if len(g) < 8:
            continue
        A = np.column_stack([np.ones(len(g)), g.ldist, g.gpk, g.t - g.t.mean()])
        for src, dst in [("lspd","resid_spd"), ("max_speed_kmh","resid_maxspd")]:
            yv = g[src].values
            coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
            r = yv - A @ coef
            F.loc[idx, dst] = r / (r.std() + 1e-6)
    F["resid_spd_rk"] = F.groupby(["user_id","sport"], observed=True).resid_spd.rank(pct=True)
    # forme récente : écart à la vitesse médiane des 10 séances voisines dans le temps
    F = F.sort_values(["user_id","sport","t","workout_id"])
    F["spd_roll"] = F.groupby(["user_id","sport"], observed=True).lspd.transform(lambda s: s.rolling(11, center=True, min_periods=3).median())
    F["spd_vs_roll"] = F.lspd - F.spd_roll
    return F.sort_index()
