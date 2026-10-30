"""Variables d'effort physiologique, SANS cible (calculées sur train+test).
add_physio(F) prend la sortie de add_resid(build(train, test)) et ajoute des colonnes ph_* :
 1) allure attendue globale : HGBRegressor (cross-fit 5 plis, sans cible) qui prédit log(vitesse) - médiane user/sport
    à partir de la distance relative, pente, altitude, mois, heure, ancienneté, profil du user -> résidu ph_res_g
 2) allure attendue par user/sport : régression ridge (distance, distance², pente, saison sin/cos, heure, tendance t, t²)
    -> résidu standardisé ph_res_u ; idem vitesse max -> ph_resmax_u
 3) rangs de ces résidus et de la vitesse dans user, user+mois, user+trimestre, user+année ; z dans user+mois
 4) charge d'entraînement : nb de séances / durée totale du user ce mois-ci (relatives à son habitude), mois voisins
 5) fractionné : cv = std/vitesse, max/moyenne relatifs au user ; « TRIMP » = log durée + résidu
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold


def _rk(F, col, keys):
    return F.groupby(keys, observed=True)[col].rank(pct=True)


def _z(F, col, keys, minn=3):
    g = F.groupby(keys, observed=True)[col]
    n = g.transform("size")
    z = (F[col] - g.transform("median")) / (g.transform("std") + 1e-6)
    return z.where(n >= minn)


def add_physio(F, seed=0, glob_iters=300, ridge=1.0, light=False):
    F = F.copy()
    F["ym"] = F.year * 12 + F.month
    F["yq"] = F.year * 4 + (F.month - 1) // 3
    us = ["user_id", "sport"]
    g_us = F.groupby(us, observed=True)
    F["lmax"] = np.log1p(F.max_speed_kmh)
    F["lpk"] = F.altitude_loss_m / F.distance_km
    for c in ["lspd", "ldist", "lmax", "gpk", "altitude_mean_m", "ldur"]:
        F[f"ph_med_{c}"] = g_us[c].transform("median")
    F["ph_t0"] = F.t - g_us.t.transform("min")                      # ancienneté dans le sport
    F["ph_trel"] = F.t - g_us.t.transform("mean")
    F["ph_dalt"] = F.altitude_mean_m - F.ph_med_altitude_mean_m

    # 1) modèle global d'allure attendue (cross-fit, sans cible)
    feats = ["ldist", "gpk", "lpk", "altitude_mean_m", "ph_dalt", "month", "start_hour_utc", "weekday", "ph_t0", "ph_trel",
             "ph_med_lspd", "ph_med_ldist", "ph_med_gpk", "ph_med_lmax", "n_us"]
    D = F[feats].copy()
    D["dl"] = F.ldist - F.ph_med_ldist
    D["sport"] = F.sport.cat.codes; D["gender"] = F.gender.cat.codes
    D = D.values.astype(float)
    for tgt, name in [("lspd", "ph_res_g"), ("lmax", "ph_resmax_g")]:
        yv = (F[tgt] - F[f"ph_med_{tgt}"]).values
        pred = np.zeros(len(F))
        for a, b in KFold(5, shuffle=True, random_state=seed).split(D):
            m = HistGradientBoostingRegressor(max_iter=glob_iters, learning_rate=0.05, random_state=seed,
                                              categorical_features=[D.shape[1] - 2, D.shape[1] - 1])
            pred[b] = m.fit(D[a], yv[a]).predict(D[b])
        F[name] = yv - pred
    # 2) allure attendue par user/sport (ridge, design riche)
    F["ph_res_u"] = np.nan; F["ph_resmax_u"] = np.nan
    ms, mc = np.sin(2 * np.pi * F.month / 12), np.cos(2 * np.pi * F.month / 12)
    hs, hc = np.sin(2 * np.pi * F.start_hour_utc / 24), np.cos(2 * np.pi * F.start_hour_utc / 24)
    for _, idx in g_us.groups.items():
        g = F.loc[idx]
        n = len(g)
        if n < 12:
            continue
        tt = (g.t - g.t.mean()).values
        dl = (g.ldist - g.ldist.mean()).values
        cols = [np.ones(n), dl, dl ** 2, g.gpk.values, g.lpk.fillna(0).clip(0, 100).values, ms[idx], mc[idx], hs[idx], hc[idx], tt, tt ** 2]
        A = np.column_stack(cols)
        A[:, 1:] = (A[:, 1:] - A[:, 1:].mean(0)) / (A[:, 1:].std(0) + 1e-6)
        lam = ridge * np.eye(A.shape[1]); lam[0, 0] = 0
        for src, dst in [("lspd", "ph_res_u"), ("lmax", "ph_resmax_u")]:
            yv = g[src].values
            coef = np.linalg.solve(A.T @ A + lam, A.T @ yv)
            r = yv - A @ coef
            # correction du levier (résidu « hors-échantillon » approché)
            H = np.einsum("ij,jk,ik->i", A, np.linalg.inv(A.T @ A + lam), A)
            r = r / np.clip(1 - H, 0.2, 1)
            F.loc[idx, dst] = r / (r.std() + 1e-6)
    F["ph_res_mix"] = F.ph_res_u.fillna(0) + F.ph_res_g / (F.groupby(us, observed=True).ph_res_g.transform("std") + 1e-6)
    # 3) rangs / z dans des fenêtres temporelles du user
    for c in ["ph_res_g", "ph_res_u", "ph_res_mix", "lspd", "ph_resmax_u", "ldur", "ldist"]:
        if light and c in ("ldist",):
            continue
        for keys, tag in [(["user_id", "sport", "ym"], "um"), (["user_id", "sport", "yq"], "uq"), (["user_id", "sport", "year"], "uy"),
                          (["user_id"], "u")]:
            if c in ("lspd", "ldur", "ldist") and tag == "u":
                continue
            F[f"{c}_rk_{tag}"] = _rk(F, c, keys)
        F[f"{c}_z_um"] = _z(F, c, ["user_id", "sport", "ym"])
    F["ph_res_g_rk_us"] = _rk(F, "ph_res_g", us)
    # 4) charge d'entraînement
    cnt = F.groupby(["user_id", "ym"]).size().rename("c")
    dur = F.groupby(["user_id", "ym"]).duration_min.sum().rename("d")
    L = pd.concat([cnt, dur], axis=1).reset_index()
    base = L.groupby("user_id")[["c", "d"]].transform("mean")
    L["ph_load_c"] = L.c / base.c; L["ph_load_d"] = L.d / base.d
    Lk = L.set_index(["user_id", "ym"])
    key = pd.MultiIndex.from_arrays([F.user_id, F.ym])
    F["ph_load_c"] = Lk.ph_load_c.reindex(key).values
    F["ph_load_d"] = Lk.ph_load_d.reindex(key).values
    for s in (-1, 1):
        k2 = pd.MultiIndex.from_arrays([F.user_id, F.ym + s])
        F[f"ph_load_c{s:+d}"] = Lk.ph_load_c.reindex(k2).fillna(0).values
    F["ph_n_slot"] = F.groupby(["user_id", "ym", "weekday", "start_hour_utc"]).user_id.transform("size")
    F["ph_n_ym_us"] = F.groupby(["user_id", "sport", "ym"]).user_id.transform("size")
    # 5) fractionné / charge de séance
    F["ph_cv"] = F.speed_std_kmh / (F.speed_kmh + 0.1)
    F["ph_cv_z"] = _z(F, "ph_cv", us)
    F["ph_trimp"] = F.ldur + 0.5 * F.ph_res_mix
    F["ph_trimp_rk"] = _rk(F, "ph_trimp", us)
    F["ph_trimp_rk_um"] = _rk(F, "ph_trimp", ["user_id", "sport", "ym"])
    # tendance : résidu moyen des voisins temporels (forme du moment) et écart
    F = F.sort_values(us + ["t", "workout_id"])
    F["ph_res_roll"] = F.groupby(us, observed=True).ph_res_mix.transform(lambda s: s.rolling(15, center=True, min_periods=3).mean())
    F["ph_res_vs_roll"] = F.ph_res_mix - F.ph_res_roll
    F = F.sort_index()
    return F.drop(columns=["ym", "yq", "lmax", "lpk"])
