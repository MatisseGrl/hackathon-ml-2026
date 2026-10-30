"""Variables sans cible (calculÃ©es sur train+test) pour le projet intensitÃ©.
add_my_feats(F) : F = feats.add_resid(feats.build(train, test)) -> F enrichi.
Constats (detective) : label = tiers PAR (user, sport) ; le label baisse avec l'anciennetÃ© dans le user-sport
(1,21 -> 0,89 du 1er au dernier dÃ©cile de temps) ; hiver > Ã©tÃ© ; vitesse = moteur principal (r=0,35 intra-groupe)."""
import numpy as np, pandas as pd

US = ["user_id", "sport"]


def _grp_rank(F, keys, col):
    return F.groupby(keys, observed=True)[col].rank(pct=True)


def add_my_feats(F, which=("time", "vol", "month", "pow", "samp", "season", "slot", "load", "cap")):
    F = F.copy()
    if "time" in which:  # anciennetÃ© / position dans l'historique du user (dÃ©rive de la forme)
        F["tm_rk_us"] = _grp_rank(F, US, "t")
        F["tm_rk_u"] = _grp_rank(F, ["user_id"], "t")
        g = F.groupby(US, observed=True).t
        F["tm_rel_us"] = (F.t - g.transform("min")) / (g.transform("max") - g.transform("min") + 1 / 12)
        F["tm_since_first_u"] = F.t - F.groupby("user_id").t.transform("min")
    if "vol" in which:  # charge d'entraÃ®nement du mois / mÃªme crÃ©neau
        F["vol_n_um"] = F.groupby(["user_id", "year", "month"]).user_id.transform("size")
        F["vol_n_usm"] = F.groupby(US + ["year", "month"], observed=True).user_id.transform("size")
        F["vol_dur_um"] = F.groupby(["user_id", "year", "month"]).duration_min.transform("sum")
        F["vol_n_slot"] = F.groupby(["user_id", "year", "month", "weekday", "start_hour_utc"]).user_id.transform("size")
        F["vol_n_wd"] = F.groupby(["user_id", "year", "month", "weekday"]).user_id.transform("size")
        F["vol_n_um_rel"] = F.vol_n_um / F.groupby("user_id").vol_n_um.transform("median")
    if "month" in which:  # niveau de vitesse du mois (forme du moment) et Ã©cart Ã  ce niveau
        gm = F.groupby(US + ["year", "month"], observed=True).lspd
        gu = F.groupby(US, observed=True).lspd
        F["mo_spd_lvl"] = gm.transform("mean") - gu.transform("mean")
        F["mo_spd_dev"] = F.lspd - gm.transform("mean")
        F["mo_res_lvl"] = F.groupby(US + ["year", "month"], observed=True).resid_spd.transform("mean")
        F["mo_res_dev"] = F.resid_spd - F.mo_res_lvl
        F["mo_spd_rk"] = _grp_rank(F, US + ["year", "month"], "lspd")
        F["mo_dist_rk"] = _grp_rank(F, US + ["year", "month"], "distance_km")
        F["mo_n"] = gm.transform("size")
    if "pow" in which:  # puissance mÃ©canique approchÃ©e (W/kg) : aÃ©ro + roulement + gravitÃ© ; course : coÃ»t ~ v*(1+k*pente)
        v = F.speed_kmh / 3.6
        vs = (F.altitude_gain_m.fillna(0) / (F.duration_min * 60)).clip(0, 3)
        cyc = F.sport.astype(str).str.contains("cycl|mtb")
        p_cyc = 0.5 * 1.2 * 0.4 * v ** 3 / 80 + 0.005 * 9.81 * v + 9.81 * vs
        p_run = v * (1 + 9.0 * (F.altitude_gain_m.fillna(0) / (F.distance_km * 1000).clip(lower=100)))
        F["pw"] = np.where(cyc, p_cyc, p_run)
        F["pw_rk_us"] = _grp_rank(F, US, "pw")
        g = F.groupby(US, observed=True).pw
        F["pw_z_us"] = (F.pw - g.transform("median")) / (g.transform("std") + 1e-6)
        F["lpw"] = np.log(F.pw.clip(lower=1e-3))
        # dÃ©rive cardiaque : effort x durÃ©e
        F["pw_x_dur"] = F.pw_z_us * np.log1p(F.duration_min)
    if "samp" in which:  # cadence d'Ã©chantillonnage (appareil / mode d'enregistrement)
        F["samp_sec"] = F.duration_min * 60 / F.n_points
        F["samp_trunc"] = (F.n_points >= 500).astype(int)
        F["samp_sec_z_u"] = (F.samp_sec - F.groupby("user_id").samp_sec.transform("median")) / (F.groupby("user_id").samp_sec.transform("std") + 1e-6)
    if "season" in which:
        F["sea_sin"] = np.sin(2 * np.pi * (F.month - 1) / 12)
        F["sea_cos"] = np.cos(2 * np.pi * (F.month - 1) / 12)
    if "slot" in which:  # séances du même créneau (user, année, mois, jour, heure) : la plus longue/rapide est plus « hard »
        SL = ["user_id", "year", "month", "weekday", "start_hour_utc"]
        g = F.groupby(SL)
        F["slot_spd_dev"] = F.lspd - g.lspd.transform("mean")
        F["slot_dist_share"] = F.distance_km / (g.distance_km.transform("sum") + 1e-6)
        F["slot_dur_share"] = F.duration_min / (g.duration_min.transform("sum") + 1e-6)
    if "load" in which:  # charge d'entraînement glissante (forme physique) : volume des mois voisins du user
        mi = (F.year * 12 + F.month - 1).astype(int)
        um = pd.DataFrame({"u": F.user_id.values, "m": mi.values, "d": F.duration_min.values, "n": 1.0})
        piv = um.groupby(["u", "m"])[["d", "n"]].sum()
        def win(lo, hi, col):
            out = np.zeros(len(F))
            for k in range(lo, hi + 1):
                idx = pd.MultiIndex.from_arrays([F.user_id.values, mi.values + k])
                out += piv[col].reindex(idx).fillna(0).values
            return out
        F["load_dur_prev3"] = win(-3, -1, "d")
        F["load_n_prev3"] = win(-3, -1, "n")
        F["load_dur_c3"] = win(-1, 1, "d")
        F["load_dur_next3"] = win(1, 3, "d")
        F["load_n_cum"] = F.groupby("user_id").t.rank(method="max")  # nb de séances du user jusqu'à ce mois
        med = F.groupby("user_id").load_dur_c3.transform("median") + 1
        F["load_c3_rel"] = F.load_dur_c3 / med
        F["load_prev3_rel"] = F.load_dur_prev3 / med
    if "cap" in which:  # capacité du moment : quantiles de vitesse du user-sport sur ±2 mois -> intensité relative
        mi = (F.year * 12 + F.month - 1).astype(int).values
        q90 = np.full(len(F), np.nan); q50 = np.full(len(F), np.nan)
        for _, idx in F.groupby(US, observed=True).indices.items():
            m = mi[idx]; v = F.lspd.values[idx]
            for mm in np.unique(m):
                w = np.abs(m - mm) <= 2
                sel = idx[m == mm]
                if w.sum() >= 4:
                    q90[sel] = np.quantile(v[w], 0.9); q50[sel] = np.quantile(v[w], 0.5)
        F["cap_q90"] = q90 - F.groupby(US, observed=True).lspd.transform("median")
        F["cap_rel90"] = F.lspd - q90
        F["cap_rel50"] = F.lspd - q50
    return F
