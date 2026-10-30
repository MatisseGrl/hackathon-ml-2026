import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from src.validation import load_data
tr, te, _ = load_data(); tr["y"] = tr.intensity.map({"easy":0,"moderate":1,"hard":2})
tr["t"] = tr.year + (tr.month-1)/12
# paires même créneau (user, year, month, weekday, hour)
slot = ["user_id","year","month","weekday","start_hour_utc"]
g = tr.groupby(slot)
tr["slot_n"] = g.y.transform("size")
d = tr[tr.slot_n == 2].sort_values(slot)
A = d.iloc[::2].reset_index(drop=True); B = d.iloc[1::2].reset_index(drop=True)
ok = (A[slot].values == B[slot].values).all(1); A, B = A[ok], B[ok]
same = A.y.values == B.y.values
print("slot pairs", len(A), "same label", same.mean().round(3), "|dy|", np.abs(A.y.values-B.y.values).mean().round(3))
dspd = (np.log(A.speed_kmh.values) - np.log(B.speed_kmh.values)); dy = A.y.values - B.y.values
print("corr dy vs dlogspeed", np.corrcoef(dspd, dy)[0,1].round(3), " sign agreement when |dy|>0:", (np.sign(dspd[dy!=0])==np.sign(dy[dy!=0])).mean().round(3))
for c in ["duration_min","distance_km","altitude_gain_m","n_points","max_speed_kmh","speed_std_kmh","altitude_mean_m"]:
    dc = np.log1p(A[c].fillna(0).clip(lower=0).values) - np.log1p(B[c].fillna(0).clip(lower=0).values)
    print(f"  corr dy vs d{c}: {np.corrcoef(dc, dy)[0,1]:.3f}")
# même user-sport-mois : différence label vs différence vitesse (intra mois)
tr["yc_m"] = tr.y - tr.groupby(["user_id","sport","year","month"]).y.transform("mean")
tr["sc_m"] = np.log(tr.speed_kmh) - np.log(tr.speed_kmh).groupby([tr.user_id,tr.sport,tr.year,tr.month]).transform("mean")
tr["yc_u"] = tr.y - tr.groupby(["user_id","sport"]).y.transform("mean")
tr["sc_u"] = np.log(tr.speed_kmh) - np.log(tr.speed_kmh).groupby([tr.user_id,tr.sport]).transform("mean")
m = tr.groupby(["user_id","sport","year","month"]).y.transform("size") >= 3
print("intra-mois corr y~speed", np.corrcoef(tr.sc_m[m], tr.yc_m[m])[0,1].round(3), " intra-user corr", np.corrcoef(tr.sc_u, tr.yc_u)[0,1].round(3))
# variance du label expliquée par le mois (user-sport-mois) : 
mm = tr.groupby(["user_id","sport","year","month"]).y.transform("mean"); n = tr.groupby(["user_id","sport","year","month"]).y.transform("size")
print("share of within-user var between months (n>=3):", (((mm-tr.groupby(['user_id','sport']).y.transform('mean'))**2)[m].mean() / (tr.yc_u[m]**2).mean()).round(3))
# speed level du mois vs label level du mois
lv = tr[m].groupby(["user_id","sport","year","month"]).agg(ym=("yc_u","mean"), sm=("sc_u","mean"), n=("y","size"), dur=("duration_min","median"))
print("corr month label level vs month speed level:", np.corrcoef(lv.ym, lv.sm)[0,1].round(3), " vs month n sessions", np.corrcoef(lv.ym, lv.n)[0,1].round(3), "vs dur", np.corrcoef(lv.ym, np.log(lv.dur))[0,1].round(3))
