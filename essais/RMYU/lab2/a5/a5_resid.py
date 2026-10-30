import sys, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
T = pd.read_pickle("lab2/a5/T.pkl")
T["res"] = T.y - T.Emean
T["g_rank_dur"] = T.groupby("us").duration_min.rank(pct=True)
T["g_rank_spd"] = T.groupby("us").speed_kmh.rank(pct=True)
T["g_rank_t"] = T.groupby("us").t.rank(pct=True)
cols = ["duration_min", "distance_km", "speed_kmh", "max_speed_kmh", "speed_std_kmh", "altitude_mean_m", "altitude_gain_m",
        "n_points", "start_hour_utc", "weekday", "month", "year", "g_rank_dur", "g_rank_spd", "g_rank_t", "N", "trank"]
out = []
for c in cols:
    try:
        b = pd.qcut(T[c], 10, duplicates="drop")
    except Exception:
        continue
    g = T.groupby(b, observed=True).res.agg(["mean", "size"])
    se = T.res.std() / np.sqrt(g["size"])
    z = (g["mean"] / se).abs().max()
    i = (g["mean"] / se).abs().idxmax()
    out.append((c, round(z, 1), str(i), round(g.loc[i, "mean"], 3), int(g.loc[i, "size"])))
for r in sorted(out, key=lambda r: -r[1]):
    print(r)
print("global mean res", round(T.res.mean(), 4), "sd", round(T.res.std(), 3))
for c in ["sport"]:
    print(T.groupby(c).res.agg(["mean", "size"]).round(3))
