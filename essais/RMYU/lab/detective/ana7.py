import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from src.validation import load_data
tr, te, _ = load_data(); tr["y"] = tr.intensity.map({"easy":0,"moderate":1,"hard":2})
PART = ["user_id","sport","duration_min","distance_km"]
FULL = ["user_id","sport","gender","year","month","weekday","start_hour_utc","duration_min","distance_km","speed_kmh","max_speed_kmh","speed_std_kmh","altitude_mean_m","altitude_gain_m","altitude_loss_m","n_points"]
g = tr.groupby(PART)
d = tr[g.y.transform("size") == 2].sort_values(PART + ["year","month"])
A = d.iloc[::2].reset_index(drop=True); B = d.iloc[1::2].reset_index(drop=True)
same_full = (A[FULL].astype(str).values == B[FULL].astype(str).values).all(1)
print("partial pairs", len(A), "exact-dup pairs", same_full.sum(), "label agree exact", (A.y==B.y)[same_full].mean().round(3), "partial-only agree", (A.y==B.y)[~same_full].mean().round(3))
P = ~same_full
for c in ["year","month","weekday","start_hour_utc","speed_kmh","max_speed_kmh","altitude_gain_m","n_points"]:
    diff = (A[c].astype(str).values != B[c].astype(str).values)[P]
    agree = (A.y.values == B.y.values)[P]
    print(f"  {c:16s} differs in {diff.mean():.2f} of partial pairs; agree if differs {agree[diff].mean() if diff.any() else np.nan:.3f} / if same {agree[~diff].mean() if (~diff).any() else np.nan:.3f}")
dt = (A.year*12+A.month - B.year*12-B.month).abs()[P]
agree = (A.y.values == B.y.values)[P]
print(pd.Series(agree).groupby(pd.cut(dt.values, [-1,0,1,3,12,100])).agg(["mean","size"]).to_string())
