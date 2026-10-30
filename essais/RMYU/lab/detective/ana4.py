import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from src.validation import load_data
tr, te, _ = load_data()
a = pd.concat([tr, te]); a["t"] = a.year + (a.month-1)/12
print("spearman workout_id vs t (all):", spearmanr(a.workout_id, a.t)[0])
w = a.groupby("user_id").apply(lambda d: spearmanr(d.workout_id, d.t)[0] if len(d)>5 else np.nan)
print("within user spearman wid vs t median:", w.median(), w.quantile([.1,.5,.9]).round(3).tolist())
print("wid vs user:", spearmanr(a.workout_id, a.user_id)[0])
print(a.sort_values("workout_id")[["workout_id","user_id","year","month"]].head(12).to_string())
# label vs t within user-sport, by year
tr["y"] = tr.intensity.map({"easy":0,"moderate":1,"hard":2})
tr["t"] = tr.year + (tr.month-1)/12
tr["trk"] = tr.groupby(["user_id","sport"]).t.rank(pct=True)
print(tr.groupby(pd.cut(tr.trk,10)).y.mean().round(3).to_string())
print("month effect (global):", tr.groupby("month").y.mean().round(3).to_dict())
print("hour effect:", tr.groupby("start_hour_utc").y.mean().round(2).to_dict())
print("n_points<500 share", (tr.n_points<500).mean(), tr.groupby(tr.n_points<500).y.mean().to_dict())
