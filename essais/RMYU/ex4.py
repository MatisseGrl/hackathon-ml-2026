import sys; sys.path.append("../.."); sys.path.append(".")
import pandas as pd, numpy as np
from src.validation import load_data
tr,te,_=load_data(); tr["y"]=tr.intensity.map({"easy":0,"moderate":1,"hard":2})
tr["t"]=tr.year+(tr.month-1)/12
tr["trk"]=tr.groupby("user_id").t.rank(pct=True)
print(tr.groupby(pd.cut(tr.trk,10)).y.mean().round(2).values)
# per-user correlation of y with time
c = tr.groupby("user_id").apply(lambda d: d.y.corr(d.t) if len(d)>30 else np.nan).dropna()
print("per-user corr(y,t): mean", c.mean().round(3), "abs mean", c.abs().mean().round(3))
# same-month neighbors label mean
tr["ym"]=tr.year*12+tr.month
g=tr.groupby(["user_id","ym"]).y; tr["loo"]=(g.transform("sum")-tr.y)/(g.transform("size")-1)
print("corr y vs LOO same user-month:", tr[["y","loo"]].corr().iloc[0,1].round(3))
# workout_id neighbors within user
tr=tr.sort_values(["user_id","workout_id"]); g=tr.groupby("user_id")
print("corr y vs wid-prev", tr.y.corr(g.y.shift()).round(3), "wid vs t corr", tr.workout_id.corr(tr.t).round(3))
print("n_points by class", tr.groupby("y").n_points.mean().round(1).to_dict(), (tr.n_points<500).groupby(tr.y).mean().round(3).to_dict())
print(tr.groupby("y")[["weekday","start_hour_utc","month"]].mean().round(2))
