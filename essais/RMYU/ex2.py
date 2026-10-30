import sys; sys.path.append("../.."); sys.path.append(".")
import pandas as pd, numpy as np
from src.validation import load_data
tr,te,_=load_data(); tr["y"]=tr.intensity.map({"easy":0,"moderate":1,"hard":2})
us=tr.groupby(["user_id","sport"]).y.agg(["size",lambda s:(s==2).mean(),lambda s:(s==0).mean()]); us.columns=["n","hard","easy"]
print("user-sport n>=30 hard share std:", us[us.n>=30].hard.std().round(3), "easy std", us[us.n>=30].easy.std().round(3))
u=tr.groupby("user_id").y.agg(["size",lambda s:(s==2).mean()]); print("user hard std", u[u["size"]>=30].iloc[:,1].std().round(3))
# per user mix of sports: does sport matter inside user?
m=tr.groupby(["user_id"]).sport.nunique(); multi=m[m>1].index
x=tr[tr.user_id.isin(multi)].groupby(["user_id","sport"]).y.mean().unstack(); print(x.mean().round(2))
print(tr.groupby("year").y.mean().round(2).to_dict())
print(tr.groupby("month").y.mean().round(2).to_dict())
print(tr.groupby("start_hour_utc").y.mean().round(2).values)
# within user: label vs year
tr["yr_rel"]=tr.year-tr.groupby("user_id").year.transform("min")
print(tr.groupby("yr_rel").y.mean().round(2).to_dict())
print(tr.groupby(pd.qcut(tr.duration_min,10)).y.mean().round(2).values)
