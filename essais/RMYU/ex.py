import sys; sys.path.append("../.."); sys.path.append(".")
import pandas as pd, numpy as np
from src.validation import load_data
from feats import build
train,test,_ = load_data()
F = build(train,test); T=F[F.part=="tr"].copy(); T["y"]=train.intensity.map({"easy":0,"moderate":1,"hard":2}).values
rk=[c for c in T.columns if "_rk_" in c or "_z_" in c]
print(T.groupby("y")[rk].mean().T.assign(d=lambda d:d[2]-d[0]).sort_values("d").round(3).iloc[list(range(6))+list(range(-8,0))])
# autocorrelation by workout_id within user
T=T.sort_values(["user_id","workout_id"]); T["prev"]=T.groupby("user_id").y.shift(); T["wid_gap"]=T.groupby("user_id").workout_id.diff()
print("corr y,prev:",T[["y","prev"]].corr().iloc[0,1].round(3))
print(train.sort_values("workout_id")[["workout_id","user_id","year","month","weekday"]].head(8))
print(T.groupby("user_id").size().describe())
