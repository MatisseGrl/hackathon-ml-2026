import sys; sys.path.append("../.."); sys.path.append("."); sys.path.append("lab/detective")
import numpy as np, pandas as pd
from src.validation import load_data
from myfeats import add_my_feats
tr, te, _ = load_data()
F = add_my_feats(pd.read_pickle("lab/detective/F_base.pkl"))
y = tr.intensity.map({"easy":0,"moderate":1,"hard":2}).values
F["y"] = np.nan; F.loc[F.part=="tr", "y"] = y
K = ["user_id","sport","year","month"]
F["yc"] = F.y - F.groupby(["user_id","sport"], observed=True).y.transform("mean")
num = [c for c in F.columns if F[c].dtype.kind in "fi" and c not in ("y","yc","user_id","workout_id")]
# month-level means of each feature centered by user-sport (train+test rows), vs month label level (train rows)
G = F.groupby(K, observed=True)
lab = G.yc.agg(["mean","count"]); lab = lab[lab["count"]>=4]
res = {}
for c in num:
    z = F[c] - F.groupby(["user_id","sport"], observed=True)[c].transform("mean")
    mz = z.groupby([F[k] for k in K], observed=True).mean().reindex(lab.index)
    ok = mz.notna()
    if ok.sum() > 100: res[c] = np.corrcoef(mz[ok], lab["mean"][ok])[0,1]
print(pd.Series(res).sort_values(key=abs, ascending=False).head(25).round(3).to_string())
