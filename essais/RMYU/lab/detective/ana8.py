import sys; sys.path.append("../.."); sys.path.append("."); sys.path.append("lab/detective")
import numpy as np, pandas as pd
from src.validation import load_data
from myfeats import add_my_feats
tr, te, _ = load_data()
F0 = pd.read_pickle("lab/detective/F_base.pkl"); base = set(F0.columns)
F = add_my_feats(F0)
Ft = F[F.part=="tr"].reset_index(drop=True); Ft["y"] = tr.intensity.map({"easy":0,"moderate":1,"hard":2}).values
yc = Ft.y - Ft.groupby(["user_id","sport"], observed=True).y.transform("mean")
res = {}
for c in [c for c in F.columns if c not in base]:
    r = Ft.groupby(["user_id","sport"], observed=True)[c].rank(pct=True) - 0.5; m = r.notna()
    res[c] = (np.corrcoef(r[m], yc[m])[0,1], np.corrcoef(Ft[c][m].rank(pct=True), Ft.y[m])[0,1])
print(pd.DataFrame(res, index=["r_intra_us(rank)", "r_global(rank)"]).T.sort_values("r_intra_us(rank)", key=abs, ascending=False).round(3).to_string())
