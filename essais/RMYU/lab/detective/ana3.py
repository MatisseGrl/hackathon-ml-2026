import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from src.validation import load_data
tr, te, _ = load_data()
F = pd.read_pickle("lab/detective/F_base.pkl")
Ft = F[F.part=="tr"].reset_index(drop=True)
y = tr.intensity.map({"easy":0,"moderate":1,"hard":2}).values
Ft["y"] = y
# within user-sport centered correlation: y - mean_us(y) vs feature rank within us
num = [c for c in Ft.columns if c not in ("part","workout_id","y","sport","gender") and Ft[c].dtype.kind in "fi"]
yc = Ft.y - Ft.groupby(["user_id","sport"], observed=True).y.transform("mean")
res = {}
for c in num:
    r = Ft.groupby(["user_id","sport"], observed=True)[c].rank(pct=True) - 0.5
    m = r.notna()
    res[c] = np.corrcoef(r[m], yc[m])[0,1]
s = pd.Series(res).sort_values(key=abs, ascending=False)
print(s.head(45).round(3).to_string())
