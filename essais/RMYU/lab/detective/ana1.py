import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from src.validation import load_data
tr, te, _ = load_data()
tr["y"] = tr.intensity.map({"easy":0,"moderate":1,"hard":2})
print("users train", tr.user_id.nunique(), "test", te.user_id.nunique(), "sports", tr.sport.value_counts().to_dict())
a = pd.concat([tr, te.assign(y=np.nan)])
ns = a.groupby("user_id").sport.nunique()
multi = ns[ns > 1].index
print("multi-sport users:", len(multi))
for u in multi[:15]:
    d = tr[tr.user_id == u]
    ct = pd.crosstab(d.sport, d.intensity)[["easy","moderate","hard"]] if len(d) else None
    sp = d.groupby("sport").speed_kmh.median().round(1)
    print("user", u, "n", len(d)); print(pd.concat([ct, sp], axis=1).to_string())
# within user+sport label balance across all multi-sport users
d = tr[tr.user_id.isin(multi)]
g = d.groupby(["user_id","sport"]).y.agg(["mean","size"])
print(g[g["size"]>=10].to_string())
