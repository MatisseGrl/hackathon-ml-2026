import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from src.validation import load_data
tr, te, _ = load_data()
tr["y"] = tr.intensity.map({"easy":0,"moderate":1,"hard":2})
a = pd.concat([tr, te.assign(y=np.nan)])
for keys in [["user_id"], ["user_id","sport"]]:
    N = a.groupby(keys).size()
    K = tr.groupby(keys + ["y"]).size().unstack(fill_value=0).reindex(N.index, fill_value=0)
    over_ceil = (K.values > np.ceil(N.values/3)[:,None]).sum()
    over_floor = (K.values > np.floor(N.values/3)[:,None]+0).sum()
    # remaining per group: N/3 - K ; number of groups where any class remaining < 0 (with ceil)
    rem = np.ceil(N.values/3)[:,None] - K.values
    print(keys, "groups", len(N), "cells with known>ceil(N/3):", over_ceil, " known>floor:", over_floor, "min rem", rem.min())
    # exactness test: total labeled vs known; is K max close to N/3
    fill = (K.values.max(1) / (N.values/3))
    print("   max(K)/(N/3) quantiles", np.quantile(fill, [.5,.9,.99,1]).round(3))
# users with multiple sports: per sport, N, K
ns = a.groupby("user_id").sport.nunique(); multi = ns[ns>1].index
N = a[a.user_id.isin(multi)].groupby(["user_id","sport"]).size().rename("N")
K = tr[tr.user_id.isin(multi)].groupby(["user_id","sport","y"]).size().unstack(fill_value=0)
print(pd.concat([N, K, (N/3).round(1).rename("N/3")], axis=1).to_string())
