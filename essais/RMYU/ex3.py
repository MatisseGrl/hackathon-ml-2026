import sys; sys.path.append("../.."); sys.path.append(".")
import pandas as pd, numpy as np
from src.validation import load_data
tr,te,_=load_data(); tr["y"]=tr.intensity.map({"easy":0,"moderate":1,"hard":2})
for keys in [["user_id"],["user_id","sport"],["user_id","year"],["user_id","sport","year"]]:
    n = tr.groupby(keys).size(); N = pd.concat([tr,te]).groupby(keys).size().reindex(n.index)
    h = tr.groupby(keys).y.apply(lambda s:(s==2).sum())
    ok = n>=10
    # hypergeometric var for count of hard in sample n from N with N/3 hard
    p=1/3; var = n*p*(1-p)*(N-n)/(N-1).clip(lower=1)
    r = ((h-n*p)**2/var.clip(lower=1e-9))[ok & (N>n)]
    print(keys, "groups", ok.sum(), "ratio obs/expected var:", round(r.mean(),3))
al=pd.concat([tr,te]); ms = al.groupby("user_id").sport.nunique(); multi = ms[ms>1].index
t2=tr[tr.user_id.isin(multi)]; te2=te[te.user_id.isin(multi)]
print("multi-sport users:", len(multi), "rows", len(t2))
for keys in [["user_id"],["user_id","sport"]]:
    n = t2.groupby(keys).size(); N = pd.concat([t2,te2]).groupby(keys).size().reindex(n.index)
    for c in [0,2]:
        h = t2.groupby(keys).y.apply(lambda s:(s==c).sum()); p=1/3
        var = n*p*(1-p)*(N-n)/(N-1).clip(lower=1); ok=(n>=10)&(N>n)
        print(keys, c, round(((h-n*p)**2/var)[ok].mean(),3), ok.sum())
x = t2.groupby(["user_id","sport"]).agg(n=("y","size"), m=("y","mean")); print(x[x.n>=20].head(20))
