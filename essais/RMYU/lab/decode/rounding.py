import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from src.validation import load_data
tr, te, _ = load_data()
y = tr.intensity.map({"easy":0,"moderate":1,"hard":2})
kc = pd.crosstab(tr.user_id, y).reindex(columns=[0,1,2], fill_value=0)
ntr = tr.user_id.value_counts(); nte = te.user_id.value_counts()
N = (ntr.add(nte, fill_value=0)).reindex(kc.index).astype(int)
print("users", len(kc), "test-only users", len(set(te.user_id)-set(tr.user_id)))
print("N mod 3 dist", (N%3).value_counts().to_dict())
pats = {1: [(1,0,0),(0,1,0),(0,0,1)], 2: [(1,1,0),(1,0,1),(0,1,1)]}
for r in [0,1,2]:
    sel = N%3==r
    k = N[sel]//3
    C = kc[sel].values
    if r==0:
        viol = (C > k.values[:,None]).any(1)
        print("r=0 users", sel.sum(), "violations", viol.sum())
        continue
    for p in pats[r]:
        T = k.values[:,None] + np.array(p)[None,:]
        viol = (C > T).any(1)
        # informative: users where a class count reaches k+1 in train
        print(f"r={r} pattern {p}: users {sel.sum()} violations {viol.sum()}")
    # how many users have train count == k+1 in each class
    hit = (C == k.values[:,None]+1)
    print("   users with train count k+1 per class", hit.sum(0))
# user+sport exactness
g = tr.user_id.astype(str)+"|"+tr.sport
gt = te.user_id.astype(str)+"|"+te.sport
kc2 = pd.crosstab(g, y).reindex(columns=[0,1,2], fill_value=0)
N2 = g.value_counts().add(gt.value_counts(), fill_value=0).reindex(kc2.index).astype(int)
v2 = (kc2.values > np.ceil(N2.values/3)[:,None]).any(1)
print("user+sport groups", len(kc2), "violations of ceil(N/3):", v2.sum(), "multi-sport users", (tr.groupby('user_id').sport.nunique()>1).sum())
# overall counts
print("train class counts", np.bincount(y))
