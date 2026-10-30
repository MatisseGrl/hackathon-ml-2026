import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
import blendn as B
gtr, gte = B.group_keys("user_sport")
K = pd.crosstab(pd.Series(gtr), pd.Series(B.yi)).reindex(columns=[0, 1, 2], fill_value=0)
nte = pd.Series(gte).value_counts()
T = {g: B.targets(int(nte[g] + (K.loc[g].sum() if g in K.index else 0)), "pct") - (K.loc[g].values if g in K.index else 0) for g in nte.index}
fs = ["sub_RMYU_blend5_egal_dup_quota_usersport.csv", "sub_RMYU_blend_person06_final2_dup_quota_numpy.csv",
      "sub_RMYU_blend_person06_final2_dup_quota.csv", "sub_03_person_stack.csv"]
S = {f: pd.read_csv(f"../../submissions/{f}").intensity.map({"easy": 0, "moderate": 1, "hard": 2}).values for f in fs}
for f in fs:
    c = pd.crosstab(pd.Series(gte), pd.Series(S[f])).reindex(columns=[0, 1, 2], fill_value=0)
    ok = sum((c.loc[g].values == T[g]).all() for g in nte.index)
    dev = sum(np.abs(c.loc[g].values - T[g]).sum() for g in nte.index) // 2
    print(f"{f:55s} groupes user+sport aux quotas exacts {ok}/{len(nte)} ; lignes hors quota {dev} ; accord avec blend5 {(S[f] == S[fs[0]]).mean():.4f} ({(S[f] != S[fs[0]]).sum()} lignes)")
