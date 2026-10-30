import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from src.validation import load_data, get_folds
from feats import build, add_resid
from quota import remaining_counts
train,test,_ = load_data()
F = add_resid(build(train,test))
F["t_rk_u"] = F.groupby("user_id").t.rank(pct=True)
F["t_rk_us"] = F.groupby(["user_id","sport"], observed=True).t.rank(pct=True)
F["yr_rel"] = F.t - F.groupby("user_id").t.transform("min")
F["yr_to_end"] = F.groupby("user_id").t.transform("max") - F.t
X = F.drop(columns=["part","workout_id"]).reset_index(drop=True)
ntr = (F.part=="tr").sum(); Xtr, Xte = X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True)
y = train.intensity.reset_index(drop=True); yi = y.map({"easy":0,"moderate":1,"hard":2}).values
key = lambda d: (d.user_id.astype(str)+"_"+d.sport.astype(str)).values

def sort_assign(s, g_unl, g_lab, y_lab):
    pred = np.zeros(len(s), int); gl = pd.Series(g_lab)
    known = pd.crosstab(gl, pd.Series(y_lab)).reindex(columns=[0,1,2], fill_value=0); nl = gl.value_counts()
    gu = pd.Series(g_unl)
    for g, rows in gu.groupby(gu).groups.items():
        rows = np.asarray(rows); kc = known.loc[g].values if g in known.index else np.zeros(3)
        c = remaining_counts(len(rows)+nl.get(g,0), kc, len(rows))
        o = rows[np.argsort(s[rows])]
        pred[o[:c[0]]] = 0; pred[o[c[0]:c[0]+c[1]]] = 1; pred[o[c[0]+c[1]:]] = 2
    return pred

models = {"reg": HistGradientBoostingRegressor(random_state=42, learning_rate=0.03, max_iter=800, categorical_features="from_dtype"),
          "clf": HistGradientBoostingClassifier(random_state=42, learning_rate=0.03, max_iter=600, categorical_features="from_dtype")}
for name, mdl in models.items():
    a, q = [], []
    for tr_i, va_i in get_folds(y).split(Xtr, y):
        m = clone(mdl).fit(Xtr.iloc[tr_i], yi[tr_i])
        Xu = pd.concat([Xtr.iloc[va_i], Xte]).reset_index(drop=True); nv=len(va_i)
        s = m.predict(Xu) if name=="reg" else m.predict_proba(Xu) @ np.array([0,1,2])
        a.append(f1_score(yi[va_i], np.clip(np.rint(s[:nv]),0,2), average="macro"))
        q.append(f1_score(yi[va_i], sort_assign(s, key(Xu), key(Xtr.iloc[tr_i]), yi[tr_i])[:nv], average="macro"))
    print(name, "round", round(np.mean(a),4), "| quota-sort", round(np.mean(q),4), "±", round(np.std(q),4))
