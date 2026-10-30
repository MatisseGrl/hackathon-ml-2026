import sys, os
os.chdir(r"C:\Users\curie\hk\essais\RMYU")
sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
import blendn as B
D = pd.read_pickle("lab2/a5/oof_ref.pkl")
tr, te = B.train.reset_index(drop=True), B.test.reset_index(drop=True)
A = pd.concat([tr.drop(columns=["intensity"]), te], ignore_index=True)
ntr = len(tr)
A["us"] = A.user_id.astype(str) + "_" + A.sport.astype(str)
A["N"] = A.groupby("us").us.transform("size")
A["t"] = A.year * 12 + A.month
A["trank"] = A.groupby("user_id").t.rank(pct=True)
for c in ["speed_kmh", "duration_min", "distance_km"]:
    lo = A.groupby("sport")[c].transform(lambda x: x.quantile(0.01)); hi = A.groupby("sport")[c].transform(lambda x: x.quantile(0.99))
    A[c + "_ext"] = (A[c] < lo) | (A[c] > hi)
A["alt_bad"] = (A.altitude_mean_m < -100) | (A.altitude_mean_m > 5000)
A["gain_bad"] = A.altitude_gain_m > 5000
A["np500"] = A.n_points >= 500
A["route"] = A.us + "_" + A.distance_km.round(1).astype(str) + "_" + A.altitude_mean_m.round(-1).astype(str)
A["route_n"] = A.groupby("route").route.transform("size")
T = A.iloc[:ntr].copy()
T = T.join(D.set_index("idx"))
T["err"] = (T.pred != T.y); T["err2"] = (T.pred - T.y).abs() == 2
fold = T.fold.values
# labelled neighbours from other folds (same user)
def nb_count(keys):
    k = T[keys].astype(str).agg("_".join, axis=1).values
    out = np.zeros(ntr)
    for f in range(5):
        m = fold != f
        vc = pd.Series(k[m]).value_counts()
        out[fold == f] = pd.Series(k[fold == f]).map(vc).fillna(0).values
    return out
T["nb_ymwh"] = nb_count(["user_id", "year", "month", "weekday", "start_hour_utc"])
T["nb_ym"] = nb_count(["user_id", "year", "month"])
T["nb_part"] = nb_count(["user_id", "sport", "duration_min", "distance_km"])
T.to_pickle("lab2/a5/T.pkl")
base = T.err.mean()
print(f"err global {base:.4f}  err2 {T.err2.mean():.4f}")
def show(name, seg):
    g = T.groupby(seg)
    r = pd.DataFrame(dict(n=g.size(), err=g.err.mean(), err2=g.err2.mean(), share_err=g.err.sum() / T.err.sum()))
    print("==", name); print(r.round(3).to_string())
show("N", pd.cut(T.N, [0, 5, 10, 20, 50, 100, 200, 2000]))
show("sport", T.sport)
show("alt_bad", T.alt_bad); show("gain_bad", T.gain_bad); show("np500", T.np500)
show("speed_ext", T.speed_kmh_ext); show("dur_ext", T.duration_min_ext); show("dist_ext", T.distance_km_ext)
show("nb_ymwh", pd.cut(T.nb_ymwh, [-1, 0, 1, 3, 100]))
show("nb_ym", pd.cut(T.nb_ym, [-1, 0, 2, 5, 10, 1000]))
show("trank", pd.cut(T.trank, [0, .05, .2, .5, .8, .95, 1]))
show("pmax", pd.qcut(T.pmax, 5))
show("Estd", pd.qcut(T.Estd, 5))
show("nb_part", pd.cut(T.nb_part, [-1, 0, 1, 100]))
show("route_n", pd.cut(T.route_n, [0, 1, 2, 5, 1000]))
show("y", T.y)
print(pd.crosstab(T.y, T.pred))
