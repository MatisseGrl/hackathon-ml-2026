"""Audit indépendant (angle CSV) : reconstruit sub_RMYU_blend5_egal_dup_quota_usersport.csv sans blendn/post/quota.
Lancer depuis essais/RMYU/lab2/audit."""
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

R = "../../"
D = "../../../../data/"
tr = pd.read_csv(D + "train.csv"); te = pd.read_csv(D + "test.csv")
ss = pd.read_csv(D + "sample_submission.csv")
sub = pd.read_csv("../../../../submissions/sub_RMYU_blend5_egal_dup_quota_usersport.csv")
LAB = np.array(["easy", "moderate", "hard"])
ymap = {l: i for i, l in enumerate(LAB)}
y = tr.intensity.map(ymap).values
assert not np.isnan(y.astype(float)).any()

# format
print("format: lignes", len(sub), "colonnes", list(sub.columns), "ids==sample", (sub.workout_id.values == ss.workout_id.values).all(),
      "ids==test", (sub.workout_id.values == te.workout_id.values).all(), "labels", sorted(sub.intensity.unique()),
      "NaN", sub.isna().sum().sum(), "ids uniques", sub.workout_id.is_unique)

# (a) moyenne égale
names = ["person", "person7", "f2", "neighbor", "resid2"]
P = np.mean([np.load(f"{R}cache_{n}.npz")["Pfull"] for n in names], axis=0)
print("P somme lignes min/max", P.sum(1).min(), P.sum(1).max())

# (b) doublons
cols = ["user_id", "sport", "gender", "year", "month", "weekday", "start_hour_utc", "duration_min", "distance_km", "speed_kmh",
        "max_speed_kmh", "speed_std_kmh", "altitude_mean_m", "altitude_gain_m", "altitude_loss_m", "n_points"]
def key(df):
    return df[cols].astype(str).agg("|".join, axis=1).values
ktr, kte = key(tr), key(te)
oh = np.eye(3)[y]
dtr = pd.DataFrame(oh, columns=[0, 1, 2]); dtr["k"] = ktr
mtr = dtr.groupby("k")[[0, 1, 2]].mean()
Q = P.copy()
hit = pd.Series(kte).isin(mtr.index).values
Q[hit] = mtr.loc[kte[hit]].values
dte = pd.DataFrame(Q, columns=[0, 1, 2]); dte["k"] = kte
Q = dte.groupby("k")[[0, 1, 2]].transform("mean").values
print("test avec doublon train:", hit.sum(), "; lignes test dans une clé partagée en test:", pd.Series(kte).duplicated(keep=False).sum())

# (c) quotas
gtr = (tr.user_id.astype(str) + "|" + tr.sport.astype(str)).values
gte = (te.user_id.astype(str) + "|" + te.sport.astype(str)).values
L = np.log(np.clip(Q, 1e-9, None))

def solve_ls(Lg, q):
    """recherche locale exacte : échanges 2 classes et rotations 3 classes (graphe résiduel à 3 nœuds)."""
    n = len(Lg)
    # départ glouton faisable
    a = np.empty(n, int); rem = q.copy(); order = np.argsort(-(Lg.max(1) - np.sort(Lg, 1)[:, 1]))
    for i in order:
        for c in np.argsort(-Lg[i]):
            if rem[c] > 0: a[i] = c; rem[c] -= 1; break
    for _ in range(100000):
        # gain de déplacer i de a[i] vers c
        G = Lg - Lg[np.arange(n), a][:, None]
        best = {}
        for i0 in range(3):
            for j0 in range(3):
                if i0 == j0: continue
                m = np.where(a == i0)[0]
                if len(m) == 0: continue
                k = m[np.argmax(G[m, j0])]; best[(i0, j0)] = (G[k, j0], k)
        mv = None; bg = 1e-12
        for (i0, j0), (g1, k1) in best.items():
            if (j0, i0) in best and i0 < j0:
                g2, k2 = best[(j0, i0)]
                if g1 + g2 > bg: bg = g1 + g2; mv = [(k1, j0), (k2, i0)]
        for cyc in [(0, 1, 2), (0, 2, 1)]:
            e = [(cyc[0], cyc[1]), (cyc[1], cyc[2]), (cyc[2], cyc[0])]
            if all(x in best for x in e):
                g = sum(best[x][0] for x in e)
                if g > bg: bg = g; mv = [(best[x][1], x[1]) for x in e]
        if mv is None: break
        for k, c in mv: a[k] = c
    return a

def solve_lsa(Lg, q):
    C = -np.repeat(Lg, q, axis=1); r, c = linear_sum_assignment(C)
    lab = np.repeat(np.arange(3), q); a = np.empty(len(Lg), int); a[r] = lab[c]; return a

fileA = sub.intensity.map(ymap).values
pred = np.full(len(te), -1); pred2 = pred.copy()
ngroups = 0; bad = 0; diffs = []
for g in np.unique(gte):
    it = np.where(gte == g)[0]
    ntr_g = (gtr == g).sum(); N = ntr_g + len(it)
    tgt = np.array([N // 3, 2 * N // 3 - N // 3, N - 2 * N // 3])
    known = np.bincount(y[gtr == g], minlength=3)
    q = tgt - known
    if (q < 0).any() or q.sum() != len(it): bad += 1; print("groupe incohérent", g, tgt, known, len(it)); continue
    Lg = L[it]
    a = solve_ls(Lg, q); b = solve_lsa(Lg, q)
    pred[it] = a; pred2[it] = b; ngroups += 1
    fa = fileA[it]
    oa, ob, of = Lg[np.arange(len(it)), a].sum(), Lg[np.arange(len(it)), b].sum(), Lg[np.arange(len(it)), fa].sum()
    fq = np.bincount(fa, minlength=3)
    diffs.append((g, len(it), (a != fa).sum(), (b != fa).sum(), oa, ob, of, (fq == q).all()))
print("groupes", ngroups, "incohérents", bad)
d = pd.DataFrame(diffs, columns=["g", "n", "dif_ls", "dif_lsa", "obj_ls", "obj_lsa", "obj_file", "quota_file_ok"])
print("lignes différentes fichier vs LS:", (pred != fileA).sum(), "; vs LSA:", (pred2 != fileA).sum(), "; LS vs LSA:", (pred != pred2).sum())
print("groupes où quotas du fichier != quotas recalculés:", (~d.quota_file_ok).sum())
print("écart max obj (LS - LSA):", (d.obj_ls - d.obj_lsa).abs().max(), "; max(opt - obj_file):", (np.maximum(d.obj_ls, d.obj_lsa) - d.obj_file).max())
w = d[(d.dif_ls > 0) | (d.dif_lsa > 0)]
print("groupes avec différences:", len(w)); print(w.head(30).to_string())
np.savez("rebuild_out.npz", P=P, Q=Q, pred_ls=pred, pred_lsa=pred2, fileA=fileA)
mine = pd.DataFrame({"workout_id": te.workout_id.values, "intensity": LAB[pred]})
mine.to_csv("rebuild_ls.csv", index=False)
print("répartition fichier", sub.intensity.value_counts().to_dict(), "; mienne", mine.intensity.value_counts().to_dict())
