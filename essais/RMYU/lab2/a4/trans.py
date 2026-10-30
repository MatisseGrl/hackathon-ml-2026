"""a4 : post-traitement transductif. Avant les quotas, on lisse les probas d'une séance vers
(1) la moyenne des probas des séances NON étiquetées voisines du même groupe (user, sport) et
(2) éventuellement la moyenne des labels connus des voisines étiquetées (option beta).
Voisinage = même (year, month) / même (year, month, weekday, heure) / même parcours (distance et altitude arrondies),
pondéré par niveau. La ligne elle-même est exclue. Aucun label de la ligne évaluée ni de son pli n'est utilisé."""
import numpy as np, pandas as pd

LEVELS = {"month": ["year", "month"], "slot": ["year", "month", "weekday", "start_hour_utc"], "route": ["_dr", "_ar"]}


def level_keys(df):
    d = df[["user_id", "sport", "year", "month", "weekday", "start_hour_utc"]].copy()
    d["_dr"] = np.round(df.distance_km.values * 2) / 2          # distance arrondie à 0,5 km
    d["_ar"] = np.round(df.altitude_mean_m.values / 20)          # altitude moyenne arrondie à 20 m
    return {k: pd.util.hash_pandas_object(d[["user_id", "sport"] + c], index=False).values for k, c in LEVELS.items()}


def nb_sum(keys, wts, rows_set, V, rows_q):
    """Somme pondérée (par niveau) des V des lignes rows_set partageant la clé de chaque ligne de rows_q."""
    num = np.zeros((len(rows_q), V.shape[1])); den = np.zeros(len(rows_q))
    for lev, wl in wts.items():
        if wl == 0:
            continue
        ks = keys[lev][rows_set]; kq = keys[lev][rows_q]
        S = pd.DataFrame(V).groupby(ks).sum()
        n = pd.Series(np.ones(len(ks))).groupby(ks).sum()
        idx = S.index.get_indexer(kq); m = idx >= 0
        num[m] += wl * S.values[idx[m]]; den[m] += wl * n.reindex(S.index).values[idx[m]]
    return num, den


def smooth(Qq, keys, rows_q, U_rows, U_V, L_rows, L_y, wts, alpha, beta=0.0, fixed=None):
    """Qq : probas des lignes à classer (indices rows_q dans la table train+test concaténée).
    U_rows/U_V : séances non étiquetées (dont rows_q elles-mêmes) et leurs probas ; L_rows/L_y : étiquetées.
    Renvoie (1-a-b) Q + a * moyenne(U voisines, hors soi) + b * moyenne(labels L voisines)."""
    out = Qq.copy()
    if alpha != 0:
        num, den = nb_sum(keys, wts, U_rows, U_V, rows_q)
        pos = pd.Index(U_rows).get_indexer(rows_q); me = pos >= 0      # retirer la ligne elle-même
        tw = sum(wts.values())
        num[me] -= tw * U_V[pos[me]]; den[me] -= tw
        m = den > 1e-9
        out[m] += alpha * (num[m] / den[m, None] - Qq[m])
    if beta != 0:
        num, den = nb_sum(keys, wts, L_rows, np.eye(3)[L_y], rows_q)
        m = den > 1e-9
        out[m] += beta * (num[m] / den[m, None] - Qq[m])
    if fixed is not None:
        out[fixed] = Qq[fixed]
    return out


def nb_wsum(keys, wts, rows_set, V, r, rows_q):
    """Comme nb_sum, avec un poids r_j par ligne voisine (confiance)."""
    num = np.zeros((len(rows_q), V.shape[1])); den = np.zeros(len(rows_q))
    for lev, wl in wts.items():
        if wl == 0:
            continue
        ks = keys[lev][rows_set]; kq = keys[lev][rows_q]
        S = pd.DataFrame(V * r[:, None]).groupby(ks).sum()
        n = pd.Series(r).groupby(ks).sum().reindex(S.index)
        idx = S.index.get_indexer(kq); m = idx >= 0
        num[m] += wl * S.values[idx[m]]; den[m] += wl * n.values[idx[m]]
    return num, den


def pseudo_iter(Qq, keys, rows_q, L_rows, L_y, wts, gamma, assign_fn, n_iter=2, fixed=None, extra=None):
    """Itération pseudo-labels : quotas -> classes affectées = pseudo-labels (poids = proba de la classe affectée),
    label moyen des voisines (étiquetées poids 1 + pseudo-étiquetées, hors soi) mélangé aux probas -> quotas.
    assign_fn(Q) renvoie les classes sous quotas. extra = (rows, onehot, poids) d'autres pseudo-étiquetées (test)."""
    tw = sum(wts.values())
    pred = assign_fn(Qq); hist = [pred]
    for _ in range(n_iter):
        conf = Qq[np.arange(len(Qq)), pred]
        rows = [L_rows, rows_q]; V = [np.eye(3)[L_y], np.eye(3)[pred]]; r = [np.ones(len(L_rows)), conf]
        if extra is not None:
            rows.append(extra[0]); V.append(extra[1]); r.append(extra[2])
        rows = np.concatenate(rows); V = np.vstack(V); r = np.concatenate(r)
        num, den = nb_wsum(keys, wts, rows, V, r, rows_q)
        num -= tw * conf[:, None] * np.eye(3)[pred]; den -= tw * conf      # retirer soi-même
        Q = Qq.copy(); m = den > 1e-9
        Q[m] += gamma * (num[m] / den[m, None] - Qq[m])
        if fixed is not None:
            Q[fixed] = Qq[fixed]
        pred = assign_fn(Q); hist.append(pred)
    return hist
