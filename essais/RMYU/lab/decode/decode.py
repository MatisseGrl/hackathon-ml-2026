"""Décodage sous contrainte des tiers exacts par user.

decode(P, groups_unl, groups_lab, y_lab, rule="ceil", method="lsa_log", a=1.0)
  P          : (n,3) probas des lignes non étiquetées (easy, moderate, hard)
  groups_unl : clé de groupe (user_id) des lignes non étiquetées
  groups_lab : clé de groupe des lignes étiquetées ; y_lab : leurs labels 0/1/2
  rule       : répartition des unités en trop quand N = n_lab + n_unl n'est pas multiple de 3
               "orig" = quota.remaining_counts (équivalent à floor : +1 easy si N%3=1, +1 easy/moderate si N%3=2)
               "ceil" = rang en pourcentage, bins (0,1/3],(1/3,2/3],(2/3,1] -> (k,k,k+1) / (k,k+1,k+1)
               "mix"  = incertitude : mélange des répartitions plausibles (pour les méthodes marginales)
  method     : "lsa_log"  affectation sous quotas, coût -log P (= configuration la plus probable, MAP)
               "lsa_lin"  affectation sous quotas, coût -P^a
               "marg_argmax" marginales exactes sachant les quotas (DP) puis argmax (max d'exactitude attendue)
               "marg_lsa" marginales exactes puis affectation sous quotas avec coût -marginale
               "sink_argmax"/"sink_lsa" : idem avec marginales approchées (Sinkhorn / inclinaison exponentielle)
  a          : puissance appliquée à P avant calcul des marginales (calibration / température)
"""
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

EXTRA = {"floor": {0: (0, 0, 0), 1: (1, 0, 0), 2: (1, 1, 0)},
         "ceil": {0: (0, 0, 0), 1: (0, 0, 1), 2: (0, 1, 1)},
         "qcut": {0: (0, 0, 0), 1: (1, 0, 0), 2: (1, 0, 1)},
         "mid": {0: (0, 0, 0), 1: (0, 1, 0), 2: (1, 0, 1)}}
MIX = {0: [(0, 0, 0)], 1: [(0, 0, 1), (0, 1, 0), (1, 0, 0)], 2: [(0, 1, 1), (1, 1, 0), (1, 0, 1)]}
MIXW = {0: [1.0], 1: [0.6, 0.2, 0.2], 2: [0.4, 0.4, 0.2]}


def _orig_counts(n_total, known, m):
    target = np.full(3, n_total / 3.0)
    rem = np.clip(target - known, 0, None)
    base = np.floor(rem).astype(int)
    left = m - base.sum()
    frac = rem - base
    order = np.argsort(-frac, kind="stable") if left >= 0 else np.argsort(frac, kind="stable")
    i = 0
    while left != 0:
        c = order[i % 3]
        if left > 0:
            base[c] += 1; left -= 1
        elif base[c] > 0:
            base[c] -= 1; left += 1
        i += 1
    return base


def _fix(rem, m):
    rem = np.clip(np.asarray(rem, int), 0, None)
    while rem.sum() > m:
        rem[np.argmax(rem)] -= 1
    while rem.sum() < m:
        rem[np.argmin(rem)] += 1
    return rem


def candidate_quotas(n_total, known, m, rule):
    """liste de (quota restant, poids)"""
    k, r = divmod(int(n_total), 3)
    if rule == "orig":
        return [(_orig_counts(n_total, known, m), 1.0)]
    if rule == "mix":
        out = {}
        for e, w in zip(MIX[r], MIXW[r]):
            q = k + np.array(e) - known
            if (q < 0).any():
                continue
            out[tuple(q)] = out.get(tuple(q), 0) + w
        if not out:
            return [(_fix(k + np.array(EXTRA["ceil"][r]) - known, m), 1.0)]
        s = sum(out.values())
        return [(np.array(q), w / s) for q, w in out.items()]
    return [(_fix(k + np.array(EXTRA[rule][r]) - known, m), 1.0)]


def lsa(cost, counts):
    cols = np.concatenate([np.full(int(k), c) for c, k in enumerate(counts)]).astype(int)
    r, c = linear_sum_assignment(cost[:, cols])
    out = np.empty(len(cost), dtype=int); out[r] = cols[c]
    return out


def exact_marginals(P, qlist):
    """marginales P(y_i=c | comptes du groupe = q) pour un mélange de quotas q (poids w), P indépendantes.
    DP avant/arrière sur les comptes (n0, n1)."""
    m = len(P)
    A = max(q[0] for q, _ in qlist) + 1; B = max(q[1] for q, _ in qlist) + 1
    F = np.zeros((m + 1, A, B)); F[0, 0, 0] = 1.0
    for i in range(m):
        f = F[i]; g = f * P[i, 2]
        g[1:, :] += f[:-1, :] * P[i, 0]
        g[:, 1:] += f[:, :-1] * P[i, 1]
        s = g.sum(); F[i + 1] = g / (s if s > 0 else 1)
    M = np.zeros((m, 3))
    Bk = np.zeros((A, B)); Bk[0, 0] = 1.0   # backward: proba des rangs i+1..m-1 d'avoir comptes (r0, r1)
    E = [(1, 0), (0, 1), (0, 0)]
    for i in range(m - 1, -1, -1):
        f = F[i]
        for c in range(3):
            tot = 0.0
            for q, w in qlist:
                t0, t1 = q[0] - E[c][0], q[1] - E[c][1]
                if t0 < 0 or t1 < 0 or (q[2] - (c == 2)) < 0:
                    continue
                # sum_a f[a0,a1] * Bk[t0-a0, t1-a1]
                sub_f = f[:t0 + 1, :t1 + 1]
                sub_b = Bk[t0::-1, t1::-1][:t0 + 1, :t1 + 1] if True else None
                tot += w * (sub_f * sub_b).sum()
            M[i, c] = P[i, c] * tot
        s = M[i].sum(); M[i] = M[i] / s if s > 0 else P[i]
        # update backward with row i
        g = Bk * P[i, 2]
        g[1:, :] += Bk[:-1, :] * P[i, 0]
        g[:, 1:] += Bk[:, :-1] * P[i, 1]
        s = g.sum(); Bk = g / (s if s > 0 else 1)
    return M


def sinkhorn_marginals(P, q, it=200):
    q = np.asarray(q, float)
    lam = np.ones(3)
    for _ in range(it):
        Q = P * lam; Q /= Q.sum(1, keepdims=True)
        cs = Q.sum(0)
        lam *= np.where(q > 0, q / np.maximum(cs, 1e-12), 1e-6)
    Q = P * lam; Q /= Q.sum(1, keepdims=True)
    return Q


def decode(P, groups_unl, groups_lab, y_lab, rule="ceil", method="lsa_log", a=1.0, dp_max=350,
           return_probs=False):
    P = np.clip(np.asarray(P, float), 1e-9, 1)
    pred = P.argmax(1).copy(); PM = P.copy()
    gl = pd.Series(np.asarray(groups_lab))
    known = pd.crosstab(gl, pd.Series(np.asarray(y_lab))).reindex(columns=[0, 1, 2], fill_value=0)
    nl = gl.value_counts()
    gu = pd.Series(np.asarray(groups_unl))
    for g, rows in gu.groupby(gu).groups.items():
        rows = np.asarray(rows)
        kc = known.loc[g].values if g in known.index else np.zeros(3, int)
        n_total = len(rows) + nl.get(g, 0)
        qlist = candidate_quotas(n_total, kc, len(rows), rule)
        Pg = P[rows]
        if method == "lsa_log":
            q = max(qlist, key=lambda t: t[1])[0]
            pred[rows] = lsa(-np.log(Pg), q); continue
        if method.startswith("rank"):   # tri par un score ordinal puis découpe aux quotas
            q = max(qlist, key=lambda t: t[1])[0]
            Pa = Pg ** a; Pa /= Pa.sum(1, keepdims=True)
            if method == "rank_sink":
                Pa = sinkhorn_marginals(Pa, q)
            if method == "rank_lr":
                sc = np.log(Pa[:, 2]) - np.log(Pa[:, 0])
            elif method == "rank_d":
                sc = Pa[:, 2] - Pa[:, 0]
            else:
                sc = Pa[:, 1] + 2 * Pa[:, 2]
            o = np.argsort(sc, kind="stable"); lab_ = np.repeat([0, 1, 2], q)
            pr = np.empty(len(rows), int); pr[o] = lab_; pred[rows] = pr; PM[rows] = Pa; continue
        if method == "lsa_lin":
            q = max(qlist, key=lambda t: t[1])[0]
            pred[rows] = lsa(-Pg ** a, q); continue
        Pa = Pg ** a; Pa /= Pa.sum(1, keepdims=True)
        if method.startswith("marg") and len(rows) <= dp_max:
            M = exact_marginals(Pa, qlist)
        else:
            M = sum(w * sinkhorn_marginals(Pa, q) for q, w in qlist)
        PM[rows] = M
        if method.endswith("argmax"):
            pred[rows] = M.argmax(1)
        else:
            q = max(qlist, key=lambda t: t[1])[0]
            pred[rows] = lsa(-M, q)
    return (pred, PM) if return_probs else pred


def f1opt(M, it=20):
    """décision maximisant l'espérance (approchée) de la macro-F1 à partir de marginales M :
    pred_i = argmax_c (M_ic - F1_c/2) / D_c, D_c = n_pred_c + n_true_c attendus (point fixe)."""
    pred = M.argmax(1)
    for _ in range(it):
        oh = np.eye(3)[pred]
        S = (oh * M).sum(0); D = oh.sum(0) + M.sum(0); F = 2 * S / D
        new = ((M - F / 2) / D).argmax(1)
        if (new == pred).all():
            break
        pred = new
    return pred
