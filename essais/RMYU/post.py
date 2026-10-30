"""Post-traitement d'un cache de probas (cache.py) : copie des doublons exacts + quotas par user.
Usage : python post.py <nom_cache> [--submit chemin.csv]
- doublon exact (16 colonnes identiques, même user) d'une séance étiquetée -> on recopie son label (99,6 % d'accord dans le train)
- doublon partiel (user, sport, durée, distance) -> mélange des probas avec le label du doublon (poids w4)
- doublons entre séances non étiquetées -> probas moyennées dans le groupe
- puis affectation sous quotas (tiers exacts par user, train+test)"""
import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from src.validation import load_data
from quota import constrained_predict, LAB

FULL = ["user_id", "sport", "gender", "year", "month", "weekday", "start_hour_utc", "duration_min", "distance_km",
        "speed_kmh", "max_speed_kmh", "speed_std_kmh", "altitude_mean_m", "altitude_gain_m", "altitude_loss_m", "n_points"]
PART = ["user_id", "sport", "duration_min", "distance_km"]


def keyof(df, cols):
    return df[cols].astype(str).agg("|".join, axis=1).values


def adjust(P, ku_full, ku_part, kl_full, kl_part, yl, w4=0.5, wfull=1.0, share=True):
    P = P.copy()
    for ku, kl, w in [(ku_part, kl_part, w4), (ku_full, kl_full, wfull)]:
        if w <= 0:
            continue
        lab = pd.DataFrame(np.eye(3)[yl], index=kl).groupby(level=0).mean()
        hit = pd.Index(lab.index).get_indexer(ku)
        m = hit >= 0
        P[m] = (1 - w) * P[m] + w * lab.values[hit[m]]
    if share:  # doublons exacts entre séances non étiquetées : même proba
        P = pd.DataFrame(P).groupby(ku_full).transform("mean").values
    return P


def run(name, submit=None, verbose=True):
    train, test, sample = load_data()
    yi = train.intensity.map({"easy": 0, "moderate": 1, "hard": 2}).values
    C = np.load(f"cache_{name}.npz")
    kf_tr, kp_tr, kf_te, kp_te = keyof(train, FULL), keyof(train, PART), keyof(test, FULL), keyof(test, PART)
    variants = {"argmax": None, "quota": dict(w4=0, wfull=0, share=False), "dupfull+quota": dict(w4=0, wfull=1.0),
                "dup(w4=.3)+quota": dict(w4=0.3), "dup(w4=.5)+quota": dict(w4=0.5), "dup(w4=.7)+quota": dict(w4=0.7)}
    res = {k: [] for k in variants}
    for f in range(5):
        va = C[f"va{f}"]; P = C[f"P{f}"]; nv = len(va)
        lab = np.setdiff1d(np.arange(len(train)), va)
        ku_f = np.concatenate([kf_tr[va], kf_te]); ku_p = np.concatenate([kp_tr[va], kp_te])
        gu = np.concatenate([train.user_id.values[va], test.user_id.values])
        for k, v in variants.items():
            if v is None:
                pred = P.argmax(1)
            else:
                Q = adjust(P, ku_f, ku_p, kf_tr[lab], kp_tr[lab], yi[lab], **v)
                pred = constrained_predict(Q, gu, train.user_id.values[lab], yi[lab])
            res[k].append(f1_score(yi[va], pred[:nv], average="macro"))
    if verbose:
        for k, v in res.items():
            print(f"{name:12s} {k:20s} {np.mean(v):.4f} ± {np.std(v):.4f}")
    if submit:
        best = max([k for k in res if k != "argmax"], key=lambda k: np.mean(res[k]))
        Q = adjust(C["Pfull"], kf_te, kp_te, kf_tr, kp_tr, yi, **variants[best])
        pred = constrained_predict(Q, test.user_id.values, train.user_id.values, yi)
        sub = pd.DataFrame({"workout_id": test.workout_id.values, "intensity": LAB[pred]})
        assert len(sub) == len(sample) and (sub.workout_id.values == sample.workout_id.values).all()
        sub.to_csv(submit, index=False); print("écrit", submit, "avec", best, sub.intensity.value_counts().to_dict())
    return res


if __name__ == "__main__":
    sub = sys.argv[sys.argv.index("--submit") + 1] if "--submit" in sys.argv else None
    run(sys.argv[1], sub)
