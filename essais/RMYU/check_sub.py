"""Vérification indépendante d'un CSV de soumission : format, quotas exacts par user + sport (train + prédits = tiers de la
règle N//3, 2N//3 - N//3, reste), doublons exacts recopiés. Usage : python check_sub.py ../../submissions/<fichier>.csv"""
import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from src.validation import load_data
train, test, sample = load_data()
sub = pd.read_csv(sys.argv[1])
assert sub.shape == (7669, 2) and list(sub.columns) == ["workout_id", "intensity"], sub.shape
assert (sub.workout_id.values == test.workout_id.values).all() and (sub.workout_id.values == sample.workout_id.values).all()
assert set(sub.intensity) <= {"easy", "moderate", "hard"}
cls = ["easy", "moderate", "hard"]
allg = pd.concat([train[["user_id", "sport", "intensity"]], test[["user_id", "sport"]].assign(intensity=sub.intensity.values)])
ct = pd.crosstab([allg.user_id, allg.sport], allg.intensity).reindex(columns=cls, fill_value=0)
N = ct.sum(1).values
target = np.stack([N // 3, 2 * N // 3 - N // 3, N - 2 * N // 3], 1)
print("groupes user+sport :", len(ct), "; groupes où train + prédits ≠ tiers exacts :", int((ct.values != target).any(1).sum()))
FULL = ["user_id", "sport", "gender", "year", "month", "weekday", "start_hour_utc", "duration_min", "distance_km", "speed_kmh",
        "max_speed_kmh", "speed_std_kmh", "altitude_mean_m", "altitude_gain_m", "altitude_loss_m", "n_points"]
k = lambda d: d[FULL].astype(str).agg("|".join, axis=1)
lab = train.assign(key=k(train)).groupby("key").intensity.agg(lambda s: s.mode().iloc[0] if s.nunique() == 1 else None)
kt = k(test); hit = kt.map(lab)
m = hit.notna()
print("séances du test avec doublon exact au label unanime :", int(m.sum()), "; recopiées à l'identique :", int((hit[m].values == sub.intensity.values[m]).sum()))
print("répartition :", sub.intensity.value_counts().to_dict())
