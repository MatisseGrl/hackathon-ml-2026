import sys; sys.path.append("../.."); sys.path.append(".")
from src.validation import load_data, evaluate
from feats import build, add_resid
from knnlab import KnnLabel
train,test,_ = load_data()
F = add_resid(build(train,test))
Xtr = F[F.part=="tr"].drop(columns=["part","workout_id"]).reset_index(drop=True)
y = train.intensity
for cols in [("speed_kmh_z_us","distance_km_z_us","gain_per_km_z_us","t"), ("resid_spd","ldist","t"), ("speed_kmh_z_us","max_speed_kmh_z_us","distance_km_z_us","altitude_gain_m_z_us","spd_ratio_z_us","t")]:
    r = evaluate(KnnLabel(cols=cols), Xtr, y, verbose=False); print(cols, round(r["mean"],4), round(r["std"],4))
