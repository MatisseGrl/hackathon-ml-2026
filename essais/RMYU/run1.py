import sys; sys.path.append("../.."); sys.path.append(".")
from sklearn.ensemble import HistGradientBoostingClassifier
from src.validation import load_data, evaluate
from feats import build
train,test,_ = load_data()
F = build(train,test)
Xtr = F[F.part=="tr"].drop(columns=["part","workout_id"]).reset_index(drop=True)
y = train.intensity
m = HistGradientBoostingClassifier(random_state=42, learning_rate=0.05, max_iter=300, categorical_features="from_dtype")
r = evaluate(m, Xtr, y, verbose=False); print("HGB all rel:", round(r["mean"],4), round(r["std"],4))
