"""Effet du nettoyage sur HGB, évalué avec la validation commune."""
import sys
sys.path.append("../..")
sys.path.append(".")
from sklearn.ensemble import HistGradientBoostingClassifier
from src.validation import load_data, evaluate
from cleaning import clean

train, test, _ = load_data()
y = train["intensity"]
drop = ["workout_id", "intensity"]
X_raw = train.drop(columns=drop)
X_cln = clean(train).drop(columns=drop)

model = HistGradientBoostingClassifier(random_state=42, categorical_features="from_dtype")
for name, X in [("brut", X_raw), ("nettoyé", X_cln)]:
    X = X.copy()
    for c in X.select_dtypes("object").columns.union(X.select_dtypes("string").columns):
        X[c] = X[c].astype("category")
    print(name); evaluate(model, X, y)
