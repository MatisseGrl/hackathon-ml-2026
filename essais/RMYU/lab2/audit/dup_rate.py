import sys; sys.path.append("../../../.."); sys.path.append("../..")
import numpy as np, pandas as pd
from src.validation import load_data, get_folds
from post import keyof, FULL
train, test, _ = load_data()
y = train.intensity.values
ktr, kte = keyof(train, FULL), keyof(test, FULL)
s_tr = set(ktr)
print("test : part des lignes avec doublon exact dans tout le train = %.4f" % np.mean([k in s_tr for k in kte]))
for f, (tr, va) in enumerate(get_folds(y).split(np.zeros(len(y)), y)):
    st = set(ktr[tr])
    print(f"val {f}: part avec doublon exact dans le pli d'entrainement = %.4f" % np.mean([k in st for k in ktr[va]]))
