import sys, numpy as np, trans as T, stack as S
C = dict(np.load(sys.argv[1]))
# référence : même LR mais sans aucune variable de voisinage (calibration seule)
r, _ = S.run(C, groups=(), use_lab=False); T.rep("stack LR logP seul", r)
r, _ = S.run(C, use_lab=False); T.rep("stack +voisins non étiquetés", r)
r, _ = S.run(C, use_lab=False, pool_test_only=True); T.rep("stack +voisins non étiq. (pool test seul)", r)
r, _ = S.run(C, use_lab=True); T.rep("stack +non étiq. +étiquetés", r)
