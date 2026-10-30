# Hackathon ML 2026 : intensité des séances

Prédire `intensity` (`easy` / `moderate` / `hard`) pour chaque séance de `test.csv`.
Métrique : **macro-F1**. Soumission : `submission.csv` avec les colonnes `workout_id,intensity`.

## Règles (à relire avant de coder)
- ✅ pandas, NumPy, matplotlib, **tout modèle scikit-learn qui n'est pas un réseau de neurones**
  (régression logistique, arbres, random forest, extra trees, HistGradientBoosting, kNN, SVM, Bayes naïf, vote, stacking…)
- ❌ deep learning (y compris `MLPClassifier`), XGBoost/LightGBM/CatBoost (hors scikit-learn), données externes, partage de code entre équipes
- Chaque modèle soumis doit pouvoir être **expliqué sur papier**.
- **15 soumissions Kaggle maximum** pour l'équipe → une ligne dans `LOG.md` par soumission.
- Le dépôt reste **privé** (il contient les données Kaggle, `data/`).
- Règles complètes de l'équipe et des assistants IA : **`AGENTS.md`** (lu automatiquement par Claude Code et Codex).

## Installation (5 min)
```bash
git clone <URL_DU_DEPOT>
cd hackathon-ml-2026
pip install -r requirements.txt
```
Les données Kaggle (`train.csv`, `test.csv`, `sample_submission.csv`) sont déjà dans `data/` : rien à télécharger.

## Organisation
```
data/                 # CSV Kaggle (versionnés, dépôt privé)
src/validation.py     # validation commune : mêmes folds, même graine, macro-F1
notebook_equipe.ipynb # notebook final rendu sur Boostcamp
essais/<code>/        # un dossier par personne (code participant, pas de nom)
cours/                # notebooks du cours (classification, régression, clustering…), lecture seule
submissions/          # fichiers soumis sur Kaggle (versionnés)
conversations/        # conversations IA, un fichier <login_github>.md par personne
LOG.md                # journal des soumissions et des expériences
```

**Règle d'or** : tout score de validation est calculé avec `src/validation.py`, sinon les scores ne sont pas comparables.

```python
from src.validation import evaluate
res = evaluate(model, X, y)       # 5 folds stratifiés, graine 42
print(res["mean"], res["std"])
```

## Git en 5 commandes
```bash
git pull                          # AVANT de commencer à travailler
git add <tes fichiers précis>     # jamais "git add ." ni "git add -A"
git commit -m "kNN k=15, val macro-F1 0.61"
git pull                          # récupère le travail des autres
git push
```
- Tout le monde peut modifier n'importe quel fichier (sauf `cours/`), sans effacer le travail des autres. Pour tester une idée sans risque, utilise `essais/<ton_code>/`.
- Une seule personne à la fois modifie `notebook_equipe.ipynb` (on se met d'accord à l'oral).
- Les humains travaillent sur `main`. Les assistants IA travaillent sur une branche `<login_github>/<sujet>` et fusionnent leur pull request après ton accord (voir `AGENTS.md`).
- Conflit sur un notebook ? Ne force rien, appelle la personne qui gère le dépôt.

## Rendu (18h10 → 18h20)
- [ ] Kaggle : la soumission finale de l'équipe est **sélectionnée**
- [ ] Boostcamp Code : `notebook_equipe.ipynb` + `LOG.md` (chaque membre dépose les mêmes fichiers)
- [ ] Boostcamp Conversations IA : liens ou fichiers + déclaration
