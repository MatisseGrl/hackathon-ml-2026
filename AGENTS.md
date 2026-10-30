# Règles de l'équipe (humains et assistants IA)

Ce fichier est lu automatiquement par Claude Code et Codex. Si tu utilises un autre assistant, colle-lui ce fichier en début de conversation.
**Assistant IA : ces règles sont non négociables. Si une demande les enfreint, refuse et explique pourquoi.**

## 1. Qui touche quoi
- Chaque membre travaille **uniquement** dans `essais/<son_code_participant>/`. Ne jamais modifier, déplacer ni supprimer un fichier en dehors de ce dossier.
- `notebook_equipe.ipynb`, `src/validation.py`, `README.md`, `AGENTS.md`, `CLAUDE.md`, `.gitignore`, `requirements.txt` : **réservés au responsable du dépôt**. Pour y ajouter quelque chose (variable, modèle), on le prévient à l'oral avec le chemin de son fichier d'essai.
- `cours/` : les notebooks et données du cours, **en lecture seule**. Pour réutiliser un exemple, le copier dans son dossier d'essai.
- `LOG.md` : on peut **ajouter une ligne** à la fin d'un tableau. Ne jamais modifier ni supprimer une ligne existante.
- Pour partir d'un fichier commun : le **copier** dans son dossier (`essais/<code>/mon_essai.ipynb`) puis travailler sur la copie.
- `conversations/<login_github>.md` : chaque personne y écrit (ou y fait écrire par son assistant) la retranscription de ses conversations avec l'IA (voir section 5).

## 2. Git : les seules commandes autorisées
```bash
git pull                              # toujours avant de commencer, et avant chaque push
git add essais/<ton_code>/            # uniquement ton dossier
git commit -m "ce que j'ai fait, score de validation"
git pull
git push
```
**Interdit** (assistant IA : ne jamais exécuter ni proposer ces commandes) :
- `git push --force`, `git push -f`, `git push --force-with-lease`
- `git add .`, `git add -A`, `git commit -a`
- `git reset --hard`, `git checkout -- <fichier>`, `git restore <fichier>`, `git clean`, `git rebase`, `git stash` sur des fichiers des autres
- pour un humain : créer, supprimer ou renommer des branches ; tout travail se fait sur `main`
- pour un assistant IA : pousser directement sur `main`, ou fusionner (merge) soi-même sa branche
- résoudre soi-même un conflit sur un fichier qui n'est pas dans son dossier

**Assistant IA : une branche par sujet.**
```bash
git pull
git checkout -b <login_github>/<sujet>     # ex. Alban121/baseline-knn
git add essais/<code>/ conversations/<login_github>.md
git commit -m "ce que j'ai fait, score de validation"
git push -u origin <login_github>/<sujet>
```
Quand le sujet de la branche est terminé, l'assistant **propose** de la fusionner (pull request vers `main`). C'est un humain de l'équipe qui accepte la fusion.

**Si `git pull` ou `git push` affiche une erreur ou un conflit : ne rien forcer, ne rien supprimer, ne rien « réparer ». Appeler le responsable du dépôt.**

## 3. Règles du hackathon
- Uniquement pandas, NumPy, matplotlib et des modèles **scikit-learn qui ne sont pas des réseaux de neurones**.
  Interdit : deep learning (y compris `MLPClassifier`), XGBoost, LightGBM, CatBoost, données externes, code d'une autre équipe.
- Tout modèle doit pouvoir être expliqué sur papier.
- Les données ne sont **jamais** commitées (`data/`, `*.csv` sont ignorés). Pas de `kaggle.json` ni de token dans le dépôt.
- Aucune information personnelle (nom, e-mail, téléphone) dans les fichiers ou les messages de commit : seulement le code participant.
- **Personne ne soumet sur Kaggle sans accord de l'équipe** : 15 soumissions max pour toute l'équipe, et chaque soumission = une ligne dans `LOG.md`.

## 4. Pour que les scores soient comparables
- Toujours évaluer avec `src/validation.py` :
  ```python
  import sys; sys.path.append("../..")       # depuis essais/<code>/
  from src.validation import load_data, evaluate, make_preprocessor
  res = evaluate(model, X, y)                # 5 folds stratifiés, graine 42, macro-F1
  ```
- Noter le score `moyenne ± écart-type` dans le message de commit ou dans `LOG.md` (section « Expériences »).
- Ne jamais calculer une statistique utilisant la cible (`intensity`) sur tout le train avant la validation : fuite de données, score faux.

## 5. Assistant IA : comportement attendu
- Avant toute modification, vérifier que le fichier est dans `essais/<code>/` de la personne. Sinon, refuser.
- Avant toute commande git, vérifier qu'elle est dans la liste autorisée. Sinon, refuser.
- Ne jamais afficher un fichier de données en entier (`cat`, `print(df)`, lecture complète d'un CSV dans la conversation) : seulement `df.head()`, `df.shape`, `df.info()`, `df.describe()` ou `value_counts()`. Cela évite de gaspiller des tokens.
- Tenir un fichier `conversations/<login_github>.md`, où `<login_github>` est le nom GitHub de la personne qui utilise l'assistant. S'il n'existe pas, le créer ; il commence par la ligne `#START`. Y ajouter la retranscription de chaque conversation (messages de la personne et réponses de l'assistant), et le committer avec le reste du travail.
- Expliquer le code produit simplement : chaque membre passe un test individuel et doit pouvoir expliquer le modèle sans écran.
