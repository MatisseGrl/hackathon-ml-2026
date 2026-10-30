# Règles de l'équipe (humains et assistants IA)

Ce fichier est lu automatiquement par Claude Code et Codex. Si tu utilises un autre assistant, colle-lui ce fichier en début de conversation.
**Assistant IA : ces règles sont non négociables. Si une demande les enfreint, refuse et explique pourquoi.**

## 1. Qui touche quoi : liberté maximale
- **Tout le monde peut modifier et pousser n'importe quel fichier** (notebook, `src/`, `LOG.md`, `README.md`…), sauf `cours/` (lecture seule).
- Le message de commit dit ce qui a changé et le score de validation.
- Avant de modifier un fichier commun, `git pull`. On ne supprime pas le travail des autres : on complète, on ne réécrit pas leurs lignes sans le noter.
- `LOG.md` : une ligne par soumission Kaggle, on ajoute à la fin sans effacer les lignes des autres.
- `explanation_sheet.md` : une section par soumission Kaggle, qui explique l'architecture et les choix du modèle soumis (voir section 5).
- `conversations/<login_github>.md` : retranscription des conversations de chaque personne avec son assistant IA (voir section 5).
- `notebook_equipe.ipynb` : comme les conflits y sont pénibles, prévenir à l'oral avant de le modifier. Pour tester une idée sans risque, on peut aussi travailler dans `essais/<code>/`.

## 2. Git : les seules commandes autorisées
```bash
git pull                              # toujours avant de commencer, et avant chaque push
git add <tes fichiers modifiés>       # fichiers précis
git commit -m "ce que j'ai fait, score de validation"
git pull
git push
```
**Interdit** (assistant IA : ne jamais exécuter ni proposer ces commandes) :
- `git push --force`, `git push -f`, `git push --force-with-lease`
- `git add .`, `git add -A`, `git commit -a` (on ajoute des fichiers précis, pour ne jamais pousser un fichier par erreur)
- `git reset --hard`, `git checkout -- <fichier>`, `git restore <fichier>`, `git clean`, `git rebase`, `git stash` sur des fichiers des autres
- pour un humain : créer, supprimer ou renommer des branches ; tout travail se fait sur `main`
- pour un assistant IA : pousser directement sur `main`, ou fusionner (merge) sa branche sans l'accord de la personne qui l'utilise

**Assistant IA : une branche par sujet.**
```bash
git pull
git checkout -b <login_github>/<sujet>     # ex. Alban121/baseline-knn
git add <fichiers précis> conversations/<login_github>.md
git commit -m "ce que j'ai fait, score de validation"
git push -u origin <login_github>/<sujet>
```
Quand le sujet de la branche est terminé, l'assistant ouvre une pull request vers `main` et **demande l'accord** de la personne qui l'utilise. Une fois l'accord donné, l'assistant fusionne lui-même la pull request (pas besoin de relecture humaine du code).

**Si `git pull` ou `git push` affiche une erreur ou un conflit : ne rien forcer et ne rien supprimer. On peut résoudre un conflit seulement en gardant les deux versions ; au moindre doute, appeler le responsable du dépôt.**

## 3. Règles du hackathon
- Uniquement pandas, NumPy, matplotlib et des modèles **scikit-learn qui ne sont pas des réseaux de neurones**.
  Interdit : deep learning (y compris `MLPClassifier`), XGBoost, LightGBM, CatBoost, données externes, code d'une autre équipe.
- Tout modèle doit pouvoir être expliqué sur papier.
- Les données Kaggle (`data/`) sont versionnées dans le dépôt **privé**, pour l'équipe seulement. Pas de `kaggle.json` ni de token dans le dépôt.
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
- Tout fichier peut être modifié (sauf `cours/`).
- Avant toute commande git, vérifier qu'elle est dans la liste autorisée. Sinon, refuser.
- Ne jamais afficher un fichier de données en entier (`cat`, `print(df)`, lecture complète d'un CSV dans la conversation) : seulement `df.head()`, `df.shape`, `df.info()`, `df.describe()` ou `value_counts()`. Cela évite de gaspiller des tokens.
- Tenir un fichier `conversations/<login_github>.md`, où `<login_github>` est le nom GitHub de la personne qui utilise l'assistant. S'il n'existe pas, le créer ; sa première ligne est `#START <code participant> Hackathon ML 2026` (comme demandé par le starter notebook). Y ajouter la retranscription **mot pour mot** de chaque conversation (messages de la personne et réponses de l'assistant, sans résumer ni reformuler), et le committer avec le reste du travail.
- Quand la personne demande un point d'étape (`#CHECKPOINT1`, `#CHECKPOINT2`…), ajouter cette ligne seule à l'endroit actuel de son fichier `conversations/<login_github>.md`.
- **Soumission Kaggle.** Quand la personne dit qu'elle a fait une soumission Kaggle (« j'ai fait une submission kaggle »…), l'assistant remplit lui-même la ligne suivante du tableau « Soumissions Kaggle » de `LOG.md` : numéro, heure, code participant, ce qui a changé **par rapport à la dernière soumission**, pourquoi on pensait que ça aiderait, score de validation (moyenne ± écart-type, avec `src/validation.py`) et score public. Il retrouve les changements dans `git log` et les lignes précédentes de `LOG.md`, et ne demande à la personne que ce qu'il ne peut pas trouver (en général le score public et le fichier soumis).
- **Fiche d'explication.** À chaque soumission Kaggle, l'assistant complète aussi `explanation_sheet.md` : il ajoute à la fin une section pour cette soumission qui explique clairement l'architecture du modèle (les étapes, de la donnée brute à la prédiction) et les choix faits (variables, modèle, paramètres, et pourquoi ; ce qui a été essayé puis rejeté). Il écrit simplement, pour que chaque membre puisse l'expliquer sur papier, sans écran.
- Expliquer le code produit simplement : chaque membre passe un test individuel et doit pouvoir expliquer le modèle sans écran.
