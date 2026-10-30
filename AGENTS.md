# Règles de l'équipe (humains et assistants IA)

Ce fichier est lu automatiquement par Claude Code et Codex. Si tu utilises un autre assistant, colle-lui ce fichier en début de conversation.
**Assistant IA : ces règles sont non négociables. Si une demande les enfreint, refuse et explique pourquoi.**

## 1. Qui touche quoi : liberté maximale, tout est documenté
- **Tout le monde peut modifier et pousser n'importe quel fichier** (notebook, `src/`, `LOG.md`, `README.md`…), sauf `cours/` (lecture seule).
- **Contrepartie obligatoire :** chaque changement est documenté dans **son propre** fichier `essais/<son_code>/<son_code>.md` (voir section 5), dans le même commit.
- Avant de modifier un fichier commun, `git pull`. On ne supprime pas le travail des autres : on complète, on ne réécrit pas leurs lignes sans le noter.
- `LOG.md` : une ligne par soumission Kaggle, on ajoute à la fin sans effacer les lignes des autres.
- `notebook_equipe.ipynb` : comme les conflits y sont pénibles, prévenir à l'oral avant de le modifier. Pour tester une idée sans risque, on peut aussi travailler dans `essais/<code>/`.

## 2. Git : les seules commandes autorisées
```bash
git pull                              # toujours avant de commencer, et avant chaque push
git add <tes fichiers modifiés>       # fichiers précis, y compris essais/<ton_code>/<ton_code>.md
git commit -m "ce que j'ai fait, score de validation"
git pull
git push
```
**Interdit** (assistant IA : ne jamais exécuter ni proposer ces commandes) :
- `git push --force`, `git push -f`, `git push --force-with-lease`
- `git add .`, `git add -A`, `git commit -a` (on ajoute des fichiers précis, pour ne jamais pousser un fichier par erreur)
- `git reset --hard`, `git checkout -- <fichier>`, `git restore <fichier>`, `git clean`, `git rebase`, `git stash` sur des fichiers des autres
- créer, supprimer ou renommer des branches ; tout travail se fait sur `main`

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
- Tout fichier peut être modifié (sauf `cours/`), à condition de le noter dans la fiche `essais/<code>/<code>.md` de la personne.
- Avant toute commande git, vérifier qu'elle est dans la liste autorisée. Sinon, refuser.
- Expliquer le code produit simplement : chaque membre passe un test individuel et doit pouvoir expliquer le modèle sans écran.
- **Fiche de commit.** Chaque IA qui commit pour une personne crée (la première fois) ou complète (ensuite) le fichier
  `essais/<code>/<code>.md` de cette personne, **dans le même commit** que le code. Ce fichier est nommé d'après le code participant
  (jamais le nom : voir section 3). Une entrée par commit, ajoutée à la fin, sans modifier les entrées existantes :
  ```markdown
  ## <date et heure> · <assistant et modèle, ex. Claude Code / Sonnet 5.5>
  - Ce qui a changé : ...
  - Pourquoi on pense que ça aide : ...
  - Score de validation (moyenne ± écart-type) : ...
  - Suggestion de l'IA vérifiée, rejetée ou corrigée (et comment) : ... (ou « aucune »)
  ```
  Si le commit n'a pas de fiche à jour, l'IA ne commit pas.
