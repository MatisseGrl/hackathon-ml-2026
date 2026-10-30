# Fiche de commit : 3MAS

## 2026-10-30 15h46 · Claude Code / Claude Opus 5.5
- Ce qui a changé : `AGENTS.md`, nouvelles règles pour les assistants IA : une branche par sujet puis pull request fusionnée par l'IA après l'accord de la personne ; interdiction d'afficher un fichier de données en entier ; fichier `conversations/<login_github>.md` avec la retranscription des conversations (créé pour Alban121).
- Pourquoi on pense que ça aide : les IA ne poussent plus directement sur `main`, on économise des tokens, et les conversations sont gardées pour le rendu Boostcamp.
- Score de validation (moyenne ± écart-type) : sans objet (règles uniquement, pas de modèle).
- Suggestion de l'IA vérifiée, rejetée ou corrigée (et comment) : l'IA proposait aussi de demander le code participant au début ; jugé non nécessaire par 3MAS. Explications sur la fuite de données par utilisateur et sur `GroupKFold` données, sans changement de règle.

## 2026-10-30 16h30 · Claude Code / Claude Opus 5.5
- Ce qui a changé : nouveau `essais/3MAS/modele_06.py` (classe `PersonFeatures` + `build_model`) et CSV de test `essais/3MAS/submission_3MAS.csv` (non soumis) ; une ligne ajoutée dans `LOG.md`, section « Expériences ». Aucun fichier commun modifié à part cette ligne.
- Pourquoi on pense que ça aide : (1) les habitudes de chaque utilisateur (UserStats) sont calculées sur train + test réunis, sans la cible, donc mieux estimées ; (2) une régression linéaire par utilisateur-sport prédit la vitesse attendue selon durée, pente, saison et année ; le résidu dit « plus rapide que d'habitude pour ce parcours » ; (3) les arbres reçoivent `user_id` (catégoriel pour HistGB, one-hot pour ExtraTrees) et peuvent apprendre des règles propres à chaque personne ; (4) la moyenne des probabilités des deux modèles réduit leurs erreurs.
- Score de validation (moyenne ± écart-type) : 0,6007 ± 0,0053 (folds : 0,599 / 0,605 / 0,591 / 0,603 / 0,606). Étapes : base équipe 0,568 ; habitudes sur train+test 0,574 ; + résidus 0,576 ; + user_id catégoriel 0,589 ; HistGB 63 feuilles 0,595 ; + ExtraTrees 0,601.
- Suggestion de l'IA vérifiée, rejetée ou corrigée (et comment) : rejetées car sans gain (≤ 0,575) : profils moyens de chaque classe par utilisateur (calculés en validation interne), corrélations par utilisateur entre variables et difficulté, et découpage forcé en tiers par utilisateur. Vérifié : tous les `user_id` du test sont dans le train, donc `StratifiedKFold` reste adapté (pas besoin de `GroupKFold`).
