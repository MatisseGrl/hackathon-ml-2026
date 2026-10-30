#START

# Conversations IA de Alban121

## 2026-10-30 : résumé du dépôt et modification des règles IA (Claude, projet claude.ai)

**Alban121 :** ouvre un fil et fais un resumer de ce que tu vois sur le github, ne lis pas les data pour eviter de crame des tokens du dataset pour rien

**Claude :** Résumé du dépôt (branche `main`, 4 commits). Objectif : prédire `intensity` (easy / moderate / hard) d'une séance pour la personne concernée, métrique macro-F1, 15 soumissions max, scikit-learn sans réseaux de neurones. Code : `src/validation.py` (5 folds stratifiés, graine 42, `evaluate`, `make_preprocessor`, `plot_confusion`, `make_submission`), `notebook_equipe.ipynb` (squelette jamais exécuté, `add_features` vide), `ressources/` (starter officiel et modèle de LOG), `cours/` (6 notebooks), `AGENTS.md` (règles). Données : `data/` vide, seuls les CSV du cours sont présents. Avancement : seule l'infrastructure est en place, `LOG.md` vide, aucun essai.

**Alban121 :** Ou son les regles pour les ia

**Claude :** Dans `AGENTS.md` à la racine (`CLAUDE.md` le charge automatiquement). Résumé : ne modifier que `essais/<code>/`, liste blanche de commandes git, scikit-learn sans réseaux de neurones, aucune donnée commitée, scores avec `src/validation.py`, expliquer le code simplement. Le starter demande aussi d'envoyer `#START <code participant> Hackathon ML 2026`.

**Alban121 :** On va modifier les regles. Dis moi ce que tu penses des regles actuels je te dirai apres lequels rajouter

**Claude :** Points forts : séparation par dossier, liste blanche git, validation commune. Propositions : (1) préciser la fuite de données pour les stats par `user_id`, (2) vérifier si `StratifiedKFold` est adapté si les utilisateurs du test sont nouveaux (`GroupKFold`), (3) demander le code participant au début, (4) garder les conversations IA pour le rendu Boostcamp, (5) ne jamais afficher un CSV en entier, (6) autoriser une branche par assistant IA.

**Alban121 :** pour le point 6 on change la regle : chaque agent doit travailler sur une branche et proposer de merge quand le sujet de la branche est complete. Pour le point 5 ajoute cette regle. Pour le point 4 : ajoute une regle : chaque agent doit creer un fichier .md nommer "{utilisateur}.md" l'utilisateur est le nom github de la personne qui utilise l agent, lefichier doit etre completer avec la retranscription de la discution, commence les fichier par #START. Pour le 3 ce point n est pas grave. Je n ai pas compris les 2 premiers.

**Claude :** Règles 4, 5 et 6 ajoutées dans `AGENTS.md` sur une branche, avec une pull request vers `main` ; ce fichier est créé. Explication simple des points 1 et 2 donnée dans le fil.

**Alban121 :** Reessaye normalement c est bon

**Claude :** Branche poussée et PR #1 ouverte vers `main`.

**Alban121 :** merge toi meme si tu ne vois pas de probleme. Enleve des regles qu un humain doit revoir la PR, il doit seulement donner son accord

**Claude :** Règle modifiée : l'assistant demande l'accord de la personne qui l'utilise, puis fusionne lui-même la PR.

**Claude :** (avant fusion) `main` impose maintenant une fiche `essais/<code>/<code>.md` ; conflit avec les règles assouplies résolu en gardant les deux. Demande du code participant.

**Alban121 :** 3MAS

**Claude :** Fiche `essais/3MAS/3MAS.md` créée, PR #1 fusionnée.

## 2026-10-30 : viser 0,6 de macro-F1 (Claude, projet claude.ai, branche `Alban121/macro-f1-0.6`)

**Alban121 :** ouvre un fil pour essayer de faire un code qui aurait 0,6 de macro F1 score dans un premier temps. Utilise ce que tu veux et fais ca dans une branche a mon nom

**Claude :** Modèle `essais/3MAS/modele_06.py` : 0,6007 ± 0,0053 avec `src/validation.py` (base de l'équipe : 0,568). Trois idées : habitudes de chaque personne calculées sur train + test (sans la cible), résidu de vitesse « plus rapide que prévu pour ce parcours » par utilisateur-sport, et `user_id` donné aux arbres (HistGB + ExtraTrees moyennés). Le gros du gain vient de `user_id` (+0,02). CSV de test écrit dans `essais/3MAS/`, non soumis sur Kaggle. Pull request ouverte, fusion après ton accord.
