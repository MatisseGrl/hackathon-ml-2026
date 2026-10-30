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
