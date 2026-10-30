#START 3MAS Hackathon ML 2026

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

**Claude :** (en fusionnant `main`) un coéquipier a déjà poussé `PersonStack` (`src/person_stack.py`) à 0,6207 ± 0,0028, meilleur que ce modèle (0,6007). Conflit dans `LOG.md` résolu en gardant les deux séries de lignes. À noter : l'équipe a rejeté le calcul des habitudes sur les lignes du test, que ce modèle utilise.

#CHECKPOINT2

## 2026-10-30 : vérifier les règles du dépôt (Claude, projet claude.ai, branche `Alban121/regles-readme`) - mot pour mot

**Alban121 :** ouvre un fil poiur verifier les regles du github

**Claude :** J'ai relu les règles sur `main` à jour (`AGENTS.md`, `CLAUDE.md`, `README.md`, `ressources/`) sans ouvrir les données. Sur l'essentiel, tout est bon : elles respectent bien le sujet officiel (scikit-learn sans réseau de neurones, macro-F1, 15 soumissions, une ligne par soumission dans `LOG.md`), et `CLAUDE.md` charge bien `AGENTS.md`. Je n'ai trouvé aucun import interdit dans le code.

**Ce qui se contredit**
1. Le `README.md` n'a pas été mis à jour : il dit encore que `data/` n'est « pas versionné », qu'il faut télécharger les CSV depuis Kaggle, et que chacun « travaille dans son dossier `essais/` ». Or `AGENTS.md` dit maintenant que les données sont dans le dépôt et que tout le monde peut modifier n'importe quel fichier.
2. Les branches : la règle demande `<login>/<sujet>`, mais la PR #1 utilisait `claude/project-thread-…`. Les PR #3, #4 et #5 respectent bien la règle.
3. `#START` : le starter demande `#START <code> Hackathon ML 2026`, alors que les fichiers de conversation commencent seulement par `#START`, sans le code. Je n'ai trouvé aucun `#CHECKPOINT` nulle part.
4. Préparer les données sur le test : `LOG.md` rejette « UserStats ajusté sur le test » (on ne prétraite pas sur le test), mais le modèle 3MAS (0,601) le fait. C'est permis puisque la cible n'est pas utilisée, mais l'équipe devrait choisir une seule position.

**Ce qui n'est pas appliqué**
- Seuls Alban121 et MatisseGrl ont un fichier `conversations/`. Ceux de Garlot et de RMYU (XOUMITO) manquent.
- La soumission n°2 a un score public, mais les colonnes « Heure » et « Qui » sont vides, comme pour la n°1.
- La fiche `essais/<code>/<code>.md` n'est plus obligatoire (MatisseGrl l'a retirée dans la PR #3). Seul 3MAS en a encore une.

Je n'ai modifié aucun fichier. Si tu veux, je peux corriger le README et ajouter le code dans `#START` sur une branche.

**Alban121 :** 1. Met a jour le READ.me. 

2. C est pas grave
3. Modifie la regle pour que ca corresponde. Ajoute un #CHECKPOINT2 pour ma discution. Pour la regle sur la retranscription dans les fichier utilisateur.md, il faut que ce soit mot pout mot.
4. Je ne comprends pas ce que tu veux dire.
5. Ajoute une regle : Qunad quelqu'un dit "j'ai fait une submission kaggle" l'agent remplis automatiquement le LOG.md avec les changement par rapport au dernier modele submitted, l heure et autre.
6. Il n'y a rien pour la explanation sheet. Il n y avait pas une regle dessus ?

**Claude :** Je m'y mets : je cherche d'abord s'il existait une règle sur la fiche d'explication, puis je modifie les fichiers sur une branche.

**Claude :** C'est fait, sur la branche `Alban121/regles-readme` (PR #7 : https://github.com/MatisseGrl/hackathon-ml-2026/pull/7). Je la fusionne dès que tu me donnes ton accord.

1. `README.md` est à jour : les données sont dans le dépôt, tout le monde peut modifier les fichiers, et les IA travaillent sur des branches.
3. Le fichier de conversation commence maintenant par `#START <code> Hackathon ML 2026`. La retranscription doit être mot pour mot, et une règle `#CHECKPOINT` a été ajoutée. Dans ton fichier, j'ai mis `#START 3MAS Hackathon ML 2026`, `#CHECKPOINT2`, puis ce fil recopié mot pour mot. Les conversations d'avant restent des résumés, je n'ai pas leur texte exact.
4. Dans `LOG.md`, l'équipe a écrit qu'elle refusait de calculer les habitudes de chaque personne en utilisant aussi les lignes du test. Pourtant, ton modèle à 0,601 le fait. Ce n'est pas de la triche, puisque la réponse du test n'est pas utilisée. Mais l'équipe dit une chose et en fait une autre, donc il faudra décider à l'oral si on l'autorise. Ça ne bloque rien.
5. Règle ajoutée : quand quelqu'un dit qu'il a fait une soumission Kaggle, l'IA remplit la ligne de `LOG.md` (changements depuis la dernière soumission, heure, code, scores). Elle ne demande que ce qu'elle ne trouve pas, en général le score public.
6. Non, il n'y a jamais eu de vraie règle là-dessus. Il y a seulement « tout modèle doit pouvoir être expliqué sur papier », et une case vide « Explication courte du modèle (pour la fiche papier) » en bas de `LOG.md`. Je peux ajouter une règle (par exemple : pour chaque modèle soumis, l'IA écrit son explication en quelques lignes) et préparer la fiche du meilleur modèle. Dis-moi si tu veux.

À noter : MatisseGrl doit encore ajouter son code dans son `#START`. Je ne le connais pas.
