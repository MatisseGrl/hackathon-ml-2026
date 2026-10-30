# Fiche d'explication des modèles soumis

Une section **par soumission Kaggle**, ajoutée à la fin (même numéro que dans `LOG.md`).
Chaque section remplit **exactement** la fiche d'explication individuelle papier (questions A à E, mêmes intitulés, mêmes consignes).
Le jour J, la fiche se fait en 20 minutes, appareils fermés, travail individuel, noté sur 6 points : chaque membre doit pouvoir la refaire de mémoire.
On n'écrit que le code participant, jamais le nom.

Modèle d'une section (recopier les intitulés tels quels) :

```markdown
## Soumission n° <numéro> : <nom du modèle> (<heure>, <code participant>)
Code participant : <code> · Équipe n° : <n°> · Date : <jj/mm>

### A. Dessinez le pipeline final de votre équipe (2 points)
_Du CSV brut jusqu'à submission.csv : chaque étape dans une boîte, reliées par des flèches. Écrivez dans chaque boîte ce qui est fait et avec quel outil (ex. « one-hot sur sport », « RandomForest, max_depth=12 »). Indiquez où se fait la validation._

    [ train.csv / test.csv ]
              |
              v
    [ étape 1 : ce qui est fait, avec quel outil ]
              |
              v
    [ ... ]   <-- validation : où et comment (ex. src/validation.py, 5 folds stratifiés, graine 42, macro-F1)
              |
              v
    [ submission.csv ]

### B. Pourquoi votre modèle final fait-il mieux que votre première soumission ? (2 points)
_Nommez votre modèle final et expliquez en deux lignes comment il produit une prédiction. Puis citez deux décisions (variables, modèle, hyperparamètres, validation) et expliquez par quel mécanisme chacune a amélioré le score._
- Modèle final : ...
- Comment il prédit (deux lignes) : ...
- Décision 1 : ... → mécanisme : ... (score avant → après)
- Décision 2 : ... → mécanisme : ... (score avant → après)

### C. Une suggestion de l'IA que vous avez vérifiée (1 point)
_Décrivez une suggestion de l'IA que votre équipe a rejetée ou corrigée, ou acceptée seulement après vérification. Qu'avez-vous vérifié, et comment ?_
- Suggestion : ...
- Rejetée / corrigée / acceptée après vérification : ...
- Ce qu'on a vérifié, et comment : ...

### D. De la prédiction à la recommandation (1 point)
_L'utilisateur 4127 court habituellement à une vitesse médiane de 10 km/h. Trois séances possibles : (1) 5 km à plat à 9 km/h, (2) 12 km à 10 km/h avec 300 m de dénivelé, (3) 8 km à plat à 12 km/h. Vous voulez recommander une séance de récupération facile. Comment utiliseriez-vous votre modèle pour choisir ? Quelle séance vous attendez-vous à ce qu'il choisisse, et qu'est-ce qui pourrait le rendre faux pour cet utilisateur ?_
- Comment utiliser le modèle : ...
- Séance attendue : ...
- Ce qui pourrait le rendre faux pour cet utilisateur : ...

### E. Votre contribution personnelle (non noté)
_En deux lignes : qu'avez-vous fait personnellement dans l'équipe aujourd'hui ?_
...

- Scores : validation macro-F1 <moyenne ± écart-type> ; public <score>
```

<!-- Les sections des soumissions commencent ici. La section n° 3 a été écrite avant ce format A à E. -->

## Soumission n° 3 : mélange PersonStack + Final2, doublons et quotas (16h59, RMYU)
- **Changements par rapport à la soumission précédente** (n° 2, `NeighborStack`, public 0,61791) :
  on remplace NeighborStack par la moyenne de deux modèles plus riches (PersonStack de l'équipe et Final2),
  puis on ajoute deux post-traitements : recopie du label des séances en double, et affectation sous quotas par personne.
- **Ce que « hard » veut dire** : pour chaque personne, ses séances sont coupées en 3 tiers égaux selon une mesure
  des capteurs (très probablement la fréquence cardiaque moyenne). « Hard » = le tiers le plus intense **pour elle**.
  Ces tiers sont calculés sur toutes ses séances, train + test réunis.
- **Architecture** (de la donnée brute à la prédiction, étape par étape) :
  1. **Variables relatives à la personne** : pour chaque séance, on compare la vitesse, la distance, la durée, le dénivelé…
     aux habitudes de la même personne dans le même sport (rang parmi ses séances, écart à sa médiane en écarts-types),
     et on calcule une « vitesse résiduelle » : sa vitesse moins celle qu'on attendrait d'elle pour cette distance, cette pente et cette date.
  2. **Étage 1** (HistGradientBoosting) : à partir de ces variables, prédit un niveau attendu E entre 0 (easy) et 2 (hard).
     Sur le train, E est calculé hors pli (le modèle ne voit jamais la séance qu'il note).
  3. **Contexte de la personne** : pour chaque séance, ce que disent les séances **étiquetées** de la même personne :
     label moyen et « surprise » moyenne (label − E) des séances du même jour et de la même heure, du même mois,
     des mois voisins (±1 à ±16 mois), et des séances les plus semblables (k plus proches voisins).
     Idée : à vitesse égale, la même personne est plus ou moins essoufflée selon sa forme du moment.
     Pour les séances du train, ce contexte vient uniquement des **autres** plis internes (sinon fuite de la cible).
  4. **Étage 2** : PersonStack (HistGB + ExtraTrees, avec en plus des moyennes à noyaux et une ridge par personne)
     et Final2 (HistGB) prédisent chacun 3 probabilités (easy, moderate, hard) ; on fait la moyenne 0,6 × PersonStack + 0,4 × Final2.
  5. **Doublons** : si une séance du test est identique à une séance du train sur les 16 colonnes (même personne, mêmes mesures),
     on recopie son label (vrai dans 99,6 % des cas sur le train ; 327 séances du test concernées).
  6. **Quotas** : pour chaque personne, on sait combien d'easy, de moderate et de hard il reste dans le test
     (un tiers de toutes ses séances, moins ce qui est déjà dans le train). On attribue les classes en respectant
     exactement ces nombres, en donnant « hard » aux séances jugées les plus probablement hard, etc.
     (problème d'affectation résolu par `scipy.optimize.linear_sum_assignment`, qui maximise la somme des log-probabilités).
- **Variables utilisées et pourquoi** : uniquement des variables relatives à la personne (indice 1 : « 12 km/h, c'est rapide pour qui ? »),
  la vitesse résiduelle (une montée ou une longue sortie ralentit sans être plus facile), et le contexte temporel de la personne
  (la difficulté dépend de la forme du moment). Pas de `workout_id` ni de données externes.
- **Modèle et paramètres, et pourquoi ce choix** : HistGradientBoosting (lr 0,05, 300 itérations) car les effets sont non linéaires
  et il gère les valeurs manquantes ; empilement en 2 étages pour utiliser les labels des séances voisines sans fuite ;
  moyenne de deux modèles différents pour réduire la variance (poids 0,6 : score stable de 0,5 à 0,7).
- **Essayé puis rejeté (et pourquoi)** :
  moyennes « leave-one-out » (0,58 : la moyenne du groupe sans la séance est anti-corrélée à son label, fuite) ;
  recopie des doublons partiels (même durée et distance seulement : 83 % d'accord, fait baisser le score) ;
  quotas d'un tiers imposés au seul test (0,6037 : il faut retirer ce qui est déjà connu dans le train) ;
  variables de charge d'entraînement, voisins du test non étiquetés, calibration du décodage (gains nuls ou dans le bruit).
- **Matrice de confusion** (hors pli) : les erreurs sont presque toutes d'un seul niveau (easy↔moderate, moderate↔hard) ;
  `moderate` est la plus difficile (F1 0,52 contre 0,68 et 0,69).
- **Scores** : validation macro-F1 0,6314 ± 0,0055 (plis de `src/validation.py`) ; public 0,65298
