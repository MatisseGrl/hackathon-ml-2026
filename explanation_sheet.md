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

## Soumission n° 4 : PersonStack HGB + quotas (17h00, code participant à renseigner)
- **Changements par rapport à la soumission précédente** : la soumission 2 était NeighborStack. On ajoute les variables « par personne » de `PersonStack` (séances semblables, ridge par groupe, jumeaux), avec un HGB plus régularisé au 2e étage. Surtout, la classe finale ne vient plus de la probabilité la plus forte : elle respecte les quotas de chaque couple utilisateur + sport. Le fichier soumis est `sub_04_quotas.csv` (`sub_03` n'a pas été soumis).
- **Architecture** (de la donnée brute à la prédiction, étape par étape) :
  1. Nettoyage et `UserStats` : chaque séance est comparée aux habitudes de la personne dans ce sport (ratio à la médiane, écart en écarts-types, rang).
  2. 1er étage : un HistGradientBoosting donne un niveau attendu de 0 (easy) à 2 (hard), hors fold pour le train. Le résidu (vrai niveau - niveau attendu) mesure si la séance a été plus dure que prévu.
  3. Variables de voisinage : niveaux et résidus des séances de la même personne proches dans le temps, moyennes pondérées par la ressemblance, ridge par couple utilisateur + sport, séances strictement identiques du train. Une séance du train ne voit jamais son propre label (découpage interne en 5).
  4. 2e étage : un HistGradientBoosting donne trois probabilités par séance.
  5. Quotas : pour un groupe de N séances (train + test), easy = N // 3, moderate = 2N // 3 - N // 3, hard = N - 2N // 3. On retire ce que le train contient déjà : il reste le nombre de places de chaque classe dans le test.
  6. Affectation : dans chaque groupe, la méthode hongroise répartit les places pour maximiser la somme des log-probabilités.
- **Variables utilisées et pourquoi** : « difficile » est relatif à la personne, donc on compare chaque séance à ses habitudes. À vitesse égale, la dureté dépend de la forme du moment, que révèlent les séances proches (corrélation 0,52 avec le niveau). Le lien entre vitesse et dureté varie selon les gens, d'où un ridge par groupe. Pour les quotas, l'indice est que les parts de classes par groupe varient de 0,025, contre 0,048 au hasard et 0,024 attendu pour des terciles exacts vus à 75 %.
- **Modèle et paramètres, et pourquoi ce choix** : 2e étage HistGradientBoosting (500 itérations, lr 0,03, 15 feuilles, `min_samples_leaf=40`, `l2_regularization=5`) : régularisé car il y a 140 variables. HGB seul plutôt que le vote avec ExtraTrees : même score avec quotas (0,6404 contre 0,6406) et plus simple à expliquer. Pas de poids sur `moderate` : les quotas fixent déjà le nombre de séances de chaque classe.
- **Essayé puis rejeté (et pourquoi)** : quotas d'un tiers fixes (0,6037), car ils ignorent ce que le train a déjà pris ; variables physiques (0,6423 ± 0,0061), autres réglages (0,6399 à 0,6416) et 20 découpages internes (0,6377) : dans le bruit ; ordinal (0,638) et régression (0,6251 avec erreur simulée) : moins bons ; UserStats seul + quotas (0,5926) : l'empilement reste nécessaire.
- **Scores** : validation macro-F1 de 0,6344 ± 0,0026 (erreur d'arrondi simulée) à 0,6404 ± 0,0038 (effectifs réels du fold), mêmes folds que `evaluate` mais script séparé, car `evaluate` ne peut pas appliquer les quotas ; public 0,65332.

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

## Soumission n° 5 : moyenne de 5 modèles, doublons et quotas par utilisateur + sport (17h37, RMYU)
Code participant : RMYU · Équipe n° : à renseigner · Date : 30/10

### A. Dessinez le pipeline final de votre équipe (2 points)
_Du CSV brut jusqu'à submission.csv : chaque étape dans une boîte, reliées par des flèches. Écrivez dans chaque boîte ce qui est fait et avec quel outil (ex. « one-hot sur sport », « RandomForest, max_depth=12 »). Indiquez où se fait la validation._

    [ train.csv / test.csv ]
              |
              v
    [ nettoyage (altitudes aberrantes) + variables relatives à la personne : rang et écart à sa médiane
      par utilisateur + sport (vitesse, distance, durée, dénivelé), vitesse résiduelle (pandas, sans la cible) ]
              |
              v
    [ 5 modèles HistGradientBoosting empilés en 2 étages : PersonStack (graines 42 et 7), Final2, NeighborStack, Resid2.
      Étage 1 : niveau attendu ; étage 2 : + labels des séances voisines de la personne (même jour, même mois, ±k mois),
      calculés hors pli pour ne pas voir sa propre cible ]   <-- validation : src/validation.py, 5 folds stratifiés, graine 42, macro-F1
              |
              v
    [ moyenne simple des probabilités des 5 modèles (poids 1/5) ]
              |
              v
    [ doublons : séance du test identique à une séance du train sur 16 colonnes → on recopie son label (326 séances) ]
              |
              v
    [ quotas : par utilisateur + sport, tiers exacts sur train + test (N//3 easy, 2N//3 − N//3 moderate, le reste hard)
      moins ce que le train contient → affectation qui maximise la somme des log-probabilités (NumPy) ]
              |                     <-- validation des quotas : appliqués au pli avec ses vrais comptes, comme en soumission
              v
    [ submission.csv ]

### B. Pourquoi votre modèle final fait-il mieux que votre première soumission ? (2 points)
_Nommez votre modèle final et expliquez en deux lignes comment il produit une prédiction. Puis citez deux décisions (variables, modèle, hyperparamètres, validation) et expliquez par quel mécanisme chacune a amélioré le score._
- Modèle final : moyenne de 5 modèles HistGradientBoosting empilés (PersonStack ×2, Final2, NeighborStack, Resid2) + doublons + quotas par utilisateur et sport.
- Comment il prédit (deux lignes) : chaque modèle compare la séance aux habitudes de la personne et à ses séances voisines dans le temps, et donne 3 probabilités ; on les moyenne.
  Puis, dans chaque couple personne + sport, on attribue exactement le nombre d'easy, de moderate et de hard qui reste, aux séances les plus probables pour chaque classe.
- Décision 1 : variables relatives à la personne + labels des séances voisines dans le temps (empilement) → mécanisme : « hard » est un tiers de la fréquence cardiaque **de cette personne**,
  qui dérive avec sa forme du moment ; on compare donc à ses habitudes et à ses séances proches (validation 0,568 → 0,6195).
- Décision 2 : quotas exacts par utilisateur + sport → mécanisme : les classes sont des tiers exacts calculés sur train + test, donc on connaît le nombre de chaque classe restant dans le test ;
  il ne reste qu'à bien classer les séances à l'intérieur de la personne (PersonStack 0,6195 → 0,6406 avec les comptes réels ; public 0,61791 (n° 2) → 0,65523 (n° 5)).

### C. Une suggestion de l'IA que vous avez vérifiée (1 point)
_Décrivez une suggestion de l'IA que votre équipe a rejetée ou corrigée, ou acceptée seulement après vérification. Qu'avez-vous vérifié, et comment ?_
- Suggestion : situer chaque séance par rapport aux « seuils locaux » estimés à partir des scores des séances étiquetées proches dans le temps ; la validation annonçait +0,010 (0,6468 → 0,6568, 5 plis sur 5).
- Rejetée / corrigée / acceptée après vérification : rejetée.
- Ce qu'on a vérifié, et comment : le gain venait uniquement des voisines du même mois. En les retirant, le gain tombait à +0,0000. Cause : le score hors pli d'une voisine vient d'un modèle entraîné
  avec le pli évalué, dont les variables de contexte contiennent le label qu'on cherche à prédire. Au test, ce canal n'existe pas : c'était une fuite. De même, des poids de mélange réglés (0,6495) ne faisaient
  pas mieux que des poids égaux une fois choisis sur 4 plis et notés sur le 5e (0,6471 contre 0,6468) : on a gardé les poids égaux.

### D. De la prédiction à la recommandation (1 point)
_L'utilisateur 4127 court habituellement à une vitesse médiane de 10 km/h. Trois séances possibles : (1) 5 km à plat à 9 km/h, (2) 12 km à 10 km/h avec 300 m de dénivelé, (3) 8 km à plat à 12 km/h. Vous voulez recommander une séance de récupération facile. Comment utiliseriez-vous votre modèle pour choisir ? Quelle séance vous attendez-vous à ce qu'il choisisse, et qu'est-ce qui pourrait le rendre faux pour cet utilisateur ?_
- Comment utiliser le modèle : construire les variables de chaque séance candidate comme pour le test (vitesse, distance et dénivelé comparés aux habitudes de l'utilisateur 4127 en course, date du jour pour sa forme du moment),
  prédire P(easy) pour chacune (sans quotas, qui n'ont de sens que pour un ensemble de séances), et recommander celle qui a la plus forte P(easy).
- Séance attendue : la (1), 5 km à plat à 9 km/h : plus lente que sa médiane (−10 %), courte, sans dénivelé. La (2) ajoute 300 m de montée et de la longueur (plutôt moderate ou hard), la (3) est 20 % plus rapide que sa médiane (plutôt hard).
- Ce qui pourrait le rendre faux pour cet utilisateur : le modèle apprend des corrélations et non des causes (on court lentement quand on est fatigué, donc une séance lente peut être notée « hard » en fréquence cardiaque) ;
  la cible est sa fréquence cardiaque, que nous ne voyons pas (chaleur, fatigue, maladie) ; sa forme du moment peut avoir changé depuis ses dernières séances étiquetées ; enfin, la classe est relative à son propre historique
  (un tiers de ses séances), pas une intensité absolue.

### E. Votre contribution personnelle (non noté)
_En deux lignes : qu'avez-vous fait personnellement dans l'équipe aujourd'hui ?_
RMYU : modèle Final2 (2 étages sans fuite), découverte des tiers exacts et des doublons (quotas, recopie), mélange de 5 modèles de la soumission n° 5,
et vérification des pistes par sélection imbriquée et relecture adversariale (5 pistes rejetées, dont une fuite démasquée).

- Scores : validation macro-F1 0,6468 ± 0,0073 (quotas appliqués au pli avec ses vrais comptes ; même protocole : n° 3 0,6437, PersonStack seul 0,6406) ; public 0,65523

## Soumission n° 6 : moyenne de 8 modèles, second choix final (17h59, RMYU)
Code participant : RMYU · Équipe n° : à renseigner · Date : 30/10

### A. Dessinez le pipeline final de votre équipe (2 points)
_Du CSV brut jusqu'à submission.csv : chaque étape dans une boîte, reliées par des flèches. Écrivez dans chaque boîte ce qui est fait et avec quel outil (ex. « one-hot sur sport », « RandomForest, max_depth=12 »). Indiquez où se fait la validation._

    [ train.csv / test.csv ]
              |
              v
    [ variables relatives à la personne (rangs, écarts à sa médiane par utilisateur + sport, vitesse résiduelle), pandas, sans la cible ]
              |
              v
    [ 8 modèles HistGradientBoosting empilés en 2 étages : PersonStack (graines 42, 7, 11), Final2 (graines 42, 7),
      NeighborStack (graines 42, 7), Resid2 ; étage 2 = + labels des séances voisines de la personne, calculés hors pli ]
              |                     <-- validation : src/validation.py, 5 folds stratifiés, graine 42, macro-F1
              v
    [ moyenne simple des probabilités des 8 modèles (poids 1/8) ]
              |
              v
    [ doublons exacts du train (16 colonnes) : on recopie le label (326 séances) ]
              |
              v
    [ quotas exacts par utilisateur + sport (N//3 easy, 2N//3 − N//3 moderate, reste hard, moins le train) : affectation NumPy ]
              |                     <-- validation des quotas : appliqués au pli avec ses vrais comptes, comme en soumission
              v
    [ submission.csv ]

### B. Pourquoi votre modèle final fait-il mieux que votre première soumission ? (2 points)
_Nommez votre modèle final et expliquez en deux lignes comment il produit une prédiction. Puis citez deux décisions (variables, modèle, hyperparamètres, validation) et expliquez par quel mécanisme chacune a amélioré le score._
- Modèle final : la n° 5 (moyenne de 5 modèles + doublons + quotas) ; la n° 6 est la même méthode avec 8 modèles, soumise comme second choix final.
- Comment il prédit (deux lignes) : chaque modèle compare la séance aux habitudes de la personne et à ses séances voisines dans le temps, et donne 3 probabilités ; on les moyenne.
  Puis, dans chaque couple personne + sport, on attribue exactement le nombre d'easy, de moderate et de hard qui reste, aux séances les plus probables pour chaque classe.
- Décision 1 : quotas exacts par utilisateur + sport → mécanisme : les classes sont des tiers exacts calculés sur train + test, donc le nombre de chaque classe restant dans le test est connu ;
  il ne reste qu'à bien classer les séances à l'intérieur de la personne (PersonStack 0,6195 → 0,6406 ; public 0,61791 (n° 2) → 0,65523 (n° 5)).
- Décision 2 : choisir deux soumissions finales qui se trompent sur des séances différentes → mécanisme : Kaggle garde le meilleur des deux scores privés ; un second fichier de même valeur
  moyenne mais différent sur 3,8 % des séances rapporte en moyenne +0,0013 (simulation sur 3 000 « moitiés privées » ; avec la n° 3 : +0,0008).

### C. Une suggestion de l'IA que vous avez vérifiée (1 point)
_Décrivez une suggestion de l'IA que votre équipe a rejetée ou corrigée, ou acceptée seulement après vérification. Qu'avez-vous vérifié, et comment ?_
- Suggestion : ajouter des graines aléatoires (8 modèles au lieu de 5) pour améliorer le modèle.
- Rejetée / corrigée / acceptée après vérification : corrigée : rejetée comme modèle principal, acceptée seulement comme second choix.
- Ce qu'on a vérifié, et comment : règle fixée avant de voir les résultats (adoptée seulement si elle ne perd pas en validation) : 0,6462 contre 0,6468, donc pas de gain ;
  puis simulation du score final « meilleur des deux » : c'est le meilleur second choix (+0,0013), confirmé par deux estimations indépendantes (+0,0008 à +0,0016).

### D. De la prédiction à la recommandation (1 point)
_L'utilisateur 4127 court habituellement à une vitesse médiane de 10 km/h. Trois séances possibles : (1) 5 km à plat à 9 km/h, (2) 12 km à 10 km/h avec 300 m de dénivelé, (3) 8 km à plat à 12 km/h. Vous voulez recommander une séance de récupération facile. Comment utiliseriez-vous votre modèle pour choisir ? Quelle séance vous attendez-vous à ce qu'il choisisse, et qu'est-ce qui pourrait le rendre faux pour cet utilisateur ?_
- Comment utiliser le modèle : calculer les variables de chaque séance candidate par rapport aux habitudes de l'utilisateur 4127 en course (et à sa forme récente), prédire P(easy) pour chacune
  (sans quotas, qui n'ont de sens que pour un ensemble de séances), et recommander la plus forte P(easy).
- Séance attendue : la (1), 5 km à plat à 9 km/h (plus lente que sa médiane, courte, plate) ; la (2) ajoute 300 m de montée et de la longueur, la (3) est 20 % plus rapide que sa médiane.
- Ce qui pourrait le rendre faux pour cet utilisateur : le modèle apprend des corrélations, pas des causes (on court lentement quand on est fatigué) ; la cible est sa fréquence cardiaque, que nous ne voyons pas
  (chaleur, fatigue, maladie) ; sa forme a pu changer ; la classe est relative à son propre historique (un tiers de ses séances), pas une intensité absolue.

### E. Votre contribution personnelle (non noté)
_En deux lignes : qu'avez-vous fait personnellement dans l'équipe aujourd'hui ?_
RMYU : modèle Final2, quotas exacts et recopie des doublons, mélange de modèles des soumissions n° 5 et n° 6,
vérification des pistes (sélection imbriquée, relecture adversariale, fuite démasquée) et choix des deux soumissions finales.

- Scores : validation macro-F1 0,6462 ± 0,0054 (quotas appliqués au pli avec ses vrais comptes ; n° 5 : 0,6468) ; public 0,65396

## Soumission n° 7 : PersonStack HGB 3 graines + Final2 2 graines + doublons + quotas (17h58, code participant à renseigner)

Code participant : à renseigner · Équipe n° : à renseigner · Date : 30/10



### A. Dessinez le pipeline final de votre équipe (2 points)

_Du CSV brut jusqu'à submission.csv : chaque étape dans une boîte, reliées par des flèches. Écrivez dans chaque boîte ce qui est fait et avec quel outil (ex. « one-hot sur sport », « RandomForest, max_depth=12 »). Indiquez où se fait la validation._



    [ train.csv / test.csv ]

              |

              v

    [ variables relatives à la personne : ratio à sa médiane, écart, rang, par utilisateur + sport (pandas) ]

              |

              v

    [ étage 1 : HistGradientBoosting, niveau attendu de 0 à 2, hors pli sur le train ]

              |

              v

    [ contexte : labels et résidus des séances proches de la même personne, ridge par groupe, découpage interne pour éviter la fuite ]

              |

              v

    [ étage 2 : PersonStack HGB (3 graines) et Final2 (HGB, 2 graines), 3 probabilités chacun ]

              |

              v

    [ mélange : log-probabilités, poids 1 PersonStack, 0,3 par graine de Final2 ]   <-- validation : mêmes plis que src/validation.py, 5 folds stratifiés, graine 42, macro-F1, script séparé

              |

              v

    [ doublons exacts : label du train recopié ]
              |
              v
    [ quotas par utilisateur + sport : méthode hongroise sur les log-probabilités ]

              |

              v

    [ submission.csv ]



### B. Pourquoi votre modèle final fait-il mieux que votre première soumission ? (2 points)

_Nommez votre modèle final et expliquez en deux lignes comment il produit une prédiction. Puis citez deux décisions (variables, modèle, hyperparamètres, validation) et expliquez par quel mécanisme chacune a amélioré le score._

- Modèle final : mélange PersonStack HGB + Final2, avec quotas.

- Comment il prédit (deux lignes) : deux empilements de HGB comparent la séance aux habitudes de la personne et à ses séances proches, et donnent chacun trois probabilités. On moyenne leurs log-probabilités, puis on attribue les classes en respectant le nombre restant dans chaque groupe.

- Décision 1 : variables relatives à la personne et séances voisines → mécanisme : « difficile » dépend de la personne et de sa forme du moment, pas de la vitesse brute (0,568 → 0,6183 sans quotas, HGB 1 graine)

- Décision 2 : mélange de deux modèles et de 3 graines → mécanisme : leurs erreurs diffèrent en partie, la moyenne réduit la variance (0,6404 → 0,6505 avec quotas)



### C. Une suggestion de l'IA que vous avez vérifiée (1 point)

_Décrivez une suggestion de l'IA que votre équipe a rejetée ou corrigée, ou acceptée seulement après vérification. Qu'avez-vous vérifié, et comment ?_

- Suggestion : script de l'IA qui mélange PersonStack et Final2.

- Rejetée / corrigée / acceptée après vérification : corrigée.

- Ce qu'on a vérifié, et comment : Final2 seul donnait 0,3865 alors que LOG.md notait 0,6041. Les deux modèles ne rangeaient pas leurs classes dans le même ordre. Après remise dans l'ordre, on retrouve 0,6041.



### D. De la prédiction à la recommandation (1 point)

_L'utilisateur 4127 court habituellement à une vitesse médiane de 10 km/h. Trois séances possibles : (1) 5 km à plat à 9 km/h, (2) 12 km à 10 km/h avec 300 m de dénivelé, (3) 8 km à plat à 12 km/h. Vous voulez recommander une séance de récupération facile. Comment utiliseriez-vous votre modèle pour choisir ? Quelle séance vous attendez-vous à ce qu'il choisisse, et qu'est-ce qui pourrait le rendre faux pour cet utilisateur ?_

- Comment utiliser le modèle : construire les variables des trois séances pour cet utilisateur, sans quotas, et comparer la probabilité de `easy`.

- Séance attendue : la séance 1, plus lente que sa médiane, courte et à plat.

- Ce qui pourrait le rendre faux pour cet utilisateur : sa forme du moment, la chaleur ou un manque de séances semblables dans ses données ; la vraie mesure de l'effort n'est pas dans nos colonnes.



### E. Votre contribution personnelle (non noté)

_En deux lignes : qu'avez-vous fait personnellement dans l'équipe aujourd'hui ?_

À écrire par chaque membre.



- Scores : validation macro-F1 0,6505 ± 0,0064 (effectifs réels du fold, borne haute) ; public 0,65842

## Soumission n° 8 : moyenne de 5 modèles (PersonStack ×3 variantes + Final2 ×2), doublons et quotas (18h10, 3MAS)
Code participant : 3MAS · Équipe n° : à renseigner · Date : 30/10

### A. Dessinez le pipeline final de votre équipe (2 points)
_Du CSV brut jusqu'à submission.csv : chaque étape dans une boîte, reliées par des flèches. Écrivez dans chaque boîte ce qui est fait et avec quel outil (ex. « one-hot sur sport », « RandomForest, max_depth=12 »). Indiquez où se fait la validation._

    [ train.csv / test.csv ]
              |
              v
    [ variables relatives à la personne : rang, écart à sa médiane, écart-type, par utilisateur + sport (pandas, UserStats, sans la cible) ]
              |
              v
    [ 5 modèles empilés en 2 étages (HistGradientBoosting, scikit-learn) :
      PersonStack HGB seul, PersonStack HGB + ExtraTrees, PersonStackPlus (+ user_id catégoriel, + résidus de vitesse),
      Final2 graine 42, Final2 graine 7.
      Étage 1 : niveau attendu de la séance ; étage 2 : + labels des séances voisines de la même personne (même mois, même parcours),
      calculés hors pli pour ne jamais voir sa propre cible ]   <-- validation : src/validation.py, 5 folds stratifiés, graine 42, macro-F1
              |
              v
    [ moyenne simple des probabilités des 5 modèles (0,2 chacun) ]
              |
              v
    [ doublons : séance du test identique à une séance du train sur 16 colonnes → on recopie son label ]
              |
              v
    [ quotas par utilisateur + sport : tiers exacts sur train + test (N//3 easy, 2N//3 − N//3 moderate, le reste hard),
      moins ce que le train contient → affectation qui maximise la somme des log-probabilités (NumPy pur) ]
              |                     <-- validation des quotas : val + test de chaque pli traités comme non étiquetés, comme le vrai test
              v
    [ submission.csv ]

### B. Pourquoi votre modèle final fait-il mieux que votre première soumission ? (2 points)
_Nommez votre modèle final et expliquez en deux lignes comment il produit une prédiction. Puis citez deux décisions (variables, modèle, hyperparamètres, validation) et expliquez par quel mécanisme chacune a amélioré le score._
- Modèle final : moyenne de 5 modèles empilés (3 variantes de PersonStack, Final2 avec 2 graines) + doublons + quotas par utilisateur et sport.
- Comment il prédit (deux lignes) : chaque modèle compare la séance aux habitudes de la personne et à ses séances proches dans le temps, et donne 3 probabilités ; on fait leur moyenne.
  Puis, dans chaque couple personne + sport, on donne exactement le nombre d'easy, de moderate et de hard qui reste aux séances les plus probables pour chaque classe.
- Décision 1 : comparer la séance à la personne et à ses séances voisines (empilement) → mécanisme : « hard » est un tiers des séances **de cette personne**, et sa forme change dans le temps ;
  on compare donc à ses habitudes et à ses séances proches (validation 0,568 → 0,6207).
- Décision 2 : quotas exacts par utilisateur + sport → mécanisme : les classes sont des tiers exacts sur train + test, donc on sait combien de chaque classe il reste dans le test ;
  il ne reste qu'à bien classer les séances à l'intérieur de chaque personne (public 0,61791 (n° 2) → 0,65332 (n° 4)). Mélanger 5 modèles ajoute un peu (même protocole : 0,6251 → 0,6342), car leurs erreurs diffèrent en partie.

### C. Une suggestion de l'IA que vous avez vérifiée (1 point)
_Décrivez une suggestion de l'IA que votre équipe a rejetée ou corrigée, ou acceptée seulement après vérification. Qu'avez-vous vérifié, et comment ?_
- Suggestion : l'IA proposait ce mélange de 5 modèles comme meilleur que la n° 7 (0,6342 contre 0,6320, plus haut sur les 5 plis).
- Rejetée / corrigée / acceptée après vérification : acceptée après vérification en validation, puis corrigée par le score public.
- Ce qu'on a vérifié, et comment : on a refait la n° 7 avec exactement les mêmes plis et le même protocole, puis comparé pli par pli (+0,0007 à +0,0042). Mais l'écart moyen (+0,002) est plus petit que l'écart-type (±0,005),
  et la n° 7 avait 3 graines de PersonStack contre 1 ici. Au public : 0,65648, moins que la n° 7 (0,65842). Conclusion : le gain était dans le bruit ; la n° 7 reste la meilleure.

### D. De la prédiction à la recommandation (1 point)
_L'utilisateur 4127 court habituellement à une vitesse médiane de 10 km/h. Trois séances possibles : (1) 5 km à plat à 9 km/h, (2) 12 km à 10 km/h avec 300 m de dénivelé, (3) 8 km à plat à 12 km/h. Vous voulez recommander une séance de récupération facile. Comment utiliseriez-vous votre modèle pour choisir ? Quelle séance vous attendez-vous à ce qu'il choisisse, et qu'est-ce qui pourrait le rendre faux pour cet utilisateur ?_
- Comment utiliser le modèle : construire les variables des 3 séances comme pour le test (vitesse, distance, dénivelé comparés aux habitudes de 4127 en course, date du jour), prédire P(easy) pour chacune
  (sans quotas, qui n'ont de sens que pour un groupe de séances) et recommander celle qui a la plus forte P(easy).
- Séance attendue : la (1), 5 km à plat à 9 km/h : plus lente que sa médiane, courte, sans dénivelé. La (2) est longue avec 300 m de montée, la (3) est 20 % plus rapide que sa médiane.
- Ce qui pourrait le rendre faux pour cet utilisateur : la cible ressemble à sa fréquence cardiaque, que nous ne voyons pas (chaleur, fatigue, maladie) ; le modèle apprend des corrélations, pas des causes
  (on court parfois lentement parce qu'on est fatigué, donc une séance lente peut être « hard ») ; sa forme a pu changer depuis ses dernières séances connues ; et la classe est relative à son propre historique, pas absolue.

### E. Votre contribution personnelle (non noté)
_En deux lignes : qu'avez-vous fait personnellement dans l'équipe aujourd'hui ?_
3MAS : règles de travail avec l'IA (AGENTS.md), modèle PersonStackPlus (user_id catégoriel et résidus de vitesse),
puis mélange de 5 modèles avec un protocole de validation des quotas plus réaliste, soumis en n° 8.

- Scores : validation macro-F1 0,6342 ± 0,0051 (quotas sur val + test de chaque pli, non comparable aux comptes réels du pli ; même protocole : n° 7 refaite 0,6320) ; public 0,65648
