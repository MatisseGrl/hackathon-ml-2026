# LOG

Une ligne **par soumission Kaggle** (15 max) : obligatoire.
Les essais non soumis peuvent aussi être notés dans la section « Expériences en validation ».

## Seuils
- Gate 1 (15h50) : G1 = `____`
- Gate 2 (17h35) : G2 = `____` + matrice de confusion et analyse d'erreur
- G3 = `____`

## Soumissions Kaggle

| # | Heure | Qui (code) | Ce qui a changé | Pourquoi on pensait que ça aiderait | Val macro-F1 (moy ± std) | Score public |
|---|-------|------------|-----------------|-------------------------------------|--------------------------|--------------|
| 1 |       |            | HistGradientBoosting (300 itérations, lr 0,05) + `UserStats` : variables relatives à l'utilisateur et au sport (ratio à sa médiane, écart-type, rang) | « Difficile » est mesuré par personne (~1/3 de chaque classe par utilisateur) : comparer la séance aux habitudes de la personne doit battre la vitesse brute | 0,568 ± 0,008 | |
| 2 |       |            | `NeighborStack` : 2e étage HistGB qui ajoute les résidus et les niveaux des séances voisines de la même personne (même mois, semaine, heure, fenêtres ±1 et ±3 mois) | À vitesse égale, la dureté dépend de la forme du moment, que les séances proches de la personne révèlent | 0,6055 ± 0,0071 | 0,61791 |

## Expériences en validation (au moins 3 familles de modèles)

| Heure | Qui | Famille | Modèle et paramètres | Variables | Val macro-F1 (moy ± std) | Remarque |
|-------|-----|---------|----------------------|-----------|--------------------------|----------|
|       |     | Linéaire | LogisticRegression, max_iter=3000, standardisée | UserStats (46 variables) | 0,489 ± 0,004 | Trop simple : les effets sont non linéaires |
|       |     | Arbres / ensembles | HistGradientBoosting 300 it., lr 0,05 | UserStats (46 variables) | 0,568 ± 0,008 | Meilleur. RandomForest (300 arbres, min_samples_leaf=3) : 0,566 ± 0,003 |
|       |     | Distance | kNN k=25, standardisé | UserStats (46 variables) | 0,531 ± 0,007 | Sensible à l'échelle et aux variables inutiles |
| 16h20 |     | Empilement | NeighborStack (2 étages HistGB) | UserStats + voisins | 0,6055 ± 0,0071 | Base de comparaison, retrouvée à l'identique |
| 16h20 |     | Empilement | + séances strictement identiques (jumeaux) | + twin | 0,6073 ± 0,0055 | 99,6 % des doublons du train ont le même label ; 4,3 % du test a un jumeau dans le train |
| 16h20 |     | Empilement | + estimations par personne (noyaux, ridge par groupe) | + local | 0,6098 ± 0,0029 | La pente de la vitesse varie selon les gens (corrélation de -0,32 à +0,73) |
| 16h20 |     | Empilement | + jumeaux + par personne | + twin + local | 0,6140 ± 0,0036 | HGB régularisé : 0,6145 ± 0,0031 |
| 16h25 |     | Empilement | + voisins serrés, HGB régularisé seul | + close | 0,6183 ± 0,0053 | Altitude ajoutée pour reconnaître un même parcours |
| 16h25 |     | Vote | Moyenne HGB + ExtraTrees | toutes | 0,6195 ± 0,0044 | ExtraTrees seul : 0,6102 ± 0,0054 |
| 16h25 |     | Vote | **`PersonStack`** : moyenne + poids 1,1 sur `moderate` | toutes | 0,6207 ± 0,0028 | Confirmé sur `src/person_stack.py` avec `evaluate`. Poids choisi sur ces folds, donc un peu optimiste |
| 16h20 |     | Rejeté | Quotas d'un tiers par groupe | | 0,6037 | Moins bon que sans quotas |
| 16h20 |     | Rejeté | `UserStats` ajusté aussi sur les lignes du test | | 0,6146 ± 0,0049 | +0,0006 seulement, c'est du bruit, et on ne prétraite pas sur le test |
| 16h30 | 3MAS | Ensemble | Vote soft : HistGB (63 feuilles, lr 0,03, user_id catégoriel) + ExtraTrees (500 arbres, user_id one-hot) | UserStats sur train+test (sans cible) + résidus de vitesse par utilisateur-sport + user_id | 0,601 ± 0,005 | `essais/3MAS/modele_06.py`. user_id catégoriel seul : 0,568 → 0,589 |

## Qu'est-ce que « difficile » veut dire ?
_Hypothèses de l'équipe sur ce qui rend une séance easy / moderate / hard pour **cette personne** :_

- Chaque utilisateur a environ 1/3 de chaque classe (écart-type des parts : 0,04) : la cible est un niveau relatif à la personne, pas une vitesse absolue.
- Les variables comptent donc relativement aux habitudes de l'utilisateur dans le même sport (ratio à la médiane, écart-type, rang).
- Gain mesuré : HistGB avec colonnes brutes 0,525, avec variables relatives 0,558, avec rang et écart-type calculés dans chaque fold 0,568.

## Gate 2 : matrice de confusion et analyse d'erreur
_Coller la matrice de confusion (validation croisée, prédictions hors fold) :_

```
                 prédit easy   prédit moderate   prédit hard
vrai easy
vrai moderate
vrai hard
```

**Analyse d'erreur (au moins une phrase)** : (HistGB, validation croisée) la classe `moderate` est la plus mal prédite (F1 0,46 contre 0,63 pour `easy` et 0,62 pour `hard`) : les séances moyennes se confondent avec les deux extrêmes, car la frontière entre niveaux est floue autour des habitudes de la personne.

## Indices
- Indice 1 (16h20) : comparer une séance à ce que la personne fait d'habitude dans ce sport. Déjà exploité par `UserStats` et `PersonStack`, pas de gain à attendre.
- Indice 2 / 3 achetés ? (coût 0,5 / 1 pt) :

## Modèle final
- Soumission sélectionnée : #
- Pourquoi ce choix :
- Explication courte du modèle (pour la fiche papier) :
