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

## Expériences en validation (au moins 3 familles de modèles)

| Heure | Qui | Famille | Modèle et paramètres | Variables | Val macro-F1 (moy ± std) | Remarque |
|-------|-----|---------|----------------------|-----------|--------------------------|----------|
|       |     | Linéaire | LogisticRegression, max_iter=3000, standardisée | UserStats (46 variables) | 0,489 ± 0,004 | Trop simple : les effets sont non linéaires |
|       |     | Arbres / ensembles | HistGradientBoosting 300 it., lr 0,05 | UserStats (46 variables) | 0,568 ± 0,008 | Meilleur. RandomForest (300 arbres, min_samples_leaf=3) : 0,566 ± 0,003 |
|       |     | Distance | kNN k=25, standardisé | UserStats (46 variables) | 0,531 ± 0,007 | Sensible à l'échelle et aux variables inutiles |
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
- Indice 1 (16h20) :
- Indice 2 / 3 achetés ? (coût 0,5 / 1 pt) :

## Modèle final
- Soumission sélectionnée : #
- Pourquoi ce choix :
- Explication courte du modèle (pour la fiche papier) :
