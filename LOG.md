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
| 3 | 16h59 | RMYU | Mélange des probabilités : 0,6 × `PersonStack` (`src/person_stack.py`) + 0,4 × `Final2` (`essais/RMYU/final2.py`, 2 étages HistGB : niveau attendu puis labels et résidus des séances voisines de la personne). Puis 2 post-traitements (`essais/RMYU/post.py`) : (a) une séance du test identique à une séance du train sur les 16 colonnes reçoit son label ; (b) affectation sous **quotas exacts par utilisateur** : nombre restant de chaque classe = (séances train + test de l'utilisateur) / 3 − séances déjà étiquetées dans le train (`essais/RMYU/quota.py`) | Les classes sont des tiers exacts par personne sur train + test : on connaît donc combien d'easy / moderate / hard restent dans le test pour chaque utilisateur. 99,6 % des doublons exacts du train ont le même label (327 séances du test en ont un). Deux modèles différents se corrigent l'un l'autre | 0,6314 ± 0,0055 (plis de `src/validation.py` ; doublons et quotas appliqués sur validation + test de chaque pli) | 0,65298 |
| 4 | 17h00 |            | Fichier `sub_04_quotas.csv` : `PersonStack` HGB seul (`use_trees=False`) + quotas par utilisateur et sport (`src/quotas.py`, méthode hongroise sur les log-probabilités) | Chaque couple utilisateur + sport est découpé en trois parts exactes sur train + test : les labels du train donnent le nombre de séances de chaque classe restant dans le test | 0,6344 ± 0,0026 (erreur d'arrondi simulée) à 0,6404 ± 0,0038 (effectifs réels du fold) ; mêmes folds, script séparé | 0,65332 |
| 6 | 17h58 |            | Fichier `sub_06_blend_hgb_final2x2_dup_quota.csv` : moyenne des log-probabilités, poids 1 × `PersonStack` HGB (3 graines) + 0,3 × `Final2` graine 42 + 0,3 × `Final2` graine 43, recopie des doublons exacts, quotas par utilisateur et sport | Par rapport au n° 4 : plusieurs graines et un second modèle réduisent la variance, leurs erreurs diffèrent en partie | 0,6505 ± 0,0064 (effectifs réels du fold, borne haute ; poids choisis sur ces plis ; script séparé) | 0,65842 |

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
| 16h50 |     | Empilement + quotas | **`PersonStack` HGB seul + quotas** | toutes | 0,6404 ± 0,0038 | Effectifs réels du fold (borne haute). Avec une erreur simulée d'une séance dans 2 groupes sur 3 : 0,6344 ± 0,0026. Soumis |
| 16h50 |     | Vote + quotas | Moyenne HGB + ExtraTrees + quotas | toutes | 0,6406 ± 0,0053 | Pas mieux que le HGB seul, qui est plus simple |
| 16h50 |     | Arbres + quotas | HGB sur UserStats seul + quotas | UserStats | 0,5926 ± 0,0049 | L'empilement reste nécessaire |
| 16h55 |     | Rejeté | HGB + quotas, variables physiques (puissance, VO2, saison, heure) | + physique | 0,6423 ± 0,0061 | Dans le bruit |
| 16h55 |     | Rejeté | HGB + quotas, autres réglages (7 ou 31 feuilles, lr 0,015) | toutes | 0,6399 à 0,6416 | Dans le bruit |
| 16h55 |     | Rejeté | Ordinal (2 classifieurs binaires) + quotas | toutes | 0,638 ± 0,0038 | Moins bon |
| 16h50 |     | Rejeté | Régression du niveau, coupe par quotas (erreur simulée) | toutes | 0,6251 ± 0,0058 | Moins bon que la méthode hongroise (0,6344) |
| 16h50 |     | Rejeté | 20 découpages internes au lieu de 5, HGB + quotas | toutes | 0,6377 ± 0,0063 | Pas de gain |
| 16h50 | RMYU | Empilement | `Final2` (`essais/RMYU/final2.py`) : étage 1 HistGB → niveau attendu hors pli ; contexte tiré des labels (k-NN de la personne, mêmes jour + heure / mois, fenêtres de résidus ±k mois) calculé par plis internes ; étage 2 HistGB | `essais/RMYU/feats.py` : rangs et écarts par utilisateur, utilisateur + sport et sport (train + test, sans cible), vitesse résiduelle vs distance / pente / date | 0,6041 ± 0,0070 | Version leave-one-out rejetée (0,58) : la moyenne d'un groupe sans la séance est anti-corrélée à son label, le modèle apprend un biais absent du test |
| 16h50 | RMYU | Post-traitement | Quotas exacts par utilisateur (affectation `linear_sum_assignment` sur −log P) | | Final2 0,6041 → 0,6123 ; PersonStack 0,6195 → 0,6236 | Contrairement aux « quotas d'un tiers par groupe » rejetés plus haut (0,6037), on retire les comptes déjà connus du train |
| 16h50 | RMYU | Post-traitement | + recopie des doublons exacts (16 colonnes) | | Final2 → 0,6174 ; PersonStack → 0,6239 | Doublons partiels (utilisateur, sport, durée, distance ; 83 % d'accord) rejetés : 0,615 à 0,6165 |
| 16h55 | RMYU | Vote | Moyenne 0,6 PersonStack + 0,4 Final2, puis doublons + quotas | toutes | **0,6314 ± 0,0055** | Plateau 0,630–0,631 pour un poids de 0,5 à 0,7 : pas un réglage fragile. Soumission n° 3 |
| 16h40 | RMYU | Rejeté | Variables de charge / forme (séances des semaines précédentes), voisins non étiquetés du test, décodage calibré | | 0,603 / 0,611 / +0,003 | Gains nuls ou dans le bruit (`essais/RMYU/lab/`) |
| 17h34 |     | Empilement + quotas | `PersonStack` HGB, moyenne de 3 graines + quotas | toutes | 0,6424 ± 0,0043 | Effectifs réels du fold. 1 graine : 0,6404 ± 0,0038 |
| 17h34 |     | Vote + quotas | **0,8 HGB 3 graines + 0,2 `Final2`**, moyenne des log-probabilités + quotas | toutes | 0,6471 ± 0,0050 | Effectifs réels du fold. Poids 0,3 à 0,5 : 0,6451 à 0,6458. Non soumis (`sub_05_candidate_blend.csv`) |
| 17h58 |     | Vote + quotas | **1 HGB 3 graines + 0,3 `Final2` x 2 graines**, doublons + quotas | toutes | 0,6505 ± 0,0064 | 0,6491 à 0,6505 pour un poids de 0,15 à 0,5. Avec NeighborStack, sans doublons : 0,6508. ExtraTrees jamais retenu. Soumission n° 6 |

## Qu'est-ce que « difficile » veut dire ?
_Hypothèses de l'équipe sur ce qui rend une séance easy / moderate / hard pour **cette personne** :_

- Chaque utilisateur a environ 1/3 de chaque classe (écart-type des parts : 0,04) : la cible est un niveau relatif à la personne, pas une vitesse absolue.
- Les variables comptent donc relativement aux habitudes de l'utilisateur dans le même sport (ratio à la médiane, écart-type, rang).
- Gain mesuré : HistGB avec colonnes brutes 0,525, avec variables relatives 0,558, avec rang et écart-type calculés dans chaque fold 0,568.
- Le découpage en trois est exact par couple utilisateur + sport, sur train + test : parts de classes dispersées à 0,025, contre 0,048 au hasard et 0,024 attendu pour des terciles exacts vus à 75 %. Répartition déduite pour N séances : easy = N // 3, moderate = 2N // 3 - N // 3, hard = N - 2N // 3.
- (RMYU) Les classes sont des **tiers exacts par personne, calculés sur train + test** : aucun utilisateur n'a plus de N_total / 3 séances d'une classe dans le train, et la part de `hard` par utilisateur varie exactement comme un tirage sans remise dans des tiers exacts (variance observée / attendue 0,97 ; 16 si on regroupe par année). On connaît donc le nombre de chaque classe restant dans le test : affectation sous quotas, +0,004 à +0,008.
- (RMYU) La cible est très probablement la **fréquence cardiaque moyenne** (données Endomondo, « mesuré par les capteurs »). D'où une dérive dans le temps propre à chaque personne (forme, capteur) : le label moyen des autres séances du même utilisateur le même mois est corrélé à 0,27 avec la séance, et à 0,37 le même jour et la même heure.

## Gate 2 : matrice de confusion et analyse d'erreur
_Coller la matrice de confusion (validation croisée, prédictions hors fold) :_

```
                 prédit easy   prédit moderate   prédit hard
vrai easy            5178           1833             518
vrai moderate        1728           4085            1790
vrai hard             623           1685            5288
```
`PersonStack` HGB seul + quotas, prédictions hors fold, avec les effectifs réels de chaque fold.

```
                 prédit easy   prédit moderate   prédit hard
vrai easy            5163            1856            510
vrai moderate        1857            3947           1799
vrai hard             619            1736           5241
```
(Soumission n° 3 : mélange PersonStack + Final2, doublons et quotas inclus, 22 728 séances hors pli, `essais/RMYU/confusion.py`.)

**Analyse d'erreur (au moins une phrase)** : (`PersonStack` HGB + quotas, validation croisée) la classe `moderate` reste la plus mal prédite (F1 0,537 contre 0,688 pour `easy` et 0,696 pour `hard`) : les erreurs se font surtout entre classes voisines (3 518 `moderate` mal classées, 3 518 aussi prédites à tort `moderate`, contre 1 141 confusions directes `easy`/`hard`), car une séance moyenne se situe près des deux frontières du découpage de la personne.

(RMYU, soumission n° 3) `moderate` reste la classe la plus confondue (F1 0,52 contre 0,68 pour `easy` et 0,69 pour `hard`) : 1 857 séances moderate sont prédites easy et 1 799 hard, alors que easy et hard ne se confondent presque pas entre eux (510 et 619). Les erreurs sont donc presque toutes d'un seul niveau, à la frontière des tiers de la personne, là où une petite différence de fréquence cardiaque invisible dans nos variables fait changer la classe.

## Indices
- Indice 1 (16h20) : comparer une séance à ce que la personne fait d'habitude dans ce sport. Déjà exploité par `UserStats` et `PersonStack`, pas de gain à attendre.
- Indice 2 / 3 achetés ? (coût 0,5 / 1 pt) :

## Modèle final
- Soumission sélectionnée : #
- Pourquoi ce choix :
- Explication courte du modèle (pour la fiche papier) :
