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
| 5 | 17h37 | RMYU | Fichier `sub_RMYU_blend5_egal_dup_quota_usersport.csv` (`essais/RMYU/submit_blend5.py`). Par rapport à la n° 4 (PersonStack HGB seul + quotas par utilisateur et sport) : moyenne **égale** des probabilités de 5 modèles (PersonStack graines 42 et 7, Final2, NeighborStack, Resid2), puis recopie du label des doublons exacts du train (16 colonnes, 326 séances), puis les mêmes quotas par utilisateur + sport (easy = N//3, moderate = 2N//3 − N//3, hard = le reste) | Les 5 modèles se trompent sur des séances en partie différentes : la moyenne annule une partie des erreurs, sans aucun poids à régler (des poids réglés ne font pas mieux une fois le biais de sélection retiré : 0,6471 contre 0,6468) | 0,6468 ± 0,0073 (quotas appliqués au pli avec ses comptes réels, comme pour la n° 4 ; même protocole : PersonStack seul 0,6406, mélange de la n° 3 0,6437). Protocole de la n° 3 : 0,6335 ± 0,0062 | 0,65523 |
| 6 | 17h59 | RMYU | Fichier `sub_RMYU_blend8_egal_dup_quota_usersport.csv` (`essais/RMYU/submit_mix.py`). Par rapport à la n° 5 : moyenne égale de 8 modèles au lieu de 5 (3 graines de plus : PersonStack 11, NeighborStack 7, Final2 7), mêmes doublons exacts et mêmes quotas par utilisateur + sport | Pas pour battre la n° 5 seule (équivalente en validation), mais comme SECOND choix final : elle se trompe sur d'autres séances (3,8 % de lignes différentes) et Kaggle garde le meilleur des deux scores privés ; gain attendu simulé +0,0013, contre +0,0008 avec la n° 3 | 0,6462 ± 0,0054 (quotas avec les comptes réels du pli ; n° 5 : 0,6468) ; 2 plis réunis : 0,6329 (n° 5 : 0,6322) | 0,65396 |

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
| 17h15 | RMYU | Validation | Protocole « effectifs réels » de la n° 4 rejoué sur les 5 plis communs : quotas appliqués au seul pli de validation avec ses vrais comptes par utilisateur + sport | | PersonStack 0,6406 (= n° 4) ; mélange de la n° 3 0,6437 | Les deux protocoles de l'équipe concordent. Audit : avec des groupes de la taille de ceux du test (2 ou 3 plis réunis), le niveau absolu baisse (0,628 à 0,631) et l'écart n° 5 − n° 3 vaut +0,0016 à +0,0019 (7/10 et 9/10 combinaisons) au lieu de +0,0031 |
| 17h15 | RMYU | Vote | **Moyenne égale de 5 modèles** (PersonStack graines 42 et 7, Final2, NeighborStack, Resid2) + doublons exacts + quotas par utilisateur + sport | toutes | **0,6468 ± 0,0073** | Soumission n° 5. Grille de poids : 0,6495 en apparence, mais 0,6471 en sélection imbriquée, donc poids égaux. Contre PersonStack seul : +0,006 (10/10 combinaisons de plis) |
| 17h10 | RMYU | Post-traitement | Règle d'arrondi des tiers quand N n'est pas multiple de 3 | | Neutre (0,6313 à 0,6318) | Vraisemblance des comptes du train : seule la règle « 1 en trop → hard ; 2 en trop → moderate + hard » ne contredit aucun groupe (« easy d'abord » : 1 groupe contredit ; `pd.qcut` : 2) |
| 17h20 | RMYU | Rejeté | Autres décodages des quotas : somme des probabilités, recalage de Sinkhorn, tri par niveau attendu E[y] | | −0,0002 à +0,0007 ; tri par E[y] : −0,0056 | Le décodage est saturé : seul le classement des probabilités compte |
| 17h45 | RMYU | Rejeté | Même parcours : vitesse relative aux séances du même trajet (utilisateur, sport, km, altitude, dénivelé), 2e niveau HGB ou logistique | + 30 variables parcours | 0,6456 (imbriqué) | Corrélation 0,13 avec le label, mais 0,03 avec l'erreur du mélange : déjà capté par les modèles (`essais/RMYU/lab2/a1`) |
| 17h45 | RMYU | Rejeté | Comparateur par paires : HGB qui dit laquelle de deux séances d'un même groupe est la plus intense (361 000 paires, 76 % bien ordonnées), puis comparaison aux 20 séances étiquetées les plus proches dans le temps | + paires | 0,6471 (imbriqué) | Recoupe les variables relatives et le contexte des voisines déjà présents (`lab2/a2`) |
| 17h45 | RMYU | Rejeté (fuite) | Seuils locaux : position du score d'une séance parmi les scores des séances étiquetées proches dans le temps | + seuils | 0,6559 (faux) | Fuite : le score hors pli d'une séance voisine vient d'un modèle entraîné avec le pli évalué, dont le contexte contient le label cherché. Sans les voisines du même mois : +0,0000 (`lab2/a3`) |
| 17h45 | RMYU | Rejeté | Transductif : lissage vers les séances non étiquetées voisines, vers les labels connus, pseudo-labels itérés | | 0,6452 à 0,6467 (imbriqué) | Perd dès que le lissage est fort (`lab2/a4`) |
| 17h45 | RMYU | Rejeté | Corrections ciblées après analyse d'erreur : bascule vers les labels des voisines, poids par segment, température selon le désaccord des modèles | | 0,6464 à 0,6487 (imbriqué) | Pic isolé (un cran à côté : 0,6461), du bruit. 56 % des erreurs sur les séances incertaines (P max < 0,52), aucun segment « cassé » (`lab2/a5`) |
| 17h51 | RMYU | Rejeté | Graines supplémentaires (PersonStack 11, NeighborStack 7, Final2 7) : moyenne égale de 8 modèles (règle fixée d'avance), ou poids par famille de la n° 5 | toutes | égal 8 : 0,6462 (1 pli), 0,6329 (2 plis réunis ; n° 5 : 0,6322) ; familles 8 : 0,6477 / 0,6322 | Équivalents à la n° 5 (±0,001) : pas un meilleur modèle à lui seul |
| 17h56 | RMYU | Sélection finale | Second choix de soumission (Kaggle garde le meilleur des deux scores privés) : gain attendu de max(n° 5, X) − n° 5 sur 3 000 « moitiés privées » simulées de 3 835 séances | | égal 8 : +0,0013 ; familles 8 : +0,0009 (agents indépendants : +0,0016) ; n° 3 : +0,0008 ; n° 4 : +0,0002 | Second choix : `sub_RMYU_blend8_egal_dup_quota_usersport.csv` (vaut la n° 5 en espérance, mais diffère sur 3,8 % des lignes) ; `essais/RMYU/second_pick.py` |

## Qu'est-ce que « difficile » veut dire ?
_Hypothèses de l'équipe sur ce qui rend une séance easy / moderate / hard pour **cette personne** :_

- Chaque utilisateur a environ 1/3 de chaque classe (écart-type des parts : 0,04) : la cible est un niveau relatif à la personne, pas une vitesse absolue.
- Les variables comptent donc relativement aux habitudes de l'utilisateur dans le même sport (ratio à la médiane, écart-type, rang).
- Gain mesuré : HistGB avec colonnes brutes 0,525, avec variables relatives 0,558, avec rang et écart-type calculés dans chaque fold 0,568.
- Le découpage en trois est exact par couple utilisateur + sport, sur train + test : parts de classes dispersées à 0,025, contre 0,048 au hasard et 0,024 attendu pour des terciles exacts vus à 75 %. Répartition déduite pour N séances : easy = N // 3, moderate = 2N // 3 - N // 3, hard = N - 2N // 3.
- (RMYU) Règle d'arrondi confirmée indépendamment par la vraisemblance des comptes du train : c'est la seule règle testée qui ne contredit aucun groupe (« easy d'abord » est contredite par 1 groupe, `pd.qcut` par 2). En validation, elle est neutre.
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
(Soumission n° 4, `sub_04_quotas.csv` : `PersonStack` HGB seul + quotas, prédictions hors fold, avec les effectifs réels de chaque fold, macro-F1 0,6404.)

```
                 prédit easy   prédit moderate   prédit hard
vrai easy            5163            1856            510
vrai moderate        1857            3947           1799
vrai hard             619            1736           5241
```
(Soumission n° 3 : mélange PersonStack + Final2, doublons et quotas inclus, 22 728 séances hors pli, `essais/RMYU/confusion.py`.)

**Analyse d'erreur (au moins une phrase)** : (soumission n° 4, `PersonStack` HGB + quotas, validation croisée) la classe `moderate` reste la plus mal prédite (F1 0,537 contre 0,688 pour `easy` et 0,696 pour `hard`) : les erreurs se font surtout entre classes voisines (3 518 `moderate` mal classées, 3 518 aussi prédites à tort `moderate`, contre 1 141 confusions directes `easy`/`hard`), car une séance moyenne se situe près des deux frontières du découpage de la personne.

(RMYU, soumission n° 3) `moderate` reste la classe la plus confondue (F1 0,52 contre 0,68 pour `easy` et 0,69 pour `hard`) : 1 857 séances moderate sont prédites easy et 1 799 hard, alors que easy et hard ne se confondent presque pas entre eux (510 et 619). Les erreurs sont donc presque toutes d'un seul niveau, à la frontière des tiers de la personne, là où une petite différence de fréquence cardiaque invisible dans nos variables fait changer la classe.

## Indices
- Indice 1 (16h20) : comparer une séance à ce que la personne fait d'habitude dans ce sport. Déjà exploité par `UserStats` et `PersonStack`, pas de gain à attendre.
- Indice 2 / 3 achetés ? (coût 0,5 / 1 pt) :

## Modèle final
- Soumission sélectionnée : n° 5 (`sub_RMYU_blend5_egal_dup_quota_usersport.csv`, public 0,65523) ; si Kaggle permet deux sélections : n° 5 + n° 6 (`sub_RMYU_blend8_egal_dup_quota_usersport.csv`, public 0,65396) (recommandation RMYU, à confirmer par l'équipe sur Kaggle)
- Pourquoi ce choix : meilleur score public et meilleure validation (0,6468, quotas avec les comptes réels du pli) ; aucune des pistes testées ensuite (parcours, paires, seuils locaux, transductif, corrections ciblées, graines) ne fait mieux au-delà du bruit. Le second fichier vaut la n° 5 en espérance mais se trompe sur d'autres séances : comme Kaggle garde le meilleur des deux scores privés, cela rapporte environ +0,001 attendu
- Explication courte du modèle (pour la fiche papier) : moyenne des probabilités de 5 modèles HistGradientBoosting empilés (variables relatives à la personne + labels des séances voisines dans le temps), recopie du label des doublons exacts du train, puis quotas exacts par utilisateur + sport (les classes sont des tiers exacts calculés sur train + test : on attribue exactement le nombre restant de chaque classe)
