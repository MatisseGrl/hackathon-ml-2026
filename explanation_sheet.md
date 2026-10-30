# Fiche d'explication des modèles soumis

Une section **par soumission Kaggle**, ajoutée à la fin (même numéro que dans `LOG.md`).
Le but : que chaque membre puisse expliquer le modèle sur papier, sans écran.

Modèle d'une section :

```markdown
## Soumission n° <numéro> : <nom du modèle> (<heure>, <code participant>)
- **Changements par rapport à la soumission précédente** : ...
- **Architecture** (de la donnée brute à la prédiction, étape par étape) :
  1. ...
  2. ...
- **Variables utilisées et pourquoi** : ...
- **Modèle et paramètres, et pourquoi ce choix** : ...
- **Essayé puis rejeté (et pourquoi)** : ...
- **Scores** : validation macro-F1 <moyenne ± écart-type> ; public <score>
```

<!-- Les sections des soumissions commencent ici. -->

## Soumission n° 3 : PersonStack HGB + quotas (17h00, code participant à renseigner)
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
