#START

# Conversations IA de MatisseGrl

## 2026-10-30 : pipeline et empilement (Claude Code) - résumé

- Lecture du sujet officiel, comparaison avec le squelette, plan d'équipe.
- `UserStats` : variables relatives à l'utilisateur et au sport, validation 0,568 ± 0,008.
- Suggestions rejetées après test : quotas par groupe (+0,005 au mieux), rang annuel seul (+0,002).
- Constat vérifié : environ 1/3 de chaque classe par couple utilisateur + sport (écart-type 0,02).
- `NeighborStack` : résidus et niveaux des séances voisines de la même personne, validation 0,6055 ± 0,0071.

## 2026-10-30 : recherche d'amélioration et PersonStack (Claude Code) - résumé

**MatisseGrl :** prends du recul comme un chef de lab et trouve des solutions pour nettement améliorer le score.

**Claude :** diagnostic des données, puis comparaison sur les 5 folds communs. Gain : 0,6055 -> 0,6207, sans saut net (plafond apparent autour de 0,62). Le modèle est rangé dans `src/person_stack.py`. Quotas par groupe et prétraitement sur le test rejetés.

**MatisseGrl :** retire la règle du code participant pour pousser, puis pousse tout sur main.

**Claude :** fiche de commit retirée de `AGENTS.md`. Poussé avant la fin de la revalidation avec `evaluate`, ce qui est signalé dans `LOG.md`.
