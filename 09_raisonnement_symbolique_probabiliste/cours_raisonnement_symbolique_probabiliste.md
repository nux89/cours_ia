# Module 9 — Recherche, logique, contraintes et raisonnement probabiliste

## Objectifs et parcours

Ce module complète l'apprentissage statistique par des systèmes dont les états, règles et hypothèses sont explicites. À l'issue du parcours, vous saurez formaliser un problème de recherche, choisir une stratégie, vérifier une règle logique, résoudre un petit problème de contraintes et calculer une probabilité dans un réseau bayésien.

Prérequis : Python et probabilités du module 0. Prévoir 6 à 9 heures avec le [TP corrigé](01_recherche_contraintes_bayes.ipynb). Lire successivement recherche, logique, contraintes, incertitude, puis réaliser les exercices. Les nombres utilisés sont des exemples synthétiques.

## 1. Trois manières de construire une décision

Un classifieur apprend une fonction à partir d'exemples. Un moteur de règles applique des implications déclarées. Un planificateur explore des suites d'actions pour atteindre un but. Un système industriel peut combiner ces trois mécanismes : vision pour reconnaître une pièce, contraintes pour affecter une machine, planification pour organiser les mouvements.

| Problème | Représentation | Résultat attendu | Risque principal |
|---|---|---|---|
| Classification | caractéristiques et paramètres appris | classe ou probabilité | décalage des données |
| Recherche | états, actions, coûts, but | chemin ou plan | explosion combinatoire |
| Logique | faits, règles, sémantique | conséquence et justification | règles erronées/incomplètes |
| Contraintes | variables, domaines, contraintes | affectation faisable | contradiction ou espace immense |
| Inférence probabiliste | variables aléatoires, dépendances | distribution conditionnelle | modèle de dépendance incorrect |

L'IA symbolique n'est pas synonyme de programme entièrement écrit à la main : des règles peuvent être apprises et des solveurs utilisent des heuristiques. Inversement, un texte produit par un LLM n'est pas une preuve logique sans vérification formelle.

## 2. Formaliser une recherche

Un problème est défini par un état initial, un ensemble d'actions possibles, une transition, un test de but et un coût. L'état doit contenir tout ce qui détermine les transitions futures. Pour livrer des colis, la position seule ne suffit pas : carburant, chargement et colis déjà livrés peuvent être nécessaires.

Un **arbre de recherche** contient les chemins explorés ; un **graphe d'états** fusionne les états identiques. Deux chemins différents peuvent atteindre le même état. Une table du meilleur coût connu évite de recommencer inutilement et permet de rouvrir un état si un chemin meilleur apparaît.

| Algorithme | Priorité d'exploration | Garantie sous conditions | Limite |
|---|---|---|---|
| BFS, largeur | profondeur minimale | plus petit nombre d'actions si coût uniforme | mémoire exponentielle |
| DFS, profondeur | dernier état ajouté | trouve une solution sur graphe fini avec contrôle des cycles | pas nécessairement optimale |
| Coût uniforme / Dijkstra | coût cumulé g | coût optimal avec coûts non négatifs sur graphe fini | explore sans direction vers le but |
| Glouton | estimation h du coût restant | aucune garantie générale d'optimalité | se laisse tromper par h |
| A* | f = g + h | optimal avec conditions ci-dessous | mémoire et qualité de l'heuristique |

Avec facteur de branchement b et profondeur d, BFS peut stocker de l'ordre de b^d nœuds. Ne confondez pas cette complexité dans l'arbre implicite avec celle d'un parcours sur un graphe explicite de V sommets et E arêtes.

### 2.1 Comprendre A*

A* choisit dans une file de priorité le nœud dont f(n)=g(n)+h(n) est minimal. g est le coût déjà payé ; h estime le coût restant. Une heuristique **admissible** ne surestime jamais le vrai coût restant. Elle est **cohérente** si h(n) ≤ c(n,n') + h(n') pour chaque transition et h(but)=0.

Avec coûts non négatifs et un graphe fini, A* est optimal avec une heuristique admissible si les améliorations de coût peuvent rouvrir les états ; une heuristique cohérente permet de finaliser les états sans réouverture. On vérifie le but à sa sortie de la file, pas dès sa découverte.

Sur une grille à quatre directions, coût unitaire et sans téléportation, la distance de Manhattan |x−x_but|+|y−y_but| est admissible et cohérente. Avec diagonales à coût unitaire, elle peut surestimer ; utiliser une distance adaptée. Avec coûts variables, multiplier une borne sur le nombre de pas par le coût minimal d'un pas, pas par le coût moyen.

Exemple : S→A coûte 2, A→G coûte 2, S→B coûte 1, B→G coûte 7. Avec h(A)=2 et h(B)=0, A* explore B avant A mais trouve ensuite S→A→G de coût 4. Être exploré en premier ne signifie pas appartenir à la solution optimale.

### 2.2 Jeux et planification

Minimax choisit une action contre un adversaire supposé optimal ; alpha-bêta évite certaines branches sans changer la valeur minimax lorsque les conditions du même arbre sont conservées. Une profondeur bornée exige une fonction d'évaluation ; le résultat devient dépendant de cette approximation.

La planification classique représente les actions par préconditions et effets. « Charger un colis » exige le robot et le colis au même endroit, puis change la possession. Elle se distingue d'une simple recherche de chemin : les actions modifient plusieurs propriétés du monde. Les effets duratifs, ressources et observations incomplètes rendent le problème plus difficile.

## 3. Logique, connaissances et systèmes experts

Une proposition vaut vrai ou faux dans une interprétation. Avec P = « capteur valide » et Q = « alerte autorisée », P→Q n'affirme pas P. De P et P→Q on déduit Q : c'est le modus ponens. De Q seul on ne déduit pas P ; une autre règle pourrait autoriser l'alerte.

Une base de connaissances **implique** une formule si celle-ci est vraie dans tous ses modèles. La **satisfaisabilité** demande s'il existe au moins un modèle. Pour prouver KB ⊨ Q par réfutation, on cherche l'insatisfaisabilité de KB ∧ ¬Q.

La logique du premier ordre ajoute objets, prédicats et quantificateurs : ∀x, CapteurValide(x) ∧ TemperatureHaute(x) → Alerte(x). Une variable logique n'est pas une variable aléatoire. L'unification rapproche des expressions en substituant leurs variables ; elle ne mesure pas une similarité statistique.

### 3.1 Chaînage et preuve

Le chaînage avant part des faits et ajoute les conséquences jusqu'à stabilisation. Le chaînage arrière part d'une question et cherche quelles règles et sous-buts permettraient de la démontrer. Pour des règles de Horn positives finies sans génération infinie de nouveaux objets, l'ajout monotone de faits atteint un point fixe.

Une trace utile conserve la règle utilisée et les faits qui l'ont déclenchée. Un système expert associe base de connaissances, moteur d'inférence et interface d'explication. La difficulté industrielle est souvent la maintenance des règles : exceptions, versions, conflits et responsabilité de la décision.

**Monde ouvert** : un fait non connu n'est pas nécessairement faux. **Monde fermé** : l'absence d'un fait peut être traitée comme sa négation dans un cadre défini. Mélanger les deux provoque des erreurs, par exemple interpréter un consentement non renseigné comme un consentement obtenu.

### 3.2 Exercice corrigé

Faits : température_haute et capteur_valide. Règles : leur conjonction implique alerte ; alerte implique inspection. Peut-on conclure panne ?

Correction : on obtient alerte puis inspection. Panne n'est pas déductible sans règle supplémentaire. Inspection ne prouve pas panne. Ajouter une règle « inspection→panne » modifierait la connaissance métier, pas seulement le code.

## 4. Satisfaction de contraintes et optimisation

Un CSP comprend variables X₁…Xₙ, domaines D₁…Dₙ et contraintes. Pour planifier trois examens A, B et C sur deux créneaux : A≠B et B≠C sont satisfaisables avec A=C. Ajouter A≠C rend le problème impossible avec seulement deux créneaux.

Le **backtracking** affecte une variable, teste les contraintes déjà évaluables et revient en arrière en cas de conflit. Les heuristiques réduisent le travail : MRV choisit la variable ayant le moins de valeurs possibles ; le degré privilégie la variable contraignant beaucoup d'autres ; la valeur la moins contraignante préserve les choix futurs.

Le forward checking retire des domaines voisins les valeurs devenues impossibles. La cohérence d'arc exige que chaque valeur d'une variable ait au moins un support chez sa voisine. Être cohérent par arc ne prouve pas qu'une solution globale existe : le triangle à deux couleurs en est un contre-exemple.

| Formulation | Question | Exemple |
|---|---|---|
| CSP / SAT | existe-t-il une solution ? | affecter salles et créneaux |
| Optimisation sous contraintes | quelle solution minimise un coût ? | réduire retard et déplacement |
| Contraintes souples | quelles violations sont acceptables ? | préférences horaires pondérées |

Un score de préférence ne doit pas remplacer une contrainte dure comme une capacité physique. Un solveur doit distinguer « impossible », « solution trouvée » et « délai dépassé sans preuve ».

## 5. Réseaux bayésiens et incertitude

Un réseau bayésien est un graphe orienté acyclique avec une distribution conditionnelle par variable. Il factorise la loi jointe : P(X₁,…,Xₙ)=∏ᵢP(Xᵢ|Parents(Xᵢ)). Cette factorisation économise des paramètres quand les indépendances supposées sont raisonnables. Une arête ne constitue pas, à elle seule, une causalité.

Considérons Pluie→RouteMouillée et Arrosage→RouteMouillée, pluie et arrosage étant indépendants a priori. Avec P(P)=0,2 et P(A)=0,3, la table P(M|P,A) vaut respectivement 0,99, 0,9, 0,8, 0,01 pour (1,1), (1,0), (0,1), (0,0).

On somme les quatre scénarios : P(M)=0,06×0,99+0,14×0,9+0,24×0,8+0,56×0,01=0,383. P(P,M)=0,1854, donc P(P|M)≈0,484. Une observation modifie la croyance ; elle n'a pas fait pleuvoir.

### 5.1 Indépendances conditionnelles

- Chaîne A→B→C : conditionner sur B bloque le chemin.
- Fourche A←B→C : B peut expliquer une association entre A et C ; conditionner sur B bloque ce chemin.
- Collision A→B←C : le chemin est bloqué sans observation ; conditionner sur B, ou certains descendants, peut créer une dépendance.

Dans l'exemple, sachant la route mouillée, apprendre que l'arrosage fonctionnait peut diminuer la probabilité de pluie : c'est l'explication alternative. La d-séparation généralise ces règles aux chemins d'un DAG.

### 5.2 Calculer et apprendre

L'énumération marginalise toutes les configurations cachées. L'élimination de variables réutilise des facteurs intermédiaires ; son coût dépend fortement de la structure et de l'ordre d'élimination. L'échantillonnage approche les probabilités lorsque l'inférence exacte est trop coûteuse, mais demande diagnostics et incertitude Monte Carlo.

Les tables conditionnelles peuvent être données par des experts ou estimées. Un lissage évite les zéros issus de petits effectifs. Apprendre une structure à partir d'observations peut produire plusieurs graphes statistiquement équivalents : les orientations causales exigent des hypothèses supplémentaires.

## 6. Combiner raisonnement, perception et agents

Une architecture hybride peut suivre : perception probabiliste → état estimé → solveur de contraintes → plan → outil d'exécution → nouvelle observation. Un LLM peut proposer un plan, mais le validateur teste les préconditions et les autorisations avant l'exécution. Conserver provenance, version des règles et motifs de refus permet d'auditer la décision.

En robotique, il faut ajouter localisation, fusion de capteurs, contrôle et cinématique. Le présent module enseigne la couche décisionnelle ; il ne constitue pas un cours complet de commande robotique. Un plan optimal dans une grille n'est pas une preuve de sûreté d'un robot physique.

## 7. TP, évaluation et limites

Le [TP recherche, contraintes et Bayes](01_recherche_contraintes_bayes.ipynb) implémente A*, compare à Dijkstra, détecte un CSP impossible et reproduit l'inférence numérique. Réussite : coût optimal identique, contraintes satisfaites, absence de solution correctement identifiée et probabilité normalisée.

Questions : pourquoi h=0 transforme A* en coût uniforme ? Pourquoi une corrélation ne donne-t-elle pas l'orientation causale ? Pourquoi un délai dépassé ne prouve-t-il pas l'impossibilité ?

Corrections : la priorité devient g ; plusieurs structures et facteurs confondants peuvent produire la même association ; une branche non explorée peut contenir une solution.

Sources : [AIMA, code des auteurs](https://github.com/aimacode/aima-python), [cours CS188 de Berkeley](https://inst.eecs.berkeley.edu/~cs188/textbook/). Pour l'intervention causale, poursuivre avec le [module 10](../10_methodes_specialisees/cours_methodes_specialisees.md).
