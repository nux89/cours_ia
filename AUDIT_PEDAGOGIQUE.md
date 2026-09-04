# Audit pédagogique et technique du cursus

Audit réalisé le 4 septembre 2026 sur les quatre cours Markdown et les neuf notebooks du projet.

## Verdict initial

Le projet possédait une bonne progression générale — données, machine learning, deep learning, systèmes agentiques — et un ton accessible. Les analogies, les schémas textuels et les petits jeux de données rendaient les notions abordables. En revanche, la promesse de « maîtrise approfondie » dépassait ce que permettaient des notebooks courts, sans sorties exécutées, sans critères de réussite et avec plusieurs conclusions formulées comme certaines avant l'expérience.

Le défaut le plus grave se trouvait dans le TP de classification médicale : le jeu `load_breast_cancer` code `0 = malignant` et `1 = benign`, mais l'AUC et le `recall` optimisaient la classe `1`. Le texte affirmait donc surveiller les tumeurs malignes alors que le code privilégiait les cas bénins.

## Forces conservées

- Découpage modulaire lisible et progression cohérente.
- Alternance de théorie, intuition, code et visualisation.
- Jeux de données de taille raisonnable pour un ordinateur personnel.
- Présence de pipelines Scikit-Learn, de plusieurs familles de modèles et d'une introduction aux risques agentiques.
- Vocabulaire français accompagné des termes anglais utiles dans la documentation technique.

## Faiblesses observées

| Priorité | Constat initial | Risque pédagogique |
|---|---|---|
| Critique | Classe positive médicale inversée dans les métriques et la recherche d'hyperparamètres | Faire apprendre une pratique contraire à l'objectif métier annoncé |
| Critique | TP agentique fondé sur des traces écrites à l'avance, présentées comme du raisonnement et de l'auto-correction | Confondre orchestration simulée, appel d'outils structuré et autonomie réelle |
| Élevée | Calculatrice basée sur `eval`, schémas d'outils incomplets et autorisations non vérifiées | Normaliser des patrons fragiles dans un chapitre sur la sécurité |
| Élevée | Les huit notebooks initiaux n'étaient pas exécutés et ne comportaient ni sorties ni assertions de contrôle | Impossible de savoir si les conclusions correspondaient réellement aux résultats |
| Élevée | CNN dépendant du téléchargement de Fashion-MNIST | Échec hors ligne et validation difficile en salle de cours |
| Élevée | Entraînement et visualisation du MLP sur les mêmes observations | Confusion entre mémorisation du train et généralisation |
| Moyenne | Solutions placées immédiatement après les énoncés, sans cellule de départ ni critère de réussite | Activité trop passive et auto-évaluation difficile |
| Moyenne | Nombreux absolus : performance « garantie », RNN amnésique après un nombre fixe de pas, LSTM toujours supérieur, outil infaillible | Transformer des tendances empiriques en lois universelles |
| Moyenne | Pas de public explicite, de durée, de grille d'évaluation ni de projet intégrateur | Parcours difficile à planifier et à évaluer |
| Moyenne | Dépendances uniquement bornées par le bas | Résultats susceptibles de changer avec une future rupture d'API |
| Faible | Références absentes et plusieurs affirmations historiques ou industrielles non sourcées | Vérification et approfondissement difficiles |

## Axes d'amélioration appliqués

1. **Rétablir la justesse avant d'ajouter du contenu.** La classe « maligne » devient explicitement la classe positive ; les métriques, labels, courbes ROC et recherches d'hyperparamètres suivent cette convention.
2. **Rendre les TP reproductibles.** Les graines aléatoires, contrôles de formes et assertions sont visibles. Le CNN utilise le jeu `digits` embarqué dans Scikit-Learn et ne dépend plus d'un téléchargement.
3. **Séparer apprentissage et évaluation.** Les démonstrations prédictives utilisent des ensembles d'entraînement et de test distincts. Les conclusions chiffrées sont produites par le code.
4. **Passer de la lecture à l'activité.** Les exercices disposent d'une cellule `TODO`, d'une solution identifiée et de critères de réussite observables.
5. **Moderniser l'agentique.** Les décisions, appels d'outils et observations sont des objets structurés ; les arguments sont validés ; le calcul arithmétique n'utilise plus `eval` ; les limites de la simulation sont annoncées.
6. **Nuancer les affirmations.** Les règles générales distinguent conditions, exceptions et limites : scaling selon l'algorithme, CV comme estimation, attention quadratique, LSTM non garanti, multi-agents non automatiquement fiables.
7. **Ajouter un cadre pédagogique.** Le README précise le public, les prérequis, la charge indicative et les acquis. Un projet final et une grille sur 100 rendent l'évaluation explicite.
8. **Rendre la maintenance vérifiable.** `scripts/validate_notebooks.py` exécute tous les notebooks de haut en bas et échoue dès qu'une cellule échoue.
9. **Transformer les Markdown en supports autonomes.** Chaque module dispose désormais d'objectifs, d'un approfondissement méthodologique, de checklists, de questions de compréhension et d'une mini-étude de cas.
10. **Compléter le continuum deep learning.** Le support relie tenseurs, entraînement, Transformers, transfert, fine-tuning, encodeurs–décodeurs, CNN–LSTM, image-to-caption, alignement texte–image et diffusion conditionnelle ou latente.
11. **Structurer l'ingénierie agentique.** Le cours distingue base de connaissances, RAG, mémoire, outils et skills, puis couvre orchestration multi-agent, politiques d'autorisation, défense en profondeur, évaluation de trajectoire et déploiement progressif.
12. **Donner une carte de choix des modèles.** Le module ML classe les principales familles par problème — régression, classification, survie, clustering, réduction, anomalies, séries, recommandation, ranking et données non structurées — avec baselines, métriques, atouts et limites.
13. **Approfondir la compréhension des données.** Le premier module distingue échelle de mesure, type physique, structure logique et rôle métier ; il couvre les données tabulaires, temporelles, géospatiales, graphes, multimodales, imbriquées, creuses et censurées. Un nouveau TP relie Pandas, Polars, visualisations, matrices de Pearson et Spearman, V de Cramér, information mutuelle, sélection raisonnée et validation hors échantillon.

## Limites restantes et suite conseillée

- Les supports Markdown sont volontairement larges, mais les notebooks n'implémentent pas encore tous les approfondissements : captioning multimodal, fine-tuning d'un grand backbone, diffusion et RAG connecté exigeraient davantage de calcul, de données ou de services.
- L'inférence causale est distinguée de la prédiction sans constituer un cours complet de causalité ; elle mériterait un module dédié.
- Les résultats PyTorch peuvent varier légèrement selon la version, le système et l'accélérateur. Les seuils pédagogiques doivent rester des contrôles de cohérence, pas des garanties industrielles.
- Le TP agentique démontre une orchestration hors ligne et sûre ; il ne prouve pas la robustesse d'un agent relié à un LLM, à une base documentaire privée ou à des outils réels.
- Pour une utilisation diplômante, les durées, les critères, la charge cognitive et les exigences d'accessibilité doivent être testés avec un groupe pilote.

## Validation finale

Les neuf notebooks ont été exécutés de haut en bas dans un environnement Python 3.12 avec Scikit-Learn 1.9.0, Polars 1.44.1 et PyTorch 2.14.0. Toutes les cellules, y compris les solutions et assertions, ont terminé sans erreur.

Contrôles observés lors de cette exécution :

- prétraitement des données : 13 caractéristiques produites, ensembles disjoints et valeurs toutes finies ;
- exploration du crédit : les contrôles négatifs restent en retrait, la sélection métier atteint 0,770 de ROC-AUC moyenne en validation croisée, puis 0,700 de ROC-AUC et 0,360 de PR-AUC sur le test pour une prévalence de 0,117 ;
- classification : 212 cas malins explicitement recodés en classe positive, ROC et rappel calculés dans ce sens ;
- MLP : 96 % d'exactitude sur le test des deux lunes ;
- CNN hors ligne : 98,06 % d'exactitude sur le test `digits` ;
- série temporelle : MSE du RNN de 0,01482 contre 0,02068 pour la baseline de persistance ;
- comparaison séquentielle : le LSTM n'a gagné que légèrement et le GRU a perdu sur cette graine, ce qui confirme l'importance de ne pas annoncer un vainqueur à l'avance ;
- auto-encodeur : MSE de test réduite de 0,1327 pour l'entrée bruitée à 0,0508 après reconstruction ;
- agentique : outil inconnu, type invalide, division par zéro, budget de boucle et approbation obligatoire tous testés.

Ces valeurs sont des contrôles de cohérence de l'exécution, pas des seuils de performance générale.

Les sources utilisées pour la vérification sont listées dans [REFERENCES.md](REFERENCES.md).
