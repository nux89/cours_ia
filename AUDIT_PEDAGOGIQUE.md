# Audit pédagogique et technique du cursus

Historique : premier audit du 4 septembre 2026 sur quatre cours et neuf notebooks. La mise à jour du 5 septembre ci-dessous décrit le périmètre actuel ; les sections suivantes conservent les constats et résultats historiques.

## Mise à jour : cohérence et couverture au 5 septembre 2026

Le cursus comprend désormais 11 modules et 21 notebooks. La revue a relié objectifs, prérequis, supports, exercices et critères dans [PARCOURS_ET_EVALUATION.md](PARCOURS_ET_EVALUATION.md). Le graphe a servi à repérer les relations entre données, mathématiques, attention et multimodal ; les corrections ont été vérifiées dans les sources, pas déduites du seul graphe.

### Corrections et enrichissements appliqués

- Mathématiques : préciser axes des tenseurs, cosinus de vecteurs non nuls, convexité et unicité, probabilité continue, hypothèses des lois et du TCL, erreur standard et dépendance des folds. Corriger le point-selle, expliciter biais–variance, dimensions de backpropagation, mécanisme d'Adam et conditions de convergence du Q-learning.
- Ajouter un [support de calculs guidés](00_fondements_maths_python/complements_mathematiques.md) : produit matriciel, dérivée MSE, softmax/entropie croisée, covariance/PCA, attention et pas d'optimisation avec exemples numériques.
- Développer [images, audio, vidéo et multimodal](01_nature_et_preparation_des_donnees/traitement_images_video_audio_multimodal.md) : formats industriels, unités, annotations, rééchantillonnage, STFT, alignement temporel, modes de fusion, absence de modalités et séparation par groupe.
- Ajouter l'[ouverture sourcée sur les architectures](03_deep_learning/architectures_emergentes.md), en distinguant innovation architecturale, objectif d'apprentissage et méthode générative. Les résultats des articles ne sont pas présentés comme des garanties universelles.
- Ajouter recherche/logique/CSP/Bayes, causalité/prévision/recommandation/GCN, et sept TP exécutables : raisonnement, causalité/prévision, recommandation/graphes, transfert, captioning, diffusion et récupération documentaire.
- Mettre à jour parcours, renvois, charge indicative et voies du projet final ; la numérotation historique est conservée, avec ordre de lecture explicite.

### Vérifications effectivement réalisées

- `scripts/validate_notebooks.py` : **21/21 notebooks exécutés de haut en bas sans erreur**, assertions comprises. Les sept nouveaux TP ont leurs sorties enregistrées ; le contrôle global n'a pas réécrit les notebooks existants.
- Validation de **24 fichiers Markdown pédagogiques et transversaux** : liens locaux, ancres et blocs de code valides. Les répertoires de dépendances et de skills ne font pas partie de ce contrôle.
- `git diff --check` : aucun problème d'espacement détecté au contrôle.
- `graphify update .` : graphe de code actualisé ; cette commande ne réalise pas une nouvelle extraction sémantique exhaustive des Markdown. Le graphe ne doit donc pas être considéré comme un inventaire exhaustif des nouveaux paragraphes.

Quelques résultats des nouveaux TP : fine-tuning 95,84 % contre 96,65 % depuis zéro sur ce split de chiffres (le transfert ne gagne pas systématiquement) ; captioning synthétique 100 % de légendes exactes contre 23 % avec images supprimées ; bruit DDPM de test MSE 0,302 contre 0,983 pour prédiction nulle ; effet causal injecté 2, estimé 1,986 après ajustement contre différence brute 4,431. Ce sont des contrôles pédagogiques sur petits jeux, pas des scores industriels.

### Limites restantes

La préparation de médias est un support détaillé avec exercices, mais pas un pipeline validé sur des codecs, capteurs et corpus industriels. Le captioning et la diffusion restent synthétiques ; les garanties du transfert nécessitent des expériences répétées. Le RAG indexe les onze cours principaux, produit des extraits cités et teste une abstention facile : il ne génère pas avec un LLM, n'indexe pas tous les compléments et ne prouve pas la résistance aux injections. Les architectures émergentes sont une ouverture bibliographique, sans implémentation dédiée. Aucun cours généraliste ne couvre intégralement la discipline ; une validation pédagogique avec des apprenants reste nécessaire.

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
