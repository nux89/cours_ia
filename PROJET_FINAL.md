# Projet final — Construire et auditer un mini-système d'IA

## Mission

Choisissez un problème de taille raisonnable : **ML tabulaire ou séquentiel**, **deep learning multimodal**, **système agentique documentaire**, **raisonnement symbolique/probabiliste**, ou **méthodes spécialisées**. Votre objectif est de produire une expérience traçable dont les choix, les limites et les risques peuvent être relus. La [matrice de compétences](PARCOURS_ET_EVALUATION.md) aide à choisir les TP préparatoires.

Le rendu peut prolonger l'un des jeux du cours ou utiliser un jeu public dont la licence et la provenance sont documentées.

## Livrables

1. Un notebook exécutable de haut en bas, sans cellule dépendant d'un état caché.
2. Une fiche de synthèse d'une page : besoin, population concernée, métrique principale, résultat de test et limite majeure.
3. Un fichier de dépendances ou une référence à l'environnement du cours.
4. Les artefacts propres à la voie choisie : fiche modèle et courbes d'apprentissage, ou registre de connaissances/outils/skills et politique d'autorisation.

## Étapes minimales attendues

- Formuler la cible, l'unité d'observation et le moment où la prédiction serait utilisée.
- Décrire la source, les valeurs manquantes, les doublons, les catégories et les risques de biais de sélection.
- Réserver un test final avant toute transformation apprise.
- Construire une baseline simple puis un modèle candidat.
- Choisir la métrique à partir du coût des faux positifs et faux négatifs.
- Ajuster les transformations et hyperparamètres sans utiliser le test.
- Présenter au moins un contrôle de robustesse : autre graine, sous-groupe, seuil, ablation ou comparaison à une baseline.
- Conclure avec ce que les résultats permettent d'affirmer et ce qu'ils ne permettent pas d'affirmer.

## Exigences propres à chaque voie

### A. ML tabulaire ou séquentiel

- Construire le prétraitement dans une pipeline sans fuite.
- Comparer au moins deux familles de modèles à une baseline.
- Justifier seuil, calibration ou intervalle d'incertitude selon le besoin.

### B. Deep learning multimodal

- Décrire l'encodeur, le décodeur ou la tête, avec les formes des tenseurs.
- Comparer extracteur figé et fine-tuning d'au moins un bloc, ou justifier une autre ablation.
- Pour image-to-caption : documenter tokenizer, masques, loss, décodage et fidélité visuelle.
- Pour une diffusion : documenter espace pixel ou latent, conditionnement, sampler, pas, guidance et protocole de comparaison.

### C. Système agentique

- Séparer base de connaissances, RAG, mémoire, outils et skills.
- Fournir les schémas d'outils, droits minimaux, budgets et critères d'arrêt.
- Tester au moins : source absente, injection indirecte, outil en panne, action non autorisée et boucle.
- Exiger citations ou abstention pour les réponses documentaires et une approbation exacte pour toute écriture sensible.
- Comparer l'orchestration retenue à un workflow ou agent unique plus simple.

### D. Raisonnement symbolique ou probabiliste

- Formaliser états/actions/coûts ou variables/domaines/contraintes ou loi factorisée.
- Comparer une méthode à une référence vérifiable : A* versus Dijkstra, backtracking versus énumération, posterior versus somme exacte.
- Fournir un cas limite : but inaccessible, contraintes contradictoires, observation rare.
- Distinguer preuve d'impossibilité, budget épuisé et incertitude probabiliste.

### E. Causalité, prévision, recommandation ou graphes

- Causalité : préciser l'estimand, le DAG et les hypothèses d'identification ; comparer brut/ajusté et discuter la sensibilité.
- Prévision : préciser origine et horizon ; comparer au naïf saisonnier avec backtest adapté.
- Recommandation : documenter expositions, candidats, split et exclusion des items vus ; calculer ranking et couverture.
- Graphes : préciser transductif/inductif, données visibles et arêtes test retirées si nécessaire ; comparer à une baseline sans propagation.

Pour les voies D/E, adapter les étapes génériques : une preuve exacte sur un CSP ne nécessite pas un split train/test ; une estimation causale ne se valide pas par un score prédictif seul. La grille juge alors la séparation entre conception et évaluation, les hypothèses et les contre-exemples.

## Grille d'évaluation sur 100

| Critère | Points | Attendu |
|---|---:|---|
| Cadrage et éthique | 15 | Objectif, usage, parties affectées, limites et risques explicites |
| Qualité des données | 15 | Diagnostic quantifié, provenance, traitement justifié |
| Étanchéité expérimentale | 20 | Train/validation/test corrects, transformations apprises hors test |
| Méthode et métriques | 15 | Baseline, métrique cohérente, comparaison loyale |
| Reproductibilité | 15 | Exécution complète, graines, dépendances, contrôles automatiques |
| Analyse des résultats | 15 | Interprétation liée aux sorties, incertitude et erreurs examinées |
| Communication | 5 | Notebook lisible, figures légendées, synthèse concise |

Un score élevé n'exige ni gros modèle ni API payante. Une fuite de données non signalée ou un notebook non exécutable plafonne la note à 50/100, car le résultat ne peut alors pas être considéré comme fiable.
