# Module 10 — Causalité, prévision, recommandation et apprentissage sur graphes

## Objectifs, prérequis et méthode

Après les modules 0 à 3, ce module apprend à définir quatre problèmes que ne résout pas un simple changement de classifieur. Prévoir 10 à 15 heures avec les deux TP et leur correction. Chaque section suit le même raisonnement : question métier, hypothèses, représentation, modèle, évaluation et limites.

| Question | Objet estimé | Découpage pertinent |
|---|---|---|
| Une action change-t-elle le résultat ? | effet causal | protocole expérimental ou identification observationnelle |
| Que se passera-t-il demain ? | futur conditionnel au passé disponible | origines chronologiques |
| Quels items proposer ? | classement de candidats | interactions passées/futures, utilisateurs et items |
| Comment exploiter des relations ? | nœuds, arêtes ou graphes | dépend de l'usage transductif/inductif |

## 1. Causalité : prédire n'est pas intervenir

Prédire Y à partir de T estime une association. Demander ce qui se passerait si l'on imposait T est une intervention. Dans un essai d'offre commerciale, T indique la réception de l'offre et Y le montant acheté. Les clients déjà intéressés peuvent recevoir plus d'offres : une différence de ventes ne mesure alors pas automatiquement l'efficacité de l'offre.

### 1.1 Résultats potentiels et estimand

Y(1) et Y(0) représentent les résultats d'une même unité sous les deux traitements. On n'en observe qu'un. L'effet moyen ATE est E[Y(1)−Y(0)] ; ATT limite la moyenne aux unités traitées ; CATE conditionne sur un sous-groupe X. Définir population, traitement, horizon et résultat précède le choix de l'estimateur.

Dans un essai randomisé bien exécuté, l'affectation rend comparables les groupes en moyenne. Les abandons différentiels, non-respect du traitement et interactions entre unités peuvent encore compliquer l'analyse. L'effet de l'affectation (intention de traiter) n'est pas nécessairement l'effet de recevoir effectivement le traitement.

### 1.2 DAG et identification

Exemple de DAG : X→T, X→Y et T→Y. X est un facteur confondant observé avant traitement. L'ajustement vise à bloquer le chemin T←X→Y sans bloquer le mécanisme causal recherché.

Sous cohérence des résultats potentiels, absence de confusion non mesurée conditionnellement à X, positivité et absence d'interférence dans ce cadre simple :

ATE = E_X[E(Y|T=1,X) − E(Y|T=0,X)].

La positivité exige une probabilité non nulle d'observer les deux traitements dans les strates pertinentes. Sans comparaison possible, le modèle extrapole. Une variable médiatrice T→M→Y ne doit pas être ajustée pour estimer l'effet total ; un collider T→C←Y peut créer un biais si on le conditionne. « Ajouter toutes les colonnes » n'est donc pas une stratégie causale.

### 1.3 Estimation et incertitude

| Méthode | Principe | Fragilité |
|---|---|---|
| Régression / g-formule | prédire chaque unité sous T=0 et T=1, moyenner la différence | mauvaise spécification et extrapolation |
| Appariement | comparer des unités proches sur X | unités sans analogue, distance inadaptée |
| Pondération par propension | repondérer avec e(X)=P(T=1|X) | poids extrêmes et manque de recouvrement |
| Double robustesse | associer modèle de résultat et modèle de traitement | ne corrige pas la confusion cachée |
| Variables instrumentales | utiliser une variation exogène du traitement | exclusion et autres hypothèses difficiles à défendre |
| Différences de différences | comparer les changements de groupes | hypothèse de tendances parallèles et anticipation |

Un estimateur IPW de type Horvitz–Thompson utilise la moyenne de TY/e(X) − (1−T)Y/(1−e(X)). Inspecter la distribution des propensions, les poids et l'équilibre des covariables. Rogner les poids peut réduire la variance mais introduit un compromis et doit être documenté.

Un bootstrap produit un intervalle sous les hypothèses de rééchantillonnage retenues ; il ne valide pas les hypothèses causales. Pour des unités dépendantes, rééchantillonner les groupes pertinents. L'analyse de sensibilité demande comment l'effet changerait en présence d'un facteur caché ; les placebos cherchent des contradictions sans prouver l'absence de biais.

### 1.4 Exemple suivi et exercice

Le [TP causalité et prévision](01_causalite_prevision.ipynb) simule X, puis un traitement plus probable lorsque X est grand, et Y=2T+3X+bruit. L'effet causal injecté vaut 2. La différence brute mélange l'effet de T avec la sélection sur X ; une régression correctement spécifiée retrouve approximativement 2.

Exercice : cacher X puis augmenter sa contribution à Y. Correction attendue : le biais de l'estimation naïve augmente généralement ici ; plus de données réduit l'incertitude sans supprimer ce biais. Aucune performance prédictive ne rétablit à elle seule l'identification.

## 2. Séries temporelles : valider le futur réellement disponible

Une série possède un index temporel, une fréquence, un horizon de prévision et éventuellement plusieurs entités. Les trous dans les dates, changements de fuseau, révisions de mesures et délais de publication font partie de la définition du problème.

### 2.1 Structure et modèles

La tendance décrit une évolution lente ; la saisonnalité un motif lié au calendrier ; l'autocorrélation une dépendance aux valeurs passées. Une décomposition additive y=tendance+saison+résidu convient à une amplitude relativement stable. Une transformation logarithmique peut être utile pour une amplitude proportionnelle au niveau, avec gestion explicite des zéros et de la retransformation.

La stationnarité faible suppose moyenne et covariance stables dans le temps. Différencier peut éliminer une tendance, mais sur-différencier crée du bruit. L'ACF décrit la corrélation avec les retards ; la PACF isole une association au retard k après contrôle des retards intermédiaires dans le cadre linéaire.

| Modèle | Mécanisme | Conditions d'usage |
|---|---|---|
| Naïf / naïf saisonnier | répéter dernière valeur / même saison | baseline obligatoire |
| ETS | mise à jour récursive de niveau, tendance, saison | structure régulière, choix additif/multiplicatif |
| ARIMA(p,d,q) | autorégression, différenciation, erreurs passées | structure linéaire après transformation |
| SARIMA | ajouter retards et différences saisonniers | saison identifiable |
| État-espace / Kalman | état latent évolutif et équation d'observation | le filtre standard suppose un modèle linéaire gaussien |
| Régression sur retards | features y(t−1), y(t−s), calendriers | variables connues à l'origine de prévision |
| Réseaux séquentiels | représentation apprise de fenêtres | assez de séries/données et validation stricte |

Exemple AR(1) : y_t = c + φ y_(t−1) + ε_t. Si |φ|<1 et le bruit est stationnaire, la moyenne stationnaire vaut c/(1−φ). ARIMA ajoute notamment différenciation et dépendance aux erreurs ; une forêt sur retards n'est pas un ARIMA même si elle prédit la même cible.

### 2.2 Un protocole sans fuite

1. Fixer l'origine t, l'horizon H et le délai réel d'accès aux covariables.
2. Construire les caractéristiques uniquement avec ce qui était connu en t.
3. Ajuster imputation, scaling et modèle sur les données antérieures.
4. Répéter sur plusieurs origines, puis réserver une dernière période intacte.
5. Publier les erreurs par horizon et par série, avec la baseline.

Une moyenne glissante de y qui inclut y_t fuit si la cible est y_t. Utiliser un décalage avant la fenêtre. Pour un H-step forecast émis en une fois, on ne peut pas fournir les vraies observations intermédiaires au modèle. Cela serait légitime pour une autre tâche : prévisions à un pas réémises chaque jour.

### 2.3 Métriques et intervalles

MAE conserve l'unité ; RMSE pénalise davantage les extrêmes. MAPE est instable près de zéro. MASE divise la MAE de test par l'erreur absolue moyenne d'une baseline naïve calculée sur le train (retard saisonnier s si adapté). Si ce dénominateur vaut zéro, MASE n'est pas défini.

La pinball loss pour un quantile τ est max(τe,(τ−1)e), e=y−q. Pour un intervalle à 90 %, rapporter couverture empirique et largeur. Une bonne couverture globale peut cacher de mauvais sous-groupes et ne garantit pas une couverture future sous dérive.

Le TP compare trois baselines/modèles sur une série saisonnière synthétique, avec validation chronologique et test terminal. Il évalue explicitement des prévisions à un pas.

## 3. Recommandation et ranking

Une interaction est un triplet utilisateur, item, temps, parfois accompagné d'une note ou d'une exposition. Une absence de clic ne signifie pas que l'utilisateur a vu puis rejeté l'item. On distingue notes explicites et feedback implicite.

### 3.1 Représentations et apprentissage

La popularité est une baseline robuste pour les nouveaux utilisateurs. Le contenu encode les attributs des items ; le filtrage collaboratif utilise les comportements communs. Un modèle hybride combine ces signaux pour le démarrage à froid.

La factorisation approxime une note par μ+b_u+b_i+p_uᵀq_i. Les facteurs p et q sont appris en minimisant les erreurs sur les notes **observées**, avec régularisation. Remplir arbitrairement toutes les notes manquantes par zéro puis appliquer une SVD change le problème.

En implicite, une pondération de confiance (comme dans ALS implicite) distingue l'observation d'interaction de son intensité. Le choix des négatifs échantillonnés influence la difficulté et l'interprétation des scores. Les encodeurs two-tower produisent des vecteurs utilisateur/item pour une récupération rapide ; un second étage peut classer les candidats avec davantage de features.

### 3.2 Évaluer un classement

Precision@k mesure la proportion de recommandations pertinentes. Recall@k mesure la part des items pertinents retrouvés. Pour pertinence binaire, DCG@k=Σ rel_i/log₂(i+1) et NDCG divise par le DCG idéal. Spécifier quoi faire lorsqu'un utilisateur n'a aucun item pertinent ; le TP n'évalue que les utilisateurs ayant une vérité test définie.

Le [TP recommandation et graphes](02_recommandation_graphes.ipynb) masque un item pertinent par utilisateur synthétique, exclut les items vus des candidats et compare popularité à similarité entre items. Ce test explique le mécanisme mais ne remplace pas un split temporel sur des logs réels.

Les métriques dépendent du catalogue candidat : un classement parmi 100 items n'est pas comparable à un classement parmi un million avec une autre distribution de négatifs. Compléter l'exactitude par diversité, couverture du catalogue, répétition, fraîcheur et impact sur les producteurs. Un test A/B peut mesurer un effet produit si son protocole est correctement randomisé ; les gains hors ligne ne le garantissent pas.

### 3.3 Exercice corrigé

Un utilisateur a deux items pertinents et reçoit [pertinent, non pertinent, pertinent]. Precision@3=2/3 et Recall@3=1. DCG=1+1/log₂(4)=1,5 ; IDCG=1+1/log₂(3), donc NDCG≈0,920. Recommander un item déjà consommé peut artificiellement augmenter le score si les candidats ne sont pas filtrés.

## 4. Apprentissage sur graphes

Un graphe G=(V,E) représente des entités et relations. Il peut être orienté, pondéré, temporel ou hétérogène. Un graphe de connaissances décrit des relations typées ; un réseau bayésien décrit une factorisation probabiliste ; un graphe de calcul décrit des opérations. Le même dessin ne leur donne pas la même sémantique.

Les tâches principales sont classification de nœuds, prédiction de liens et classification de graphes entiers. Commencer par degré, voisins communs, PageRank ou features tabulaires permet de tester si le message passing apporte réellement un gain.

### 4.1 Message passing et GCN

Une couche collecte des messages des voisins, les agrège de façon invariante à leur ordre, puis met à jour l'état du nœud. Somme, moyenne et maximum n'ont pas la même expressivité. Pour une GCN simple :

H^(l+1)=σ(D̃^(−1/2) Ã D̃^(−1/2) H^l W^l), avec Ã=A+I.

H a une ligne par nœud ; W est partagé entre les nœuds ; D̃ est la diagonale des degrés de Ã. Les self-loops conservent les propres features. Une couche communique sur un voisinage à un saut ; plusieurs couches augmentent le champ réceptif.

Le TP calcule cette normalisation et entraîne une petite GCN sur un graphe synthétique assortatif. Il compare à un MLP utilisant uniquement les features. Les labels test n'interviennent pas dans la loss ; en revanche toutes les features et la structure sont disponibles : c'est un protocole **transductif**, explicitement différent d'un déploiement sur nouveaux graphes.

### 4.2 Limites et fuites

L'homophilie signifie que les voisins tendent à partager une propriété ; elle n'est pas universelle. En hétérophilie, moyenner peut effacer le signal. Trop de couches peut homogénéiser les représentations (oversmoothing) ; des goulots d'étranglement peuvent comprimer trop d'information distante (oversquashing).

En prédiction de liens, retirer les arêtes test avant de calculer des features structurelles ou propager les messages. En temporel, utiliser seulement les liens connus à la date de décision. En classification de graphes, séparer les graphes et contrôler les entités communes. La normalisation globale et les identifiants peuvent également transporter de l'information indésirable.

### 4.3 Exercice corrigé

Pourquoi ajouter I ? Pour inclure le nœud lui-même. Pourquoi partager W ? Pour appliquer la même opération locale et permettre le traitement de tailles variables. Peut-on permuter les lignes de H seules ? Non : permuter aussi lignes et colonnes de A ; la sortie est alors permutée de la même manière (équivariance).

## 5. Approfondissements à situer sans les confondre

- **Survie** : modéliser un délai censuré ; Kaplan–Meier estime la survie, Cox un risque relatif sous hypothèse de proportionnalité. Ne pas traiter une personne non encore décédée comme immortelle. La censure doit être compatible avec les hypothèses de l'estimateur.
- **Vision dense** : détection = boîtes/classes, segmentation = classe par pixel. IoU, mAP et Dice répondent à des objets différents ; un score de classification ne les remplace pas.
- **Audio** : signal échantillonné, fenêtres temps-fréquence, alignement et décodage. Les partitions doivent contrôler locuteurs, sessions et appareils.
- **Apprentissage distribué** : plusieurs appareils synchronisent paramètres/gradients ; le fédéré ajoute des données distribuées entre participants, souvent hétérogènes. Ni l'un ni l'autre ne garantit la confidentialité.

Ces domaines sont des prolongements identifiés ; les citer n'en fait pas des spécialisations complètes dans ce cursus.

## 6. Évaluation et sources

Réussite : distinguer identification et estimation ; expliquer les colonnes connues à chaque date ; calculer NDCG ; identifier le protocole transductif et une fuite de graphe. Les [objectifs et preuves attendues](../PARCOURS_ET_EVALUATION.md) relient chaque bloc à son exercice.

Sources : [DoWhy — effets causaux](https://www.pywhy.org/dowhy/v0.13/user_guide/causal_tasks/estimating_causal_effects/index.html), [Hyndman et Athanasopoulos — validation temporelle](https://otexts.com/fpp3/tscv.html), [Kipf et Welling — GCN](https://arxiv.org/abs/1609.02907), [Hu, Koren et Volinsky — feedback implicite](https://doi.org/10.1109/ICDM.2008.22).
