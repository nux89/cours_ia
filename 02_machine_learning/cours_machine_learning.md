# Module 2 : Fondements Théoriques et Algorithmiques du Machine Learning

> L'apprentissage automatique construit des règles de prédiction ou de décision à partir d'exemples, d'un objectif mesurable et d'hypothèses de modélisation. Il ne supprime ni la programmation, ni le besoin de spécifier correctement le problème.

**Objectifs du module.** À l'issue de ce chapitre, vous saurez formaliser une tâche supervisée ou non supervisée, établir une baseline, relier perte d'entraînement et métrique métier, choisir une famille de modèles, régler ses hyperparamètres sans contaminer le test et présenter une évaluation avec incertitudes et limites.

**Prérequis.** Le module 1, algèbre linéaire élémentaire et probabilités de base. Le notebook associé met en œuvre les pipelines, la validation croisée, la classification, la régression et la réduction de dimension.

---

## 📖 Le Dico du Débutant (Jargon Buster)
Voici les termes clés du Machine Learning expliqués sans jargon :
- **Algorithme vs Modèle** : L'algorithme est la *recette de cuisine* (ex: la formule mathématique de la régression). Le modèle est le *gâteau cuit* (l'objet final qui a mémorisé les données de votre entreprise et qui est prêt à faire des prédictions).
- **Entraînement (*Training / Fit*)** : La phase d'apprentissage pendant laquelle le modèle observe les exemples et ajuste ses réglages internes pour réduire ses erreurs.
- **Inférence (*Inference / Predict*)** : L'utilisation du modèle en conditions réelles sur de nouvelles données qu'il n'a jamais vues.
- **Paramètre** : Un nombre interne calculé et ajusté automatiquement par le modèle (ex: la pente d'une droite, le poids d'une caractéristique).
- **Hyperparamètre** : Un bouton de réglage que le développeur choisit manuellement **avant** l'entraînement (ex: le nombre d'arbres dans une forêt, la profondeur maximale d'un arbre).
- **Généralisation** : La capacité du modèle à réussir ses prédictions sur des données futures et inconnues, et pas seulement sur celles qu'il a déjà apprises par cœur.

---

## Table des Matières
1. [Taxonomie de l'Apprentissage Automatique](#1-taxonomie-de-lapprentissage-automatique)
2. [La Théorie Statistique de l'Apprentissage : Le Compromis Biais-Variance](#2-la-théorie-statistique-de-lapprentissage--le-compromis-biais-variance)
3. [Les Métriques d'Évaluation Rigoureuses (Exemple Chiffré Pas à Pas)](#3-les-métriques-dévaluation-rigoureuses-exemple-chiffré-pas-à-pas)
4. [Algorithmes Supervisés Fondamentaux](#4-algorithmes-supervisés-fondamentaux)
5. [Algorithmes Non Supervisés Fondamentaux](#5-algorithmes-non-supervisés-fondamentaux)
6. [Méthodologie de Validation et Réglage des Hyperparamètres](#6-méthodologie-de-validation-et-réglage-des-hyperparamètres)
7. [Du Besoin Métier à la Baseline](#7-du-besoin-métier-à-la-baseline)
8. [Risque Empirique, Pertes et Régularisation](#8-risque-empirique-pertes-et-régularisation)
9. [Choisir et Interpréter les Métriques](#9-choisir-et-interpréter-les-métriques)
10. [Panorama et Choix des Algorithmes](#10-panorama-et-choix-des-algorithmes)
11. [Validation Avancée et Incertitude](#11-validation-avancée-et-incertitude)
12. [Déséquilibre, Seuils et Coûts d'Erreur](#12-déséquilibre-seuils-et-coûts-derreur)
13. [Interprétabilité, Causalité et Robustesse](#13-interprétabilité-causalité-et-robustesse)
14. [Cycle de Vie et Suivi en Production](#14-cycle-de-vie-et-suivi-en-production)
15. [Checklist et Questions de Compréhension](#15-checklist-et-questions-de-compréhension)

---

## 1. Taxonomie de l'Apprentissage Automatique

### 💡 Programmation Classique vs Machine Learning
- **En programmation classique** : L'humain écrit les règles à la main en code (`if âge > 18 and revenu > 2000: accorder_credit()`). Si le monde change, l'humain doit réécrire des milliers de lignes de code.
- **En Machine Learning** : On fournit des exemples et un objectif. L'algorithme ajuste une famille de fonctions selon les données, la fonction de perte et les contraintes choisies par le concepteur.

```text
Programmation Classique :  [Règles écrites à la main] + [Données]  ──────► [Réponses]
Machine Learning :         [Données historiques]       + [Réponses] ──────► [Règles apprises (Modèle)]
```

### Les Trois Grandes Familles d'Apprentissage
```
                        MACHINE LEARNING
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
   Supervisé              Non Supervisé           Par Renforcement
 (Avec corrigé)          (Sans corrigé)        (Récompenses / Punitions)
        │                       │                       │
    ┌───┴───┐               ┌───┴───┐                   ▼
    ▼       ▼               ▼       ▼             Agents & Jeux
Régression Classification Clustering Réduction    (AlphaGo, Robots)
(Nombre)    (Catégorie)   (Groupes) de Dimension
```

1. **Apprentissage Supervisé (L'analogie de l'élève avec corrigé)** :  
   On fournit à l'algorithme des exercices accompagnés de leurs solutions exactes : $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$.
   - **Régression** : La réponse $y$ est un **nombre continu** (ex: deviner le prix d'un appartement en euros, prédire la consommation électrique en kWh).
   - **Classification** : La réponse $y$ est une **étiquette ou catégorie** discrète (ex: est-ce un spam ou non ? Ce grain de beauté est-il bénin ou malin ?).
2. **Apprentissage Non Supervisé (L'analogie de l'archéologue)** :  
   L'algorithme reçoit uniquement des données brutes $\{\mathbf{x}_i\}_{i=1}^n$ **sans aucune étiquette ni corrigé**. Son rôle est de découvrir la structure cachée ou de regrouper les profils similaires.
   - **Clustering (Partitionnement)** : Regrouper les clients d'un site en 4 profils types (les acheteurs impulsifs, les chasseurs de soldes, etc.).
   - **Réduction de dimension** : Résumer 50 caractéristiques d'un individu en 2 ou 3 axes synthétiques pour pouvoir les afficher sur un graphique 2D.
3. **Apprentissage par Renforcement (L'analogie du dressage d'un chiot)** :  
   Un agent virtuel effectue des actions dans un environnement. S'il réussit une action, il reçoit une récompense ($+1$) ; s'il échoue, une pénalité ($-1$). Il apprend par essais et erreurs successifs (utilisé pour les jeux vidéo, la robotique et l'alignement des LLMs avec RLHF).

---

## 2. La Théorie Statistique de l'Apprentissage : Le Compromis Biais-Variance

Le compromis biais-variance est le dilemme central de tout projet de Machine Learning.

### 🎭 L'Analogie des Trois Étudiants
Considérez trois élèves préparant un examen :
1. **L'élève fainéant (Biais élevé / Sous-apprentissage / *Underfitting*)** :  
   Il n'a appris qu'une seule règle simpliste : *"Je coche la réponse B à toutes les questions"*.  
   - *Résultat* : Il a une très mauvaise note aux entraînements ($4/20$) et une très mauvaise note au vrai examen ($4/20$). Son modèle est trop pauvre pour saisir la réalité.
2. **L'élève perroquet (Variance élevée / Surapprentissage / *Overfitting*)** :  
   Il a mémorisé par cœur chaque virgule, chaque numéro de page et même les fautes d'orthographe des annales d'entraînement.  
   - *Résultat* : Il obtient un fabuleux **$20/20$** aux exercices d'entraînement. Mais le jour de l'examen officiel, le professeur change légèrement l'énoncé : l'élève est totalement désorienté et obtient un désastreux **$03/20$**. Il n'a rien compris à la logique, il a simplement mémorisé le bruit !
3. **L'élève modèle (L'équilibre parfait)** :  
   Il a compris les principes fondamentaux. Il fait quelques petites erreurs d'inattention ($17/20$ aux entraînements), mais il obtient une excellente note ($16/20$) au vrai examen. Il sait **généraliser**.

```
  Erreur
    ▲
    │ \                                 /  Erreur sur Test
    │  \                               /
    │   \     Zone Optimale           /
    │    \         │                 /    Surapprentissage (Overfitting)
    │     \        ▼                /     (Le perroquet)
    │      \       *               /
    │       \_____________________/
    │  Sous-apprentissage (Underfitting)
    │  (Le fainéant)
    └────────────────────────────────────────► Complexité du modèle
```

### 2.1 La Formule Mathématique de Décomposition de l'Erreur
Sous une perte quadratique, pour une entrée fixée, un bruit conditionnel centré à variance finie et un jeu d'apprentissage indépendant du bruit de test, l'erreur prédictive attendue se décompose en :
$$\text{Erreur Totale} = \underbrace{\text{Biais}^2}_{\text{Erreur de simplification}} + \underbrace{\text{Variance}}_{\text{Sensibilité excessive aux données de train}} + \underbrace{\sigma^2}_{\text{Bruit incompressible}}$$

L'espérance et la variance du prédicteur portent sur les jeux d'entraînement possibles. Ce n'est pas une identité générale pour l'accuracy en classification. Voir les [calculs guidés](../00_fondements_maths_python/complements_mathematiques.md), qui développent également covariance, PCA et gradients.

### 2.2 Régularisation : Comment Dompter le Surapprentissage ?
Pour empêcher un modèle linéaire de devenir un "perroquet" et d'accorder des coefficients démesurés à des détails insignifiants, on lui impose une pénalité mathématique :
- **Ridge (Norme $L_2$)** :  
  $$\mathcal{L}_{\text{Ridge}} = \text{MSE} + \alpha \sum_{j=1}^d w_j^2$$  
  *Effet visuel* : Rétrécit les poids vers zéro de façon douce (*shrinkage*). En pratique, les coefficients sont rarement exactement nuls ; leur amplitude dépend aussi de l'échelle des variables.
- **Lasso (Norme $L_1$)** :  
  $$\mathcal{L}_{\text{Lasso}} = \text{MSE} + \alpha \sum_{j=1}^d |w_j|$$  
  *Effet visuel* : Peut forcer certains coefficients à zéro et servir de sélection de variables. Cette sélection peut être instable lorsque des variables sont fortement corrélées.

---

## 3. Les Métriques d'Évaluation Rigoureuses (Exemple Chiffré Pas à Pas)

Ne jugez jamais un modèle sur sa seule *accuracy* !

### 🔍 Exemple Chiffré : Test de Dépistage d'une Maladie sur 10 Patients
Supposons un groupe de 10 personnes testées à l'hôpital.
- **La réalité du terrain (*Ground Truth*)** : 2 personnes sont réellement malades (1) et 8 personnes sont saines (0).
- **Les prédictions de notre modèle d'IA** :
  - Sur les 2 malades réels : il en détecte 2 (2 Vrais Positifs - VP).
  - Sur les 8 personnes saines : il en classe 7 correctement comme saines (7 Vrais Négatifs - VN), mais envoie 1 fausse alerte sur une personne saine (1 Faux Positif - FP).
  - Faux Négatifs (malades ratés) : 0 (FN = 0).

**La Matrice de Confusion :**
| | Prédit Sain (0) | Prédit Malade (1) | Total Réel |
| :--- | :---: | :---: | :---: |
| **Réellement Sain (0)** | **7 (VN)** | **1 (FP)** *(Fausse alerte)* | 8 |
| **Réellement Malade (1)** | **0 (FN)** *(Malade raté)* | **2 (VP)** *(Malade sauvé)* | 2 |
| **Total Prédit** | 7 | 3 | **10** |

#### 1. Exactitude (*Accuracy*)
$$\text{Accuracy} = \frac{VP + VN}{\text{Total}} = \frac{2 + 7}{10} = \frac{9}{10} = 90\%$$
*Interprétation* : Le modèle a raison 9 fois sur 10 au global.

#### 2. Précision (*Precision*) : La chasse aux fausses alertes
$$\text{Précision} = \frac{VP}{VP + FP} = \frac{2}{2 + 1} = \frac{2}{3} \approx 66.7\%$$
*Question à se poser* : *"Quand le modèle sonne l'alarme en criant 'Malade !', à quel point puis-je lui faire confiance ?"*  
Ici, il a sonné 3 fois l'alarme, mais seulement 2 personnes étaient malades ($66.7\%$).  
*Où est-ce crucial ?* Dans la détection de spam (si un email important de votre patron est classé en spam par erreur, c'est une fausse alerte inadmissible).

#### 3. Rappel (*Recall* ou Sensibilité) : La chasse aux oublis critiques
$$\text{Rappel} = \frac{VP}{VP + FN} = \frac{2}{2 + 0} = \frac{2}{2} = 100\%$$
*Question à se poser* : *"Sur tous les vrais malades qui existent sur Terre, quel pourcentage mon modèle a-t-il réussi à attraper ?"*  
Ici, il a détecté 100% des malades réels : aucun patient n'a été renvoyé chez lui avec un cancer non diagnostiqué !  
*Où est-ce crucial ?* Dans les systèmes de dépistage ou de sécurité lorsque le coût d'un faux négatif est élevé. En médecine, une métrique de notebook ne suffit jamais à établir une utilité clinique : seuil, calibration, validation externe et supervision humaine restent nécessaires.

#### 4. Score $F_1$ : L'arbitre équitable
Moyenne harmonique qui sanctionne sévèrement si l'une des deux métriques s'effondre :
$$F_1 = 2 \times \frac{\text{Précision} \times \text{Rappel}}{\text{Précision} + \text{Rappel}} = 2 \times \frac{0.667 \times 1.0}{0.667 + 1.0} = \frac{1.334}{1.667} \approx 80\%$$

---

## 4. Algorithmes Supervisés Fondamentaux

### 4.1 Régression Logistique : Le Classifieur Linéaire
Malgré son nom de "régression", c'est un algorithme de **classification**.  
Il calcule une somme pondérée des caractéristiques : $z = w_1 x_1 + w_2 x_2 + \dots + b$.  
Puis il passe ce score dans la **courbe sigmoïde** $\sigma(z) = \frac{1}{1 + e^{-z}}$ :
- Si $z = 0 \implies \sigma(0) = 0.5$ ($50\%$ de probabilité).
- Si $z = +5 \implies \sigma(5) \approx 0.99$ ($99\%$ de probabilité de classe 1).
- Si $z = -5 \implies \sigma(-5) \approx 0.01$ ($1\%$ de probabilité).

### 4.2 Arbres de Décision (Le jeu du "Qui est-ce ?")
L'arbre pose une série de questions binaires successives sur les colonnes pour scinder les données en sous-groupes de plus en plus purs :
```text
                  [ Revenu > 50 000 € ? ]
                        /        \
                    Oui /          \ Non
                      ▼              ▼
           [ Âge < 30 ans ? ]      [ Score Crédit > 700 ? ]
              /        \               /         \
          Oui/          \Non       Oui/           \Non
            ▼            ▼           ▼             ▼
       [Refusé]      [Accordé]   [Accordé]      [Refusé]
```
- *Avantage* : Compréhension immédiate par les humains (boîte blanche).
- *Défaut* : Si on laisse l'arbre grandir sans limite, il mémorise chaque cas individuel et fait un surapprentissage total.

### 4.3 Forêts Aléatoires (Random Forests) : La Sagesse des Foules
Au lieu de faire confiance à un seul arbre de décision (qui peut se tromper facilement), on crée une **assemblée de 100 à 500 arbres différents** :
1. Chaque arbre s'entraîne sur un sous-échantillon aléatoire des données (*Bootstrap*).
2. À chaque question, l'arbre n'a le droit de choisir que parmi un sous-ensemble aléatoire de colonnes.
3. Pour la prédiction finale, **tous les arbres votent à la majorité**.
Si les arbres sont suffisamment diversifiés, leur agrégation réduit généralement la variance par rapport à un arbre unique. Le gain dépend toutefois des données, des hyperparamètres et de la corrélation entre arbres.

### 4.4 Machines à Vecteurs de Support (SVM)
Le SVM cherche à tracer une frontière entre deux classes de façon à laisser **la bande de sécurité (la marge) la plus large possible** entre les points les plus proches des deux camps (appelés *vecteurs de support*).
- Si les données ne peuvent pas être séparées par une droite, le SVM utilise l'**astuce du noyau (*Kernel Trick*)** pour projeter mathématiquement les données dans un espace à plus de dimensions où elles deviennent séparables par un plan !

---

## 5. Algorithmes Non Supervisés Fondamentaux

### 5.1 K-Means (L'analogie des aimants)
Comment regrouper des clients en $K$ groupes naturels sans étiquette ?
1. On place $K$ points au hasard dans l'espace : ce sont les **centroïdes** (comme des aimants).
2. Chaque donnée est attirée et assignée à l'aimant le plus proche.
3. Chaque aimant se déplace pour se repositionner exactement au centre physique de son groupe de données.
4. On recommence les étapes 2 et 3 jusqu'à ce que les aimants ne bougent plus !

### 5.2 Analyse en Composantes Principales (PCA / ACP)
*L'analogie de l'ombre chinoise* :  
Imaginez que vous ayez une théière en 3D. Vous voulez en faire une photo en 2D (réduire de 3D à 2D) sans perdre sa silhouette reconnaissable. Si vous prenez la photo du dessus, on ne voit qu'un cercle (l'anse et le bec disparaissent). Mais si vous orientez la lumière sous le meilleur angle, l'ombre projetée sur le mur révèle à la fois le corps, l'anse et le bec.  
La PCA trouve les directions orthogonales qui conservent le maximum de **variance**. Variance et information utile ne sont pas synonymes : une direction de faible variance peut parfois être très prédictive.

---

## 6. Méthodologie de Validation et Réglage des Hyperparamètres

### Validation Croisée K-Fold (L'analogie des 5 examens blancs)
Au lieu de tester votre modèle sur un seul découpage Train/Test qui pourrait être trompeur (par chance ou malchance), on découpe les données en $K$ morceaux égaux (ex: 5 blocs de 20%).
On effectue 5 entraînements successifs en changeant à chaque fois le bloc de validation. Le jeu de test final reste à part jusqu'à la fin :
```
Tour 1 : [VALID] [TRAIN] [TRAIN] [TRAIN] [TRAIN] ──► Note 1 = 88%
Tour 2 : [TRAIN] [VALID] [TRAIN] [TRAIN] [TRAIN] ──► Note 2 = 91%
Tour 3 : [TRAIN] [TRAIN] [VALID] [TRAIN] [TRAIN] ──► Note 3 = 89%
Tour 4 : [TRAIN] [TRAIN] [TRAIN] [VALID] [TRAIN] ──► Note 4 = 92%
Tour 5 : [TRAIN] [TRAIN] [TRAIN] [TRAIN] [VALID] ──► Note 5 = 90%

Estimation CV sur ces plis = 90.0% ± 1.4% (elle n'est pas une garantie de performance future)
```

### GridSearchCV : L'explorateur méthodique
Pour comparer une grille finie d'hyperparamètres (ex: profondeur d'arbre = 5 ou 10 ? nombre d'arbres = 50 ou 100 ?), `GridSearchCV` teste chaque combinaison déclarée avec une validation croisée. Il sélectionne la meilleure option **parmi cette grille, pour la métrique et les plis choisis** ; il ne garantit pas un optimum global.

Pour approfondir et vérifier ces recommandations, consultez les sources du module dans [REFERENCES.md](../REFERENCES.md).

---

## 7. Du Besoin Métier à la Baseline

Un projet de ML ne commence pas par « quel algorithme utiliser ? », mais par une décision à améliorer. Formalisez :

- l'unité prédite et l'instant de prédiction ;
- l'entrée $X$, la cible $y$ et l'horizon ;
- l'action déclenchée par le score ;
- le coût des faux positifs et faux négatifs ;
- les contraintes de latence, mémoire, explicabilité et équité ;
- la manière dont la vérité terrain sera observée après déploiement.

### 7.1 Baselines indispensables

Une **baseline** est un niveau de référence à dépasser. Elle évite de célébrer un modèle complexe qui fait moins bien qu'une règle triviale.

| Tâche | Baseline simple |
| :--- | :--- |
| Régression | moyenne ou médiane du train |
| Classification | classe majoritaire ou fréquence de base |
| Série temporelle | dernière valeur, moyenne saisonnière |
| Classement | ordre aléatoire ou règle métier actuelle |
| Segmentation | absence de segmentation ou groupes métier existants |

Ajoutez si possible la **baseline opérationnelle** : performance du processus humain ou du système actuellement utilisé. Le bon critère n'est pas « le modèle est-il précis ? », mais « améliore-t-il la décision, à coût et risque acceptables ? »

### 7.2 Corrélation, prédiction et causalité

Un modèle supervisé apprend des régularités statistiques. Une variable peut prédire une cible sans la causer. Modifier cette variable ne garantit donc pas de modifier le résultat. Pour estimer l'effet d'une intervention, il faut un raisonnement causal : expérience randomisée lorsque c'est possible, hypothèses explicites et méthodes adaptées lorsque ce ne l'est pas.

---

## 8. Risque Empirique, Pertes et Régularisation

On cherche une fonction $f_\theta$ paramétrée par $\theta$ qui minimise une perte moyenne sur l'entraînement :

$$
\hat{R}(\theta)=\frac{1}{n}\sum_{i=1}^{n}\ell\left(y_i,f_\theta(x_i)\right)+\lambda\,\Omega(\theta)
$$

- $\ell$ mesure l'erreur sur un exemple ;
- $\Omega$ pénalise la complexité ;
- $\lambda$ règle le compromis ajustement–simplicité ;
- le risque empirique $\hat{R}$ approxime imparfaitement le risque futur.

### 8.1 Pertes courantes

| Problème | Perte fréquente | Propriété importante |
| :--- | :--- | :--- |
| Régression | erreur quadratique (MSE) | pénalise fortement les grandes erreurs |
| Régression robuste | erreur absolue (MAE) ou Huber | moins sensible aux valeurs extrêmes |
| Classification binaire | log-loss / entropie croisée | évalue les probabilités, pas seulement la classe |
| Classification multiclasse | entropie croisée | compare une distribution de classes |
| Classement | pertes pairwise/listwise | optimise l'ordre relatif |

La perte optimisée n'est pas forcément la métrique communiquée. On peut entraîner avec une log-loss différentiable et décider selon le rappel à un seuil métier.

### 8.2 L1 et L2

- **L2 (Ridge)** pénalise $\sum_j w_j^2$ : les coefficients sont réduits de façon lisse et les solutions sont plus stables lorsque des variables sont corrélées.
- **L1 (Lasso)** pénalise $\sum_j |w_j|$ : certains coefficients peuvent devenir nuls, ce qui réalise une sélection de variables dépendante du modèle.

La régularisation n'annule ni les biais de données ni une mauvaise validation. Sa force est un hyperparamètre à choisir sur la validation.

---

## 9. Choisir et Interpréter les Métriques

### 9.1 Régression

Pour $e_i=y_i-\hat{y}_i$ :

$$\mathrm{MAE}=\frac{1}{n}\sum_i |e_i|, \qquad
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_i e_i^2}$$

- **MAE** s'exprime dans l'unité de la cible et traite linéairement les erreurs.
- **RMSE** donne davantage de poids aux grandes erreurs.
- **$R^2$** compare le modèle à la moyenne sur le jeu évalué ; il peut être négatif sur test.
- **MAPE** est instable lorsque $y$ est nul ou très petit et asymétrique : ne l'utilisez pas automatiquement.

Rapportez aussi les quantiles des erreurs et leurs variations par segment. Une moyenne seule peut cacher quelques erreurs catastrophiques.

### 9.2 Classification binaire

À partir de la matrice de confusion :

$$\mathrm{précision}=\frac{VP}{VP+FP}, \qquad
\mathrm{rappel}=\frac{VP}{VP+FN}, \qquad
F_1=2\frac{\mathrm{précision}\times\mathrm{rappel}}{\mathrm{précision}+\mathrm{rappel}}$$

- privilégiez le **rappel** si manquer un positif coûte très cher ;
- privilégiez la **précision** si chaque alerte est coûteuse ;
- utilisez la courbe **ROC** pour étudier taux de vrais/faux positifs ;
- préférez souvent la courbe **précision–rappel** lorsque la classe positive est rare ;
- n'interprétez aucune AUC sans la fréquence de base et le point de fonctionnement réel.

### 9.3 Multiclasse et multilabel

- **macro** : moyenne identique entre classes, utile pour voir les classes rares ;
- **micro** : agrégation des décisions, dominée par les classes fréquentes ;
- **weighted** : moyenne pondérée par le support ;
- **top-k accuracy** : pertinente lorsque plusieurs suggestions peuvent être présentées.

Indiquez toujours la convention choisie. « F1 = 0,84 » est incomplet sans classe positive, moyenne et seuil.

### 9.4 Calibration

Un classifieur est calibré si, parmi les événements annoncés à 70 %, environ 70 % se réalisent. Vérifiez courbe de calibration, score de Brier et log-loss. Une bonne discrimination (AUC) ne garantit pas une bonne probabilité. Si une probabilité alimente une décision de risque, calibrez sur des données distinctes et revalidez.

---

## 10. Panorama et Choix des Algorithmes

Il n'existe pas de « meilleur modèle » universel. Le choix dépend du type de cible, du volume, de la dimension, de la structure des données, du coût des erreurs, de la latence et du besoin d'explication. Les tableaux suivants constituent un panorama représentatif, pas une liste mathématiquement exhaustive.

### 10.1 Carte rapide : problème → modèles → métriques

| Type de problème | Première baseline | Modèles candidats | Métriques fréquentes |
| :--- | :--- | :--- | :--- |
| Régression continue | moyenne/médiane | linéaire, Ridge/Lasso, arbres, boosting, SVR, k-NN | MAE, RMSE, $R^2$ |
| Classification binaire | classe majoritaire/fréquence | logistique, arbre, forêt, boosting, SVM, Naive Bayes | précision, rappel, F1, PR-AUC, ROC-AUC, log-loss |
| Classification multiclasse | classe majoritaire | logistique multinomiale, arbres, boosting, SVM, k-NN | F1 macro/micro, matrice de confusion, top-k |
| Classification multilabel | fréquence par label | one-vs-rest, chaînes de classifieurs, arbres multi-sorties | F1 micro/macro, Hamming loss |
| Comptage positif | moyenne du compte | Poisson, binomiale négative, Tweedie, boosting | déviance, MAE, calibration |
| Temps avant événement | Kaplan–Meier | Cox, forêts de survie, modèles paramétriques | concordance, Brier temporel |
| Clustering | partition métier | K-means, GMM, hiérarchique, DBSCAN, spectral | silhouette, stabilité, utilité métier |
| Réduction de dimension | aucune réduction | PCA, SVD, NMF, ICA, UMAP, auto-encodeur | variance reconstruite, erreur, voisinages |
| Détection d'anomalies | règle métier | Isolation Forest, LOF, One-Class SVM, auto-encodeur | précision@k, rappel, PR-AUC si labels |
| Prévision temporelle | dernière valeur/saisonnalité | ETS, ARIMA, boosting sur lags, RNN/LSTM, Transformer | MAE/RMSE, MASE, pinball loss |
| Recommandation | popularité | filtrage collaboratif, factorisation, contenu, two-tower | Precision@k, Recall@k, NDCG, couverture |
| Classement (*ranking*) | ordre métier | pairwise/listwise ranking, arbres de ranking | MAP, MRR, NDCG |
| Texte | longueur/fréquences simples | TF-IDF + linéaire, Naive Bayes, Transformer | selon classification, extraction ou génération |
| Image | pixels aplatis comme baseline | HOG + SVM, CNN, Vision Transformer | accuracy/F1, IoU, mAP selon la tâche |

La métrique n'est jamais choisie uniquement à partir du modèle : elle découle de la décision et des erreurs possibles.

### 10.2 Régression : prédire une quantité

| Modèle | Quand l'essayer | Forces | Limites et vigilance |
| :--- | :--- | :--- | :--- |
| `DummyRegressor` | toujours en premier | référence minimale | ne modélise aucune relation |
| Régression linéaire | relation approximativement additive | rapide, extrapole, interprétable sous hypothèses | sensible aux valeurs extrêmes et non-linéarités |
| Ridge | nombreuses variables corrélées | stabilise les coefficients par L2 | scaling nécessaire pour comparer la pénalité |
| Lasso | représentation parcimonieuse souhaitée | certains coefficients deviennent nuls | sélection instable avec variables corrélées |
| Elastic Net | compromis L1/L2 | sélection et stabilité | deux forces de régularisation à régler |
| Régression polynomiale | courbure simple, faible dimension | reste linéaire en ses paramètres | explosion du nombre de variables et extrapolation dangereuse |
| k-NN Regressor | voisinage local pertinent | non paramétrique, intuitif | scaling, mémoire et coût à l'inférence |
| Arbre de régression | interactions et seuils | lisible s'il reste petit | instable, extrapole mal |
| Random Forest / Extra Trees | tabulaire non linéaire | robuste, peu de prétraitement | modèle lourd, sorties par moyennes d'arbres |
| Gradient Boosting / HistGradientBoosting | tabulaire, recherche de performance | interactions puissantes | hyperparamètres et surapprentissage |
| SVR | petit/moyen jeu, relation lisse | noyaux flexibles | scaling et coût quadratique/cubique possible |
| Processus gaussien | petit jeu, incertitude prédictive | prédiction probabiliste | passe mal à grande échelle, noyau à choisir |
| MLP | beaucoup de données ou structure complexe | grande flexibilité | réglage, normalisation et calcul |

Cas spécialisés : régression de **Poisson** pour des comptes, **Gamma** pour certaines valeurs positives asymétriques, **Tweedie** pour distributions mixtes, régression de **quantile** pour prédire un intervalle ou une asymétrie de coût. Vérifiez les hypothèses de domaine : une régression ordinaire peut prédire un compte négatif.

### 10.3 Classification

| Modèle | Bon candidat lorsque… | Forces | Limites et vigilance |
| :--- | :--- | :--- | :--- |
| `DummyClassifier` | toujours | baseline de fréquence/stratégie | aucune connaissance des variables |
| Régression logistique | frontière additive ou besoin de baseline | rapide, probabilités souvent exploitables | interactions à construire, calibration à vérifier |
| LDA / QDA | classes séparables par distributions simples | rapide sur petites données | hypothèses de covariance parfois irréalistes |
| Naive Bayes | texte sparse ou peu de données | extrêmement rapide | indépendance conditionnelle simplificatrice |
| k-NN | classes locales en faible dimension | simple, non linéaire | malédiction de la dimension, scaling |
| Arbre | règles et interactions | explicable s'il est petit | forte variance |
| Random Forest / Extra Trees | données tabulaires hétérogènes | solide avec peu de réglage | calibration parfois nécessaire, taille mémoire |
| Gradient Boosting | performance tabulaire | excellente capacité non linéaire | sensible au réglage et aux dérives |
| SVM linéaire | texte ou grande dimension sparse | efficace, marge maximale | probabilités non natives selon l'implémentation |
| SVM à noyau | jeu moyen, frontière complexe | très flexible | coûteux à grande échelle, scaling indispensable |
| MLP | grande quantité de données | représentations apprises | réglage et explicabilité plus difficiles |

Pour $K$ classes exclusives, on peut utiliser un modèle multinomial, **one-vs-rest** ou **one-vs-one**. Pour le multilabel, chaque observation peut porter plusieurs étiquettes : one-vs-rest, chaîne de classifieurs ou modèle multi-sorties. Les dépendances entre labels peuvent rendre les classifieurs indépendants insuffisants.

### 10.4 Ensembles

- **bagging** : entraîne des modèles sur des échantillons différents puis agrège ; réduit surtout la variance ;
- **Random Forest** : bagging d'arbres avec sous-échantillonnage aléatoire des variables ;
- **Extra Trees** : ajoute davantage de randomisation dans les seuils ;
- **boosting** : ajoute séquentiellement des modèles corrigeant les erreurs ;
- **gradient boosting** : optimise une loss par additions successives d'arbres ;
- **voting** : moyenne des probabilités ou vote de modèles différents ;
- **stacking** : un méta-modèle combine des prédictions produites hors pli pour éviter la fuite.

Des bibliothèques spécialisées proposent notamment XGBoost, LightGBM ou CatBoost. Elles peuvent être très performantes sur données tabulaires, mais leurs catégories, valeurs manquantes, contraintes et hyperparamètres ne se comportent pas exactement de la même façon. Le principe de validation reste inchangé.

### 10.5 Clustering : découvrir des groupes

| Modèle | Forme des groupes | Points forts | Limites |
| :--- | :--- | :--- | :--- |
| K-means / MiniBatch K-means | plutôt sphériques | rapide, scalable | impose $K$, sensible au scaling et aux outliers |
| Mélange gaussien (GMM) | elliptiques, probabilistes | appartenance souple | nombre de composantes et covariances à choisir |
| Hiérarchique agglomératif | hiérarchie de regroupements | dendrogramme, plusieurs distances | coûteux à grande échelle |
| DBSCAN | formes irrégulières + bruit | pas besoin de fixer $K$ | densités variables difficiles, scaling |
| HDBSCAN | densités variables | hiérarchie et bruit | dépendance externe, paramètres à interpréter |
| Spectral clustering | structure de graphe | frontières complexes | mémoire et coût élevés |
| BIRCH | très grands jeux | résumé incrémental | dépend du rayon de sous-cluster |

Un cluster n'est pas une vérité naturelle. Comparez plusieurs graines et échantillons, inspectez les profils, mesurez la stabilité et demandez si les groupes conduisent à une action utile. Le numéro d'un cluster n'a ni ordre ni sens intrinsèque.

### 10.6 Réduction de dimension et visualisation

| Méthode | Objectif | Cas d'usage | Piège fréquent |
| :--- | :--- | :--- | :--- |
| PCA | variance linéaire maximale | compression, débruitage | scaling et interprétation des composantes |
| Truncated SVD | approximation de matrice sparse | TF-IDF, grands espaces creux | centrage absent selon l'outil |
| NMF | facteurs non négatifs | thèmes, composantes additives | exige des entrées non négatives |
| ICA | sources statistiquement indépendantes | séparation de signaux | ordre/échelle des composantes arbitraires |
| t-SNE | voisinages locaux en 2D/3D | visualisation exploratoire | distances globales et amas trompeurs |
| UMAP | voisinages/manifold | visualisation et représentation | sensible aux paramètres et à la graine |
| Auto-encodeur | compression non linéaire | grandes données complexes | entraînement et validation nécessaires |

t-SNE et UMAP servent surtout à explorer ou visualiser. Une jolie carte 2D n'est pas une preuve que les groupes existent dans l'espace original. Si la transformation alimente un prédicteur, elle doit être ajustée dans les plis d'entraînement.

### 10.7 Détection d'anomalies

Trois cadres sont différents :

1. **supervisé** : anomalies étiquetées → classification déséquilibrée ;
2. **semi-supervisé/novelty detection** : entraînement surtout sur le normal ;
3. **non supervisé/outlier detection** : mélange sans labels fiables.

| Modèle | Intuition | Adapté à | Limite majeure |
| :--- | :--- | :--- | :--- |
| Isolation Forest | les anomalies s'isolent vite | tabulaire, grande taille | score relatif au jeu observé |
| Local Outlier Factor | densité locale anormalement faible | anomalies locales | inférence sur nouveaux points selon mode choisi |
| One-Class SVM | frontière autour du normal | jeu petit/moyen | scaling et réglage sensibles |
| Elliptic Envelope | distribution robuste elliptique | données proches d'une gaussienne | hypothèse restrictive |
| Auto-encodeur | forte erreur de reconstruction | image, signal, haute dimension | peut aussi reconstruire des anomalies |

Sans labels, le seuil reflète une hypothèse de contamination et une capacité de revue, pas une probabilité certaine de fraude ou de panne.

### 10.8 Séries temporelles

Une série ajoute ordre, tendance, saisonnalité, autocorrélation et parfois variables exogènes.

| Famille | Exemple | Quand l'utiliser |
| :--- | :--- | :--- |
| Baseline | dernière valeur, saison précédente | toujours avant un modèle complexe |
| Lissage exponentiel | ETS, Holt-Winters | tendance et saisonnalité régulières |
| ARIMA/SARIMA | dépendances linéaires et saisonnières | série univariée suffisamment stationnaire après transformation |
| Espace d'état / Kalman | état latent dynamique | capteurs, données bruitées, composantes interprétables |
| Régression sur retards | Ridge, forêt, boosting sur lags | variables exogènes et non-linéarités tabulaires |
| Modèle probabiliste | quantiles, distributions | décision sous incertitude |
| Réseau séquentiel | RNN, LSTM, GRU, TCN | nombreuses séries et motifs complexes |
| Transformer temporel | attention sur longues dépendances | grands jeux, covariables riches |

Le split est chronologique, les variables glissantes n'utilisent que le passé et la validation reproduit l'horizon réel. Évaluez plusieurs horizons et comparez à une baseline saisonnière avec MASE ou une métrique adaptée aux unités.

### 10.9 Recommandation et ranking

| Approche | Signal utilisé | Exemple |
| :--- | :--- | :--- |
| Popularité | fréquence globale ou segmentée | articles les plus consultés |
| Filtrage collaboratif | interactions utilisateur–item | k-NN, factorisation matricielle |
| Basé contenu | caractéristiques des items/utilisateurs | similarité de profils |
| Hybride | interactions + contenu | démarrage à froid amélioré |
| Factorisation implicite | clics, vues, achats | ALS avec confiance pondérée |
| Two-tower | encodeurs utilisateur et item | récupération à grande échelle |
| Learning-to-rank | préférences relatives/listes | arbres ou réseaux pairwise/listwise |

La recommandation comporte souvent deux étages : **candidate generation** rapide puis **ranking** plus fin. Évaluez hors ligne Recall@k, NDCG, diversité, couverture et nouveauté, puis validez en ligne sans sacrifier sécurité ni expérience. Les logs historiques reflètent ce que l'ancien système a choisi de montrer : c'est un biais d'exposition.

### 10.10 Données textuelles, visuelles, audio et graphes

- **Texte** : bag-of-words/TF-IDF + régression logistique ou SVM constituent de fortes baselines ; embeddings et Transformers deviennent utiles pour le contexte, le transfert ou la génération.
- **Image** : descripteurs manuels + SVM comme baseline ; CNN ou Vision Transformers pour classification, détection et segmentation.
- **Audio** : caractéristiques temps-fréquence + modèle classique ; CNN, RNN ou Transformers pour parole et événements sonores.
- **Graphes** : scores heuristiques, factorisation ou modèles de graphes pour nœuds, liens et graphe entier.

Ces domaines sont détaillés dans le module 3. Le protocole du module 2—split réaliste, baseline, métrique, incertitude et test intact—reste obligatoire.

### 10.11 Apprentissage semi-supervisé, actif et par renforcement

- **semi-supervisé** : combine peu d'exemples étiquetés et beaucoup d'exemples non étiquetés ; pseudo-labels et régularisation de cohérence exigent de contrôler les erreurs auto-renforcées ;
- **apprentissage actif** : sélectionne les observations à faire annoter ; utile quand l'annotation est chère, mais la stratégie de sélection peut biaiser l'échantillon ;
- **apprentissage en ligne** : met à jour progressivement le modèle ; utile pour les flux, avec surveillance renforcée des dérives ;
- **apprentissage par renforcement** : apprend une politique maximisant une récompense cumulative dans un environnement ; bandits, méthodes par valeur et méthodes de politique répondent à des cadres différents.

Une récompense mal spécifiée peut être optimisée d'une manière contraire à l'intention. Simulez, bornez les actions et évaluez hors politique avant tout usage réel sensible.

### 10.12 Guide de choix pratique

```text
Quelle est la cible ?
├─ nombre continu ───────────────► régression
├─ catégorie(s) ─────────────────► classification
├─ temps avant événement ────────► survie
├─ aucune cible
│  ├─ groupes ───────────────────► clustering
│  ├─ représentation compacte ───► réduction de dimension
│  └─ cas rares ─────────────────► détection d'anomalies
├─ valeur ordonnée dans le temps ► forecasting
└─ ordre d'items ────────────────► recommandation/ranking
```

Puis posez successivement :

1. **Combien de lignes et de variables ?** Les noyaux et voisins passent difficilement à très grande échelle.
2. **Données tabulaires ou non structurées ?** Le boosting est une référence tabulaire ; les réseaux dominent souvent image, audio et texte brut avec assez de données.
3. **Relation linéaire acceptable ?** Commencez par un modèle linéaire régularisé.
4. **Latence et mémoire contraintes ?** Mesurez le pipeline entier, pas seulement `predict`.
5. **Probabilités nécessaires ?** Vérifiez et, si besoin, corrigez la calibration.
6. **Explication ou contraintes réglementaires ?** Préférez une famille auditable ou une procédure d'explication validée.
7. **Valeurs manquantes, catégories, sparsité ?** Choisissez prétraitement et modèle conjointement.
8. **Le gain est-il stable ?** Comparez plusieurs splits/graines et intervalles.

Commencez par `Dummy`, un modèle linéaire et un modèle d'arbres. N'ajoutez un modèle complexe que si son gain hors échantillon justifie calcul, dette technique, latence et risque.

---

## 11. Validation Avancée et Incertitude

### 11.1 Validation croisée imbriquée

Lorsque le jeu est petit et que l'on veut estimer la performance après réglage :

1. la boucle **interne** choisit les hyperparamètres ;
2. la boucle **externe** estime la performance de toute la procédure de sélection.

Elle coûte davantage de calcul mais réduit l'optimisme lié à la sélection répétée sur les mêmes plis.

### 11.2 Intervalles et variabilité

Un score ponctuel ne décrit pas l'incertitude. Selon le plan d'échantillonnage, on peut utiliser bootstrap, répétitions de validation croisée ou intervalle analytique. Les observations groupées ou temporelles doivent être rééchantillonnées par groupe ou bloc, pas indépendamment.

Rapportez : taille du test, moyenne, dispersion ou intervalle, graines, protocole de split et variabilité entre sous-groupes. Un écart de 0,2 point n'est pas nécessairement significatif ni utile.

### 11.3 Reproductibilité

Fixer `random_state` facilite le débogage, sans garantir une reproductibilité absolue entre matériels et versions. Conservez versions des dépendances, code, configuration, empreinte des données, modèle sérialisé et environnement d'exécution.

---

## 12. Déséquilibre, Seuils et Coûts d'Erreur

### 12.1 Rééquilibrer correctement

Options courantes : poids de classes, sous-échantillonnage, sur-échantillonnage ou génération synthétique. Toute opération utilisant la cible doit être appliquée **uniquement dans les plis d'entraînement**. Ne rééquilibrez pas le test : il doit conserver la prévalence de l'usage réel.

### 12.2 Le seuil est une décision

Une probabilité devient une classe via un seuil $t$. Le seuil 0,5 n'est pas sacré. Choisissez-le sur validation selon :

- capacité de traitement des alertes ;
- coût attendu $C_{FP}FP + C_{FN}FN$ ;
- rappel minimal ;
- précision minimale ;
- exigences par sous-groupe.

Figez ensuite ce seuil avant l'évaluation finale. En production, surveillez-le avec la calibration et la prévalence.

---

## 13. Interprétabilité, Causalité et Robustesse

### 13.1 Expliquer à plusieurs niveaux

- **global** : comportement moyen, importance par permutation, courbes de dépendance ;
- **local** : variables ayant influencé une prédiction particulière ;
- **contre-factuel** : changements qui auraient modifié la décision, sous contraintes de plausibilité.

Les explications sont elles-mêmes des approximations. Une importance par permutation peut être trompeuse avec des variables corrélées ; une attribution locale ne prouve pas une cause. Testez la stabilité des explications et adaptez leur niveau à l'utilisateur.

### 13.2 Tests de robustesse

Évaluez le système sur : données incomplètes, catégories inconnues, bruit, décalage temporel, sous-groupes, entrées aux limites, changements d'unité et cas hors distribution. Définissez un comportement d'abstention ou de remontée humaine lorsque la confiance opérationnelle est insuffisante.

---

## 14. Cycle de Vie et Suivi en Production

Un modèle est un composant versionné, pas un fichier isolé. Le paquet déployé inclut prétraitement, ordre des classes, seuil, schéma d'entrée et métadonnées.

Suivez quatre niveaux :

1. **service** : disponibilité, latence, erreurs et coût ;
2. **données** : qualité, fraîcheur et dérive des entrées ;
3. **modèle** : calibration et performance lorsque les labels arrivent ;
4. **impact** : bénéfice métier, effets indésirables et équité.

Prévoyez déploiement progressif, comparaison à la version courante, journalisation, retour arrière et procédure d'incident. Ne réentraînez pas automatiquement sur toute donnée récente sans valider sa qualité et les conséquences du changement.

Une **fiche modèle** (*model card*) précise objectif, population, données, métriques, seuil, sous-groupes, limites, usages interdits, version et responsable.

---

## 15. Checklist et Questions de Compréhension

### Checklist d'une évaluation fiable

- [ ] Le problème, l'horizon, l'action et les coûts d'erreur sont définis.
- [ ] Une baseline triviale et la solution actuelle sont mesurées.
- [ ] Le split reflète le futur usage et le test reste intact.
- [ ] Le prétraitement et le rééquilibrage sont à l'intérieur de la validation croisée.
- [ ] La métrique, la classe positive, la moyenne et le seuil sont explicités.
- [ ] L'incertitude et les performances par sous-groupe sont rapportées.
- [ ] Les explications ne sont pas présentées comme des preuves causales.
- [ ] Le modèle, les données et l'environnement sont versionnés.
- [ ] Les critères de déploiement, suivi, alerte et retour arrière sont écrits.

### Questions de compréhension

1. Pourquoi choisir les hyperparamètres sur le test rend-il son score optimiste ?
2. Dans quel cas préférer MAE à RMSE ?
3. Comment un modèle peut-il avoir une bonne AUC mais être mal calibré ?
4. Pourquoi le sur-échantillonnage doit-il rester dans chaque pli d'entraînement ?
5. Quelle différence entre une explication locale et une relation causale ?

**Mini-étude de cas.** Pour une détection de fraude à 0,2 % de prévalence, définissez baseline, métriques, split, stratégie de seuil, protocole d'incertitude et comportement lorsque le système hésite.
