# Module 1 : Nature, Mise en Forme et Préparation des Données pour l'IA

> **"Garbage In, Garbage Out"** : la qualité, la représentativité et la traçabilité des données limitent directement ce que l'on peut conclure d'un modèle. La part de travail consacrée aux données varie fortement selon le projet ; l'enjeu est de rendre chaque décision de préparation explicite et vérifiable.

**Objectifs du module.** À l'issue de ce chapitre, vous saurez identifier l'unité d'observation et la cible, auditer un jeu de données, choisir une stratégie de découpage réaliste, construire un prétraitement reproductible sans fuite d'information et définir des contrôles de qualité avant la mise en production.

**Prérequis.** Python élémentaire, tableaux, moyenne, médiane et notion intuitive de probabilité. Le notebook associé transforme chaque principe en code avec Pandas et Scikit-Learn.

---

## 📖 Le Dico du Débutant (Jargon Buster)
Avant de plonger dans la technique, voici les 6 mots indispensables traduits en français simple :
- **Observation / Échantillon (*Sample* ou *Row*)** : Une ligne dans votre tableau. C'est l'entité unique que l'on étudie (un patient à l'hôpital, une maison à vendre, un client bancaire).
- **Caractéristique (*Feature* ou *Variable*)** : Une colonne dans votre tableau. C'est une information mesurée sur l'échantillon (l'âge du patient, la surface de la maison, le salaire du client).
- **Cible (*Target* ou *Label* ou *Ground Truth*)** : La colonne spéciale que le modèle doit apprendre à deviner (ex: le prix de vente de la maison, ou si le patient est malade : 1 pour oui, 0 pour non).
- **Données brutes (*Raw data*)** : Le fichier tel qu'il arrive de la vraie vie : avec des fautes de frappe, des dates écrites dans tous les sens, des cases vides et du texte non formaté.
- **Prétraitement (*Preprocessing*)** : La "cuisine" indispensable pour transformer ces données brutes en une matrice de nombres propres que l'ordinateur peut manipuler.
- **Fuite de données (*Data Leakage*)** : Le piège n°1 du débutant : laisser le modèle regarder sans faire exprès les réponses de l'examen final pendant ses révisions.

---

## Table des Matières
1. [La Nature et la Typologie des Données](#1-la-nature-et-la-typologie-des-données)
2. [Formes et Représentations Mathématiques : Démystifier X et y](#2-formes-et-représentations-mathématiques--démystifier-x-et-y)
3. [Les Pathologies des Données du Monde Réel](#3-les-pathologies-des-données-du-monde-réel)
4. [Le Pipeline de Préparation des Données (Data Preprocessing)](#4-le-pipeline-de-préparation-des-données-data-preprocessing)
5. [Le Piège Mortel du Machine Learning : Le Data Leakage](#5-le-piège-mortel-du-machine-learning--le-data-leakage)
6. [Synthèse et Bonnes Pratiques en Production](#6-synthèse-et-bonnes-pratiques-en-production)
7. [Cadrage, Collecte et Qualité des Étiquettes](#7-cadrage-collecte-et-qualité-des-étiquettes)
8. [Découpage Train, Validation et Test](#8-découpage-train-validation-et-test)
9. [Feature Engineering et Sélection de Variables](#9-feature-engineering-et-sélection-de-variables)
10. [Contrats de Données et Tests de Qualité](#10-contrats-de-données-et-tests-de-qualité)
11. [Biais, Vie Privée, Sécurité et Gouvernance](#11-biais-vie-privée-sécurité-et-gouvernance)
12. [Données en Production et Dérive](#12-données-en-production-et-dérive)
13. [Prendre en Main Pandas et Polars](#13-prendre-en-main-pandas-et-polars)
14. [Exploration Descriptive et Visualisation](#14-exploration-descriptive-et-visualisation)
15. [Corrélations et Associations avec la Cible](#15-corrélations-et-associations-avec-la-cible)
16. [Sélection Cohérente des Variables](#16-sélection-cohérente-des-variables)
17. [Démonstration de Bout en Bout](#17-démonstration-de-bout-en-bout)
18. [Checklist et Questions de Compréhension](#18-checklist-et-questions-de-compréhension)

---

## 1. La Nature et la Typologie des Données

### 💡 Pourquoi les ordinateurs ont-ils besoin de nombres ?
Imaginez que vous deviez expliquer à un extraterrestre qui ne comprend **strictement rien d'autre que les additions et les multiplications** ce qu'est une voiture de sport rouge.  
Vous ne pouvez pas lui donner le mot "Rouge", ni lui montrer une photo directement. Vous devez traduire la couleur rouge en intensités de lumière numériques (ex: `[255, 0, 0]`), la puissance en chevaux (ex: `450`), et le prix en euros (ex: `85000`).

Un algorithme de Machine Learning est exactement comme cet extraterrestre : **c'est une gigantesque calculatrice matricielle**. Tout ce qui n'est pas un nombre réel doit être converti en nombre avant qu'un modèle ne puisse faire quoi que ce soit.

```
                                  DONNÉES
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
      Structurées                                       Non-Structurées
 (Tables SQL, CSV, DataFrames)                   (Images, Textes, Audios, Graphes)
           │
     ┌─────┴─────┐
     ▼           ▼
Quantitatives  Qualitatives
 (Numériques)  (Catégorielles)
     │               │
  ┌──┴──┐         ┌──┴──┐
  ▼     ▼         ▼     ▼
Continu Discret Nominal Ordinal
```

### 1.1 Données Quantitatives (Numériques)
Ce sont des mesures numériques directes sur lesquelles des opérations arithmétiques ont un sens mathématique immédiat (on peut calculer une moyenne, faire une somme) :
- **Continues** : Peuvent prendre une infinité de valeurs décimales dans un intervalle réel $\mathbb{R}$.
  - *Exemples du quotidien* : Votre taille ($1.78\text{ m}$), la température extérieure ($19.4^\circ\text{C}$), le prix d'un billet de train ($42.50\text{ €}$).
- **Discrètes** : Valeurs dénombrables, presque toujours des nombres entiers naturels $\mathbb{N}$ (on ne peut pas avoir un demi-enfant ou un tiers de clic).
  - *Exemples du quotidien* : Nombre de pièces dans un appartement ($3$), nombre d'enfants ($2$), nombre de tentatives de connexion échouées ($4$).

### 1.2 Données Qualitatives (Catégorielles)
Elles représentent des états, des types, des textes ou des attributs descriptifs :
- **Nominales** : Des catégories sans hiérarchie ni ordre naturel. Aucune n'est "supérieure" à une autre.
  - *Exemples du quotidien* : Couleur d'une voiture (`Rouge`, `Bleue`, `Verte`), ville de résidence (`Paris`, `Lyon`, `Marseille`), statut marital (`Célibataire`, `Marié`, `Divorcé`).
  - *Attention* : Il n'y a aucun sens à dire que `Marseille` est plus grand que `Paris` ou que la moyenne de `Bleu` et `Rouge` donne `Vert`.
- **Ordinales** : Des catégories qui possèdent un ordre logique évident, mais l'écart entre les niveaux n'est pas un nombre mesurable avec précision.
  - *Exemples du quotidien* : Niveau d'études (`Bac`, `Licence`, `Master`, `Doctorat`), avis client (`Très insatisfait`, `Neutre`, `Très satisfait`), taille d'un vêtement (`S`, `M`, `L`, `XL`).
  - *Note* : On sait que `Master` est supérieur à `Licence`, mais on ne peut pas affirmer mathématiquement que `Master - Licence = Licence - Bac`.

### 1.3 Données Non-Structurées et Multimodalité
- **Images** : Une image n'est rien d'autre qu'une grille de pixels (une matrice à 3 dimensions : Hauteur, Largeur, Canaux de couleur Rouge/Vert/Bleu). Chaque case contient un entier de 0 (noir complet) à 255 (couleur maximale).
- **Texte naturel (NLP)** : Les phrases sont découpées en unités appelées jetons (*tokens*), qui peuvent être des mots, des sous-mots ou des caractères. Ces jetons sont associés à des vecteurs (*embeddings*) dont la représentation est ensuite contextualisée par le modèle.
- **Séries temporelles** : Des mesures répétées à intervalles réguliers (température chaque heure, cours d'une action à la bourse chaque seconde). L'ordre des lignes a une importance capitale !
- **Graphes** : Des réseaux de points reliés par des flèches (réseau social d'amis, réseau routier, molécules chimiques).

### 1.4 Échelles de mesure : ce que les opérations signifient

Le type informatique ne suffit pas. Une colonne entière peut représenter un compte, une catégorie ou un identifiant : les opérations licites ne sont pas les mêmes.

| Échelle | Ce qui est préservé | Opérations pertinentes | Exemple |
| :--- | :--- | :--- | :--- |
| Nominale | égalité/différence | fréquences, mode | pays, type de panne |
| Ordinale | ordre | médiane, quantiles, rangs | satisfaction 1–5 |
| Intervalle | ordre et écarts | moyenne, écart-type | température en °C |
| Ratio | intervalle et zéro absolu | rapports, moyenne géométrique selon le cas | masse, durée, revenu |
| Cyclique | voisinage circulaire | encodage sinus/cosinus | heure, jour de semaine |

Dire que 20 °C vaut « deux fois » 10 °C n'a pas de sens, contrairement à 20 kg contre 10 kg. Une note Likert codée `1, 2, 3, 4, 5` reste ordinale même si son `dtype` est entier.

### 1.5 Type logique, type physique et rôle métier

Trois descriptions doivent être séparées :

- **type physique** : `int64`, `float32`, chaîne UTF-8, booléen, date, binaire ;
- **type logique** : montant, durée, catégorie, géométrie, liste, document ;
- **rôle analytique** : identifiant, caractéristique, cible, poids, groupe, horodatage, variable sensible ou métadonnée.

Exemples de pièges :

- un code postal lu comme entier perd ses zéros initiaux et ne doit pas être moyenné ;
- un identifiant numérique n'est pas une quantité ;
- une date stockée comme texte ne se trie pas toujours chronologiquement ;
- une somme monétaire en `float` peut accumuler des erreurs d'arrondi ; un type décimal peut être requis ;
- `NaN`, chaîne vide, `NULL`, `-999` et « inconnu » ne représentent pas nécessairement le même manque.

### 1.6 Panorama des données industrielles complexes

| Famille | Structure | Exemple industriel | Difficulté principale |
| :--- | :--- | :--- | :--- |
| Table relationnelle | lignes/colonnes liées par clés | clients, commandes, produits | jointures, granularité, doublons |
| Transaction | une ligne par opération | paiement, achat, mouvement de stock | volumes, fraude, ordre temporel |
| Journal d'événements | événement + horodatage + entité | clics, logs applicatifs, audit | ordre, sessions, événements tardifs |
| Série temporelle | valeur indexée par le temps | capteur, énergie, finance | tendance, saisonnalité, fréquence irrégulière |
| Données de panel | mêmes entités répétées dans le temps | patients, magasins, machines | dépendance intra-entité, split par groupe/temps |
| Flux (*stream*) | données continues et potentiellement infinies | télémétrie IoT, événements web | latence, ordre, fenêtre, reprise |
| Semi-structuré | schéma flexible et imbriqué | JSON d'API, XML, logs | champs optionnels, listes, évolution du schéma |
| Texte | séquences de symboles | contrats, tickets, emails | langue, contexte, données sensibles |
| Image | grille ou volume de pixels | radiologie, satellite, contrôle qualité | résolution, annotation, augmentation |
| Audio | onde et représentation temps-fréquence | appel, moteur, musique | échantillonnage, bruit, durée variable |
| Vidéo | images + temps + parfois audio | surveillance, sport, robotique | volume, synchronisation, temporalité |
| Géospatial | points, lignes, polygones, raster | mobilité, réseaux, agriculture | projection, distance, voisinage spatial |
| Graphe | nœuds, arêtes et attributs | fraude, molécules, supply chain | dépendances, sous-graphes, fuite entre voisins |
| Sparse haute dimension | majorité de zéros | mots, clics, produits | mémoire, métriques de distance |
| Embeddings | vecteurs denses appris | recherche sémantique, recommandation | version du modèle, dérive, proximité trompeuse |
| Multimodal | modalités alignées | image–légende, vidéo–audio, document scanné | alignement, modalités manquantes |
| Survie/censure | durée partiellement observée | panne, churn, guérison | événement non encore observé |
| Intervalles/incertitude | valeur non ponctuelle | mesure capteur ± erreur | propagation de l'incertitude |

Une ligne peut contenir plusieurs structures : un sinistre possède des champs tabulaires, des photographies, un rapport textuel, des coordonnées et une chronologie d'événements.

### 1.7 Tables longues, larges et imbriquées

- **format large** : une ligne par entité et une colonne par mesure ou période ; pratique pour certains modèles, moins flexible ;
- **format long** : une ligne par entité–variable–temps ; pratique pour agréger et visualiser ;
- **structures imbriquées** : listes et objets dans une cellule, fréquents en JSON ou Arrow ; elles doivent parfois être éclatées, agrégées ou encodées ;
- **table de faits et dimensions** : le fait porte l'événement, les dimensions décrivent produit, client, lieu ou temps.

Toujours écrire la **granularité** : « une ligne par client » diffère de « une ligne par achat ». Une jointure entre granularités peut multiplier les lignes et modifier artificiellement les statistiques.

### 1.8 Stockage et accès : CSV n'est pas l'unique format

| Format/système | Bon usage | Vigilance |
| :--- | :--- | :--- |
| CSV | échange simple, petit jeu | types non conservés, séparateur, encodage |
| JSON/JSONL | événements et objets flexibles | schéma évolutif, champs imbriqués |
| Parquet/Arrow | analytique colonnaire, types riches | partitions, compatibilité des schémas |
| SQL/warehouse | données relationnelles gouvernées | requête, clés, coût et fraîcheur |
| Lake/lakehouse | fichiers volumineux et historisés | catalogue, qualité, petits fichiers |
| Base graphe | parcours de relations | modèle de graphe et coût des traversées |
| Index vectoriel | voisins d'embeddings | filtrage, version, droits d'accès |

Un format **ligne** favorise souvent l'écriture ou la lecture d'un enregistrement complet ; un format **colonnaire** favorise l'analyse de quelques colonnes sur beaucoup de lignes. Le choix modifie temps de lecture, mémoire et conservation des types, pas la signification métier.

---

## 2. Formes et Représentations Mathématiques : Démystifier X et y

Ne laissez pas les notations mathématiques vous effrayer ! Derrière les symboles se cache simplement votre tableau Excel habituel :

$$\mathbf{X} \in \mathbb{R}^{n \times d}, \quad \mathbf{y} \in \mathbb{R}^n$$

- **$n$** : Le nombre de lignes (**observations** ou exemples, ex: 1 000 clients).
- **$d$** : Le nombre de colonnes de caractéristiques (**features**, ex: âge, salaire, ancienneté).
- **$\mathbf{X}$** : La grande matrice contenant toutes les informations descriptives des clients.
- **$\mathbf{y}$** : Le vecteur colonne contenant la réponse à deviner pour chaque client (**la cible**).

### 🔍 Exemple Concret Pas à Pas :
Imaginons que nous voulons prédire si un client va souscrire à une assurance premium ($\mathbf{y} \in \{0, 1\}$) à partir de son Âge et de son Salaire annuel :

**Le tableau métier d'origine (comme dans Excel) :**
| Client | Âge ($x_1$) | Salaire en k€ ($x_2$) | Souscription Assurance ($y$) |
| :--- | :---: | :---: | :---: |
| Alice | 25 | 32 | Non (0) |
| Bob | 45 | 65 | Oui (1) |
| Charlie | 38 | 48 | Oui (1) |

**Ce que voit l'algorithme :**
$$\mathbf{X} = \begin{bmatrix} 25 & 32 \\ 45 & 65 \\ 38 & 48 \end{bmatrix} \quad (n=3 \text{ lignes}, d=2 \text{ colonnes}), \quad \mathbf{y} = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix} \quad (n=3 \text{ cibles})$$

Chaque ligne $i$ est un vecteur d'attributs $\mathbf{x}_i$ décrivant un individu unique dans un espace géométrique à $d$ dimensions :
$$\mathbf{x}_{\text{Alice}} = \begin{bmatrix} 25 \\ 32 \end{bmatrix}, \quad \mathbf{x}_{\text{Bob}} = \begin{bmatrix} 45 \\ 65 \end{bmatrix}$$

---

## 3. Les Pathologies des Données du Monde Réel

Dans les cours théoriques purs, les données sont impeccables. Dans la vraie vie, elles sont "sales", incomplètes et piégées :

### 3.1 Les Données Manquantes : 3 Mécanismes à Comprendre
Pourquoi une case est-elle vide ? La réponse change totalement la façon dont vous devez réagir :

1. **MCAR (Missing Completely At Random - Manquant Totalement au Hasard)** :
   - *L'analogie du quotidien* : Le facteur a renversé son café sur quelques fiches d'un sondage papier. Les taches effacent des cases au hasard, sans aucun lien avec l'âge, le sexe ou les revenus des personnes.
   - *Traitement possible* : Une suppression ou une imputation simple peut être raisonnable, mais elle réduit ou modifie l'incertitude. Le choix dépend du taux de valeurs manquantes et de l'analyse visée.
2. **MAR (Missing At Random - Manquant au Hasard Conditionnel)** :
   - *L'analogie du quotidien* : Dans une enquête de santé, les hommes de plus de 50 ans oublient deux fois plus souvent de renseigner leur niveau d'activité sportive que les jeunes femmes. Cependant, au sein du groupe "hommes de plus de 50 ans", l'oubli est aléatoire.
   - *Traitement possible* : Une imputation conditionnelle (k-NN, modèles itératifs) peut exploiter les variables observées. Elle doit être ajustée sur les données d'entraînement et validée ; elle ne reconstitue pas une vérité certaine.
3. **MNAR (Missing Not At Random - Manquant Non Aléatoire)** :
   - *L'analogie du quotidien* : Les contribuables ayant un patrimoine supérieur à 2 millions d'euros refusent volontairement de répondre à la question "Quel est votre patrimoine net ?". La valeur est manquante **précisément parce qu'elle est très élevée** !
   - *Risque* : Une imputation médiane peut sous-estimer systématiquement les grandes valeurs. Il faut expliciter le mécanisme supposé, envisager un indicateur de valeur manquante et conduire une analyse de sensibilité.

### 3.2 Les Valeurs Aberrantes (Outliers)
Une valeur aberrante est une mesure extrême qui s'éloigne drastiquement du reste du groupe :
- *L'erreur de saisie* : Un bébé pesé à la maternité noté à $350\text{ kg}$ au lieu de $3.50\text{ kg}$.
- *L'événement réel mais exceptionnel* : Elon Musk entre dans un café de quartier. Le revenu moyen des clients du café passe instantanément de $3\,000\text{ €}$ à $10\,000\,000\text{ €}$ par mois ! La moyenne est totalement détruite par un seul point, alors que la médiane ne bouge pratiquement pas.
- *Impact en IA* : Les modèles minimisant les erreurs au carré (comme la régression linéaire ou les réseaux de neurones) vont faire des contorsions désespérées pour essayer de satisfaire cette valeur extrême, ruinant la prédiction pour tous les autres clients normaux.

### 3.3 Le Déséquilibre de Classes (Class Imbalance)
Dans les problèmes critiques (détection de fraude bancaire, diagnostic d'une tumeur rare), les cas positifs sont rarissimes : par exemple, $999$ transactions légitimes pour $1$ transaction frauduleuse ($0.1\%$).
- **Le piège du débutant ("Le piège de l'Accuracy")** :  
  Si un modèle bête prédit constamment : *"Aucune transaction n'est jamais une fraude"*, quel est son taux de réussite ?  
  Il a raison 999 fois sur 1 000, soit **$99.9\%$ d'accuracy** ! Pourtant, ce modèle est totalement inutile car il laisse passer 100% des criminels.  
  *Morale* : L'accuracy doit être comparée à une baseline et complétée par des métriques adaptées au coût des erreurs : **rappel**, **précision**, **$F_1$**, courbe précision-rappel, calibration et matrice de confusion.

---

## 4. Le Pipeline de Préparation des Données (Data Preprocessing)

Le pipeline est la chaîne de montage industrielle qui nettoie et formate la donnée avant de la donner au modèle :

```
Données Brutes (texte, dates, NaN, outliers)
     │
     ▼
[1. Nettoyage] ──────► Éliminer les doublons, corriger les types aberrants
     │
     ▼
[2. Imputation] ─────► Boucher les trous (valeurs manquantes) avec méthode
     │
     ▼
[3. Encodage] ───────► Traduire les mots en chiffres (One-Hot ou Ordinal)
     │
     ▼
[4. Scaling] ────────► Mettre toutes les échelles numériques au même niveau
     │
     ▼
[5. Feature Eng.] ───► Créer de nouvelles colonnes intelligentes (ratios, durées)
     │
     ▼
Matrice X propre prête pour l'algorithme d'IA
```

### 4.1 Stratégies d'Imputation (Boucher les cases vides)
| Méthode | En quoi ça consiste ? | Exemple d'utilisation | Quand l'éviter ? |
| :--- | :--- | :--- | :--- |
| **Suppression (Drop)** | Jeter purement et simplement la ligne | Si moins de 1% des lignes ont un trou aléatoire | Si vous avez peu de données ou beaucoup de colonnes trouées |
| **Médiane** | Remplacer par la valeur du milieu | Âge, Salaire, Prix (très robuste aux valeurs extrêmes) | Si les données sont qualitatives (du texte) |
| **Mode (Fréquence)** | Remplacer par la réponse la plus fréquente | Ville, Statut marital, Niveau d'études | Si aucune catégorie ne domine clairement |
| **k-NN (Voisins)** | Chercher les $k$ personnes les plus proches et faire la moyenne | Imputation précise basée sur le profil global | Très lent sur les gros datasets ($> 100\,000$ lignes) |

### 4.2 L'Encodage des Variables Catégorielles (Mots $\to$ Nombres)
Les algorithmes ne savent pas calculer la racine carrée de `"Paris"`. Comment convertir du texte en nombres ?

#### A. One-Hot Encoding (OHE) - Pour les données Nominales (sans ordre)
On crée une nouvelle colonne binaire ($0$ ou $1$) pour chaque modalité possible :

**Avant encodage :**
| Client | Ville |
| :--- | :--- |
| Alice | Paris |
| Bob | Lyon |
| Charlie | Marseille |

**Après One-Hot Encoding :**
| Client | Ville_Paris | Ville_Lyon | Ville_Marseille |
| :--- | :---: | :---: | :---: |
| Alice | **1** | 0 | 0 |
| Bob | 0 | **1** | 0 |
| Charlie | 0 | 0 | **1** |

> ⚠️ **L'Erreur Fatale du Débutant** : Utiliser un encodage numérique naïf (`Paris=1`, `Lyon=2`, `Marseille=3`).  
> Si vous faites cela, l'algorithme va calculer : $\text{Marseille} (3) = 3 \times \text{Paris} (1)$ et $(\text{Paris} + \text{Marseille})/2 = \text{Lyon} (2)$. C'est une absurdité totale qui ruine le modèle !

#### B. Ordinal Encoding - Uniquement pour les données Ordinales (avec ordre logique)
Ici, les nombres $0, 1, 2, 3$ ont un vrai sens d'échelle :
- `Secondaire` $\to 0$
- `Licence` $\to 1$
- `Master` $\to 2$
- `Doctorat` $\to 3$

Cet encodage conserve l'ordre, mais peut aussi faire supposer au modèle que les écarts entre niveaux sont réguliers. Il faut tester cette hypothèse ; selon le modèle et le cas d'usage, un autre encodage peut mieux convenir.

### 4.3 Mise à l'Échelle (Feature Scaling) : Quand est-elle utile ?
Considérons deux candidats à un crédit :
- Alice : Âge = $25\text{ ans}$, Salaire = $30\,000\text{ €}$
- Bob : Âge = $27\text{ ans}$, Salaire = $35\,000\text{ €}$

Si un algorithme calcule la distance géométrique entre Alice et Bob :
$$\text{Distance}^2 = (25 - 27)^2 + (30\,000 - 35\,000)^2 = (-2)^2 + (-5\,000)^2 = 4 + 25\,000\,000$$

Dans un modèle fondé sur les distances ou les gradients, l'échelle du salaire domine alors le calcul. Les arbres de décision sont beaucoup moins sensibles à ce problème : la mise à l'échelle n'est donc pas systématique.

**Les trois solutions de mise à l'échelle :**
1. **Standardisation (Z-Score)** :  
   $$z = \frac{x - \mu}{\sigma} \quad (\mu = \text{moyenne}, \sigma = \text{écart-type})$$  
   Ramène la moyenne à $0$ et l'écart-type à $1$. Les valeurs typiques se situent entre $-3$ et $+3$. C'est le standard pour la régression, les SVM et le Deep Learning.
2. **Normalisation Min-Max** :  
   $$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}} \in [0, 1]$$  
   Écrase toutes les valeurs entre $0$ et $1$. Idéal pour les pixels d'images ($[0, 255] \to [0, 1]$). Sensible aux valeurs extrêmes.
3. **Mise à l'échelle Robuste (RobustScaler)** :  
   Utilise la médiane et l'écart interquartile ($\text{IQR} = Q_3 - Q_1$). C'est une option robuste aux valeurs extrêmes, à comparer avec une transformation logarithmique, une winsorisation justifiée ou un modèle peu sensible à l'échelle.

### 💡 Le Vocabulaire Scikit-Learn : fit, transform et fit_transform
Les débutants confondent souvent ces trois méthodes :
- **`fit(X)` ("Apprendre")** : Calcule et mémorise les paramètres statistiques (ex: calcule la moyenne $\mu$ et l'écart-type $\sigma$ d'une colonne). Ne modifie rien aux données.
- **`transform(X)` ("Appliquer")** : Utilise les paramètres mémorisés pour transformer les données (soustrait $\mu$ et divise par $\sigma$).
- **`fit_transform(X)` ("Apprendre et Appliquer")** : Combine les deux en une seule commande. **À n'utiliser QUE sur le jeu d'entraînement (`Train`) !**

---

## 5. Le Piège Mortel du Machine Learning : Le Data Leakage

Le **Data Leakage** (fuite de données) est l'équivalent de donner les questions de l'examen final aux étudiants avant qu'ils ne passent l'épreuve.

### L'Analogie du Contrôle de Mathématiques
Imaginez un élève qui s'entraîne. Le professeur lui donne par erreur le sujet de l'examen de fin d'année avec les réponses. L'élève révise sur ce sujet et obtient un magnifique **$20/20$**. Le professeur est ravi et pense que son élève est un prodige. Mais le jour où l'élève est embauché dans une entreprise et doit résoudre un problème inédit, il est incapable de faire le moindre calcul et échoue lamentablement.

C'est exactement ce qui arrive à votre modèle si vous faites du Data Leakage :
- Vos scores de test seront magnifiques (ex: $99\%$ de précision).
- En conditions réelles de production, le modèle sera catastrophique.

### L'Erreur Typique du Débutant en Code :
```python
# ❌ ERREUR GRAVE : Data Leakage !
scaler = StandardScaler()
# On calcule la moyenne et l'écart-type sur TOUT le dataset (Train + Test mélangés) !
X_scaled = scaler.fit_transform(X) 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
```

### Le Code Correct et Professionnel :
```python
# ✅ CODE CORRECT : Étanchéité absolue
# 1. On sépare d'ABORD le Train du Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Le scaler apprend les statistiques UNIQUEMENT sur le Train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # fit + transform sur Train

# 3. Sur le Test, on ne fait QUE transformer avec les paramètres du Train !
X_test_scaled = scaler.transform(X_test)       # AUCUN fit sur Test !
```

---

## 6. Synthèse et Bonnes Pratiques en Production

1. **La règle d'or** : Séparez votre jeu de données en `Train` et `Test` avant d'ajuster toute transformation dépendant des données (imputation, mise à l'échelle, sélection de variables). Les contrôles de schéma et de qualité restent possibles avant le découpage s'ils ne guident pas le modèle à partir du test.
2. **Choisissez le bon encodage** : `OneHotEncoder` pour les catégories sans hiérarchie (villes, couleurs), `OrdinalEncoder` pour les catégories ordonnées (niveaux d'études, tailles).
3. **Méfiez-vous de la moyenne** si vos données contiennent des valeurs extrêmes : préférez la médiane et le `RobustScaler`.
4. **Surveillez le déséquilibre de classes** : Ne vous fiez jamais à l'accuracy seule quand une classe est rare.
5. **Automatisez tout avec les `Pipeline` Scikit-Learn** : C'est le meilleur moyen d'empêcher les fuites de données et de déployer un code propre en entreprise.

Pour approfondir et vérifier ces recommandations, consultez les sources du module dans [REFERENCES.md](../REFERENCES.md).

---

## 7. Cadrage, Collecte et Qualité des Étiquettes

La préparation commence **avant** le premier `DataFrame`. Une table parfaitement nettoyée peut rester inutilisable si elle décrit la mauvaise population ou si la cible ne correspond pas à la décision métier.

### 7.1 Définir l'unité d'observation

Une ligne doit représenter une unité sans ambiguïté : un client, une transaction, une machine à une date donnée, ou un séjour hospitalier. Mélanger plusieurs niveaux crée des erreurs silencieuses.

Exemple : si une table contient une ligne par transaction mais une autre une ligne par client, une jointure naïve peut dupliquer les clients les plus actifs. Avant toute jointure, documentez :

- la clé primaire et son unicité attendue ;
- la granularité temporelle ;
- la population incluse et les exclusions ;
- la date à laquelle chaque variable devient réellement disponible.

### 7.2 Transformer un besoin en cible mesurable

« Prédire les bons clients » n'est pas une cible. Il faut préciser :

- **l'événement** : résiliation, achat, panne, défaut de paiement ;
- **l'horizon** : dans 7 jours, 3 mois ou 1 an ;
- **la date d'observation** : instant auquel la prédiction doit être calculée ;
- **le délai de maturation** : durée nécessaire pour connaître la vérité finale ;
- **l'action associée** : alerte, priorisation, contrôle humain ou décision automatisée.

Une cible construite avec une information postérieure à la date de prédiction provoque une **fuite temporelle**. Par exemple, utiliser le motif final de clôture d'un dossier pour prédire son issue au moment de son ouverture donne un score artificiellement élevé.

### 7.3 Provenance et représentativité

Un échantillon n'est pas « la réalité » : c'est le résultat d'un mécanisme de collecte. Posez systématiquement ces questions :

1. Qui peut entrer dans le jeu de données ?
2. Qui en est absent et pourquoi ?
3. Une action historique a-t-elle influencé les observations disponibles ?
4. La période contient-elle une crise, une promotion ou une panne exceptionnelle ?
5. Les conditions futures ressembleront-elles aux conditions de collecte ?

Le **biais de sélection** apparaît lorsque la population observée diffère de la population d'usage. Le **biais de survivant** exclut les cas disparus ou échoués. Une **boucle de rétroaction** apparaît quand les décisions du modèle modifient les futures données d'entraînement.

### 7.4 Étiquettes : bruit, ambiguïté et consensus

La vérité terrain peut être imparfaite. Un diagnostic, un avis de modération ou une notion de « qualité » peut varier entre annotateurs. Pour chaque tâche d'annotation :

- rédigez un guide avec définitions, exemples et cas limites ;
- mesurez l'accord entre annotateurs sur un sous-échantillon ;
- faites arbitrer les désaccords importants ;
- conservez l'identité de la version du guide et la provenance de l'étiquette ;
- quantifiez les classes ambiguës au lieu de les masquer.

Une étiquette approximative borne la performance mesurable. Accumuler davantage de données avec le même protocole défectueux ne corrige pas cette limite.

---

## 8. Découpage Train, Validation et Test

Les trois jeux n'ont pas le même rôle :

| Jeu | Rôle | Peut influencer le choix du modèle ? |
| :--- | :--- | :---: |
| **Entraînement** | Ajuster les paramètres | Oui |
| **Validation** | Choisir transformations, modèle, seuil et hyperparamètres | Oui |
| **Test** | Estimer une seule fois la performance finale | Non |

Consulter plusieurs fois le test pour corriger le système revient à l'utiliser comme validation : il n'est alors plus indépendant.

### 8.1 Choisir le découpage qui reproduit l'usage

- **Données indépendantes et équilibrées** : découpage aléatoire, souvent stratifié en classification.
- **Plusieurs lignes par personne, machine ou magasin** : découpage par groupe pour empêcher la présence d'une même entité dans le train et le test.
- **Séries temporelles** : entraînement sur le passé, validation puis test sur des périodes futures ; jamais de mélange aléatoire qui ferait voyager l'information vers le passé.
- **Données géographiques** : envisager une séparation par zone lorsque la généralisation à de nouveaux territoires est l'objectif.

```text
Temps ─────────────────────────────────────────────────────►
       [ entraînement ] [ validation ] [ test futur ]
                         aucun apprentissage sur le futur
```

### 8.2 Validation croisée sans fuite

Chaque transformation apprise—imputation, normalisation, vocabulaire, sélection de variables, sur-échantillonnage—doit être réajustée **dans chaque pli d'entraînement**. Une `Pipeline` automatise cette frontière. Pour les données temporelles ou groupées, utilisez des plis respectant ces contraintes plutôt qu'un K-fold standard.

### 8.3 Déduplication et quasi-doublons

Deux fichiers différents peuvent contenir la même observation ou des variantes presque identiques. En vision, des images issues de la même vidéo ; en texte, des copies d'un article ; en santé, plusieurs mesures d'un même patient. Dédupliquer avant le découpage, ou regrouper les variantes, évite une mémorisation déguisée.

---

## 9. Feature Engineering et Sélection de Variables

Le **feature engineering** transforme les mesures brutes en informations plus proches du mécanisme étudié. Quelques familles courantes :

- dates → heure, jour de semaine, ancienneté, saison, temps depuis le dernier événement ;
- montants → logarithme, ratio par utilisateur, variation plutôt que niveau absolu ;
- séries → retards (*lags*), moyennes glissantes, tendance et saisonnalité ;
- texte → longueur, indicateurs métier, représentation TF-IDF ou embeddings ;
- géographie → distance, zone, densité, avec vigilance sur la vie privée.

Une caractéristique doit être calculable **au moment réel de la prédiction**. Une moyenne glissante centrée, qui utilise des jours futurs, est une fuite temporelle.

### 9.1 Variables à forte cardinalité

Un identifiant client possède de nombreuses modalités mais peu de sens généralisable. Le one-hot encoding peut exploser la dimension. Selon le problème, on peut regrouper les modalités rares, utiliser un hachage, une représentation apprise ou une statistique de cible. Cette dernière doit être calculée hors pli (*out-of-fold*) et régularisée pour ne pas encoder directement la réponse.

### 9.2 Sélection de variables

Trois approches se complètent :

- **filtres** : variance, corrélation, information mutuelle ; rapides mais indépendants du modèle final ;
- **méthodes intégrées** : régularisation L1, importance d'arbres ; liées aux hypothèses du modèle ;
- **méthodes enveloppantes** : sélection récursive évaluée par validation croisée ; plus coûteuses.

L'importance n'implique pas la causalité. Deux variables corrélées peuvent se partager arbitrairement l'importance, et une variable proxy peut reproduire une information sensible.

---

## 10. Contrats de Données et Tests de Qualité

Un **contrat de données** décrit ce qu'un consommateur peut attendre d'une source. Il transforme des hypothèses implicites en tests automatisés.

| Dimension | Question | Exemple de contrôle |
| :--- | :--- | :--- |
| Schéma | Les colonnes et types sont-ils corrects ? | `age` numérique, `date` horodatée |
| Complétude | Quel taux de valeurs manque ? | `revenu` manquant < 5 % |
| Domaine | Les valeurs sont-elles plausibles ? | `0 <= age <= 120` |
| Unicité | Une clé apparaît-elle une seule fois ? | `id_transaction` unique |
| Cohérence | Deux champs se contredisent-ils ? | fin ≥ début |
| Fraîcheur | La source a-t-elle été mise à jour ? | délai < 24 h |
| Distribution | Le profil a-t-il changé ? | catégories inconnues, quantiles déplacés |

Un profil minimal par colonne contient : type, unité, taux de valeurs manquantes, cardinalité, quantiles ou fréquences, valeurs interdites et exemples. Versionnez le schéma, le code de préparation, le jeu de données ou son empreinte, et les paramètres appris.

### 10.1 Que faire lorsqu'un test échoue ?

Un bon contrôle associe un comportement explicite à chaque gravité :

- **bloquer** le pipeline si l'entrée est dangereuse ou invalide ;
- **mettre en quarantaine** les lignes concernées ;
- **continuer avec alerte** si l'écart est tolérable et documenté ;
- **revenir à un système de secours** si la source principale manque.

Ne remplacez pas silencieusement une colonne absente par des zéros : ce choix change le sens des données.

---

## 11. Biais, Vie Privée, Sécurité et Gouvernance

### 11.1 Biais et équité

Évaluez la qualité globale **et par sous-groupe pertinent** : rappel, faux positifs, calibration et taux de sélection peuvent diverger. Le choix d'une notion d'équité dépend du contexte et certaines propriétés sont incompatibles lorsque les taux de base diffèrent. Il faut donc documenter le compromis, les personnes affectées et la procédure de recours, pas seulement produire un score unique.

### 11.2 Minimisation et protection

- ne collectez que les données nécessaires à une finalité déclarée ;
- séparez identifiants directs et variables d'analyse ;
- chiffrez les données en transit et au repos ;
- appliquez des droits par rôle et journalisez les accès ;
- définissez une durée de conservation et un processus de suppression ;
- évitez d'afficher des données personnelles dans les logs et notebooks partagés.

La pseudonymisation réduit certains risques, mais ne garantit pas l'anonymat : des croisements peuvent réidentifier des individus.

### 11.3 Fiche de données (*datasheet*)

Pour chaque jeu, consignez : motivation, producteurs, méthode de collecte, période, population, exclusions, consentement ou base juridique applicable, nettoyage, usages prévus, usages déconseillés, limites, version et contact responsable. Cette documentation accompagne le jeu plutôt que de dépendre de la mémoire de l'équipe.

---

## 12. Données en Production et Dérive

En production, trois changements doivent être distingués :

- **dérive des entrées** : $P(X)$ change, par exemple l'âge moyen ou le type d'appareil ;
- **dérive de la cible** : $P(y)$ change, par exemple le taux de fraude ;
- **dérive conceptuelle** : $P(y\mid X)$ change, donc la relation apprise n'est plus valable.

Surveillez aussi la qualité opérationnelle : volumes, latence, erreurs de schéma, catégories inconnues, valeurs manquantes et fraîcheur. Une distance statistique peut déclencher une enquête, mais elle ne prouve pas à elle seule une baisse de performance. Lorsque les étiquettes arrivent tard, utilisez des indicateurs proxy avec prudence puis réconciliez-les avec la vérité terrain.

Un plan de réponse définit à l'avance : seuils d'alerte, propriétaire, diagnostic, retour à une version stable, réentraînement, validation et approbation de la nouvelle version.

---

## 13. Prendre en Main Pandas et Polars

Pandas et Polars manipulent des données tabulaires en Python. Ils partagent les notions de `Series` et `DataFrame`, mais leurs modèles d'exécution et leurs idiomes diffèrent.

### 13.1 Pandas : l'outil généraliste de l'écosystème Python

Un `DataFrame` Pandas est une table bidimensionnelle étiquetée dont les colonnes peuvent avoir des types différents. Une `Series` est une colonne étiquetée. L'index sert à aligner les données : cette commodité peut aussi créer des `NaN` inattendus si deux index ne correspondent pas.

```python
import pandas as pd

df = pd.read_parquet("demandes_credit.parquet")

df.head()                         # aperçu borné
df.shape                          # (lignes, colonnes)
df.info()                         # types et valeurs non nulles
df.describe(include="all").T     # statistiques numériques et catégorielles
df.isna().mean().sort_values()    # taux de valeurs manquantes
df.duplicated("id_demande").sum()
```

Opérations essentielles :

```python
# Sélection par noms, positions et condition
df.loc[df["revenu"] > 30_000, ["id_client", "revenu"]]
df.iloc[:5, :3]

# Création vectorisée d'une colonne
df["ratio_endettement"] = df["montant_credit"].div(df["revenu"])

# Agrégation par groupe
resume = (
    df.groupby("secteur", dropna=False)
      .agg(n=("id_client", "size"), taux_defaut=("defaut", "mean"))
      .sort_values("taux_defaut", ascending=False)
)

# Jointure validée : plusieurs demandes pour un client, un profil par client
df = demandes.merge(clients, on="id_client", how="left", validate="many_to_one")
```

`validate="many_to_one"` rend explicite la cardinalité attendue de la jointure et détecte une table clients non unique.

#### Types Pandas utiles

- `Int64` et `boolean` : types *nullable* pouvant conserver `pd.NA` ;
- `string` : chaîne explicite, préférable à un `object` ambigu ;
- `category` : ensemble limité de modalités, éventuellement ordonné ;
- `datetime64[ns]` et types avec fuseau horaire ;
- `timedelta64[ns]` : durées.

Convertissez explicitement et inspectez les échecs :

```python
df["date"] = pd.to_datetime(df["date_brute"], errors="coerce", utc=True)
df["montant"] = pd.to_numeric(df["montant_brut"], errors="coerce")
```

`errors="coerce"` transforme les valeurs illisibles en manquants ; il faut compter et analyser ces nouvelles valeurs au lieu de les oublier.

### 13.2 Polars : expressions, colonnes et exécution lazy

Polars utilise un moteur de requêtes colonnaire et un langage d'expressions. Les opérations principales sont `select`, `with_columns`, `filter`, `group_by` et `join`.

```python
import polars as pl

df_pl = pl.read_parquet("demandes_credit.parquet")

resume_pl = (
    df_pl
    .filter(pl.col("revenu") > 30_000)
    .with_columns(
        (pl.col("montant_credit") / pl.col("revenu"))
        .alias("ratio_endettement")
    )
    .group_by("secteur")
    .agg(
        pl.len().alias("n"),
        pl.col("defaut").mean().alias("taux_defaut"),
    )
    .sort("taux_defaut", descending=True)
)
```

L'API **eager** exécute immédiatement. L'API **lazy** construit un plan que le moteur peut optimiser avant `collect()` :

```python
requete = (
    pl.scan_parquet("demandes/*.parquet")
    .filter(pl.col("date") >= pl.date(2025, 1, 1))
    .select("id_client", "date", "revenu", "defaut")
    .group_by("id_client")
    .agg(pl.col("defaut").mean())
)

print(requete.explain())
resultat = requete.collect()
```

Le moteur peut pousser le filtre et la sélection vers la lecture afin d'éviter des lignes ou colonnes inutiles. La sémantique `null` et `NaN` doit être distinguée : `null` signale une valeur absente, tandis que `NaN` est une valeur flottante spéciale.

### 13.3 Pandas ou Polars ?

| Critère | Pandas | Polars |
| :--- | :--- | :--- |
| Prise en main | très répandu, nombreuses ressources | syntaxe d'expressions à apprendre |
| Écosystème | intégration historique très large | intégration Arrow et croissance rapide |
| Exécution | généralement eager | eager et lazy optimisable |
| Parallélisme | dépend des opérations | moteur multithread pour de nombreuses expressions |
| Données imbriquées | possible mais `object` fréquent | `List`, `Array`, `Struct` explicites |
| Usage naturel | exploration, compatibilité, petits/moyens jeux | pipelines analytiques et jeux plus volumineux |

Ils ne sont pas des ennemis. On peut explorer un petit extrait avec Pandas, transformer de gros fichiers avec Polars et remettre un tableau borné à Scikit-Learn. Mesurez sur votre charge réelle plutôt que d'affirmer qu'une bibliothèque est toujours plus rapide.

---

## 14. Exploration Descriptive et Visualisation

L'**EDA** (*Exploratory Data Analysis*) vise à comprendre la structure, la qualité et les relations avant la modélisation. Elle génère des hypothèses ; elle ne doit pas transformer le test final en terrain d'exploration.

### 14.1 Ordre d'un audit reproductible

1. **Contexte** : décision, population, période, unité d'observation, cible.
2. **Structure** : forme, colonnes, types physiques/logiques, clés.
3. **Qualité** : manquants, doublons, bornes, cohérence, fraîcheur.
4. **Analyse univariée** : distribution de chaque variable et de la cible.
5. **Analyse bivariée** : relation entre une variable et la cible.
6. **Analyse multivariée** : redondances, interactions, segments.
7. **Temps et groupes** : stabilité par période, site, client ou cohorte.
8. **Décisions** : conserver, transformer, exclure, collecter autrement.

Créez un tableau d'audit :

| Colonne | Rôle | Type logique | Manquants | Cardinalité | Décision |
| :--- | :--- | :--- | ---: | ---: | :--- |
| `id_client` | identifiant | nominal | 0 % | 1 500 | exclure du modèle |
| `revenu_annuel` | feature | ratio continu | 7 % | élevée | imputer dans pipeline |
| `score_credit` | feature | intervalle | 0 % | 500 | conserver, vérifier licence |
| `region` | audit | nominal sensible/proxy potentiel | 0 % | 5 | auditer, usage à justifier |
| `defaut_paiement` | cible | binaire | 0 % | 2 | stratifier le split |

### 14.2 Quel graphique pour quelle question ?

| Question | Graphique | Ce qu'il révèle |
| :--- | :--- | :--- |
| Quelle forme prend une variable continue ? | histogramme + densité | asymétrie, modes, queues |
| Où sont médiane et valeurs extrêmes ? | boxplot | quartiles et points éloignés |
| Comment une catégorie se répartit-elle ? | barres triées | fréquence et catégories rares |
| Deux nombres varient-ils ensemble ? | nuage de points/hexbin | forme, groupes, non-linéarité |
| Les distributions diffèrent-elles par cible ? | box/violin/histogrammes facettés | séparation et recouvrement |
| Plusieurs nombres sont-ils redondants ? | heatmap de corrélation | blocs et signes |
| La donnée évolue-t-elle ? | ligne par période | tendance, rupture, saisonnalité |
| Où sont les valeurs manquantes ? | barres ou matrice de manque | concentration et co-occurrence |

Bonnes pratiques : titre descriptif, unités, taille d'échantillon, axes lisibles, zéro pour les barres de magnitude, palette cohérente et distinction ne reposant pas uniquement sur la couleur.

### 14.3 Visualiser sans se tromper

- Une densité lissée dépend de sa bande passante.
- Un histogramme change avec le nombre de classes (*bins*).
- Un boxplot marque des observations extrêmes, pas forcément erronées.
- Une échelle logarithmique révèle une distribution très asymétrique mais doit être annoncée.
- Superposer des milliers de points cache la densité ; utiliser transparence, hexbin ou échantillon représentatif.
- Un axe tronqué peut dramatiser une différence.
- Une moyenne sans dispersion ni effectif masque l'incertitude.

---

## 15. Corrélations et Associations avec la Cible

### 15.1 Covariance et corrélation de Pearson

La covariance indique si deux variables varient dans le même sens, mais dépend des unités. La corrélation de Pearson la normalise :

$$
r_{XY}=\frac{\operatorname{cov}(X,Y)}{\sigma_X\sigma_Y}, \qquad -1\le r\le 1.
$$

- $r$ proche de $+1$ : relation linéaire croissante forte ;
- $r$ proche de $-1$ : relation linéaire décroissante forte ;
- $r$ proche de $0$ : absence de relation **linéaire**, pas absence de relation.

Avec une cible binaire codée 0/1 et une variable continue, Pearson correspond à la corrélation point-bisériale. Le signe dépend du codage de la cible.

```python
variables = ["revenu", "montant_credit", "score_credit", "incidents", "defaut"]
matrice_pearson = train[variables].corr(method="pearson", min_periods=100)
matrice_spearman = train[variables].corr(method="spearman", min_periods=100)
```

Pandas calcule les paires sur les observations disponibles pour chaque couple. Deux cellules de la matrice peuvent donc reposer sur des populations différentes lorsque des valeurs manquent. Indiquez `min_periods` et examinez les effectifs par paire si le manque est important.

### 15.2 Pearson, Spearman et Kendall

| Mesure | Détecte surtout | Sensible à | Bon usage |
| :--- | :--- | :--- | :--- |
| Pearson | relation linéaire | outliers, échelle | nombres continus, nuage approximativement linéaire |
| Spearman | relation monotone entre rangs | ex æquo, forme non monotone | ordinal, courbe monotone, robustesse |
| Kendall | concordance de paires | ex æquo | petits jeux, variables ordinales |

Toujours regarder le nuage de points. Des jeux très différents peuvent partager le même coefficient.

### 15.3 Construire une matrice lisible

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

corr = train.select_dtypes(include="number").corr(method="spearman")
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10, 7))
sns.heatmap(
    corr,
    mask=mask,
    cmap="vlag",
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
)
plt.title("Corrélations de Spearman — entraînement uniquement")
plt.tight_layout()
```

Ne mettez pas 200 variables annotées dans une heatmap illisible. Triez par association à la cible, regroupez par famille ou affichez seulement les valeurs absolues dépassant un seuil exploratoire.

### 15.4 Au-delà des variables numériques

| Variables comparées | Outil exploratoire | Commentaire |
| :--- | :--- | :--- |
| numérique–numérique | Pearson/Spearman/Kendall | linéaire ou monotone |
| numérique–binaire | point-bisériale, distributions par classe | équivaut à Pearson avec cible 0/1 |
| catégorielle–catégorielle | tableau croisé, $\chi^2$, V de Cramér | dépendance, pas force causale |
| catégorielle–numérique | groupes, ANOVA/eta² ou test non paramétrique | vérifier hypothèses et dispersion |
| relation générale | information mutuelle | non-linéaire, estimation dépendante de l'échantillon |
| modèle–feature | permutation, importance intégrée | dépend du modèle et du protocole |

Le V de Cramér normalise un test du khi-deux :

$$
V=\sqrt{\frac{\chi^2/n}{\min(k-1,r-1)}}.
$$

Il mesure une association entre deux catégories. Une petite p-value peut apparaître pour un effet minuscule sur un très grand jeu ; rapportez taille d'effet et effectifs.

### 15.5 Corrélation entre features et multicolinéarité

Deux variables très corrélées peuvent être redondantes. Cela peut :

- rendre les coefficients linéaires instables ;
- partager l'importance entre variables ;
- augmenter coût et difficulté d'explication ;
- révéler deux mesures du même phénomène ou une fuite.

Ne supprimez pas automatiquement toute paire avec $|r|>0{,}8$. Comparez sens métier, qualité, disponibilité en production, coût, stabilité, régularisation et performance validée. Une forte corrélation entre deux entrées n'empêche pas nécessairement un arbre ou un ensemble de fonctionner.

### 15.6 Les six limites à connaître

1. **Corrélation ≠ causalité.** Une cause commune peut expliquer les deux variables.
2. **Zéro ≠ indépendance.** Une relation en U peut avoir Pearson proche de zéro.
3. **Outlier.** Un seul point peut créer ou détruire une corrélation.
4. **Sous-groupes.** Une relation globale peut s'inverser dans chaque groupe (paradoxe de Simpson).
5. **Multiplicité.** Avec des milliers de variables, certaines corrélations élevées apparaissent par hasard.
6. **Fuite.** Une variable dérivée après la cible peut être parfaitement corrélée mais inutilisable.

La corrélation sert à **poser des questions**, détecter redondances et proposer des candidats. Seule une validation hors échantillon mesure l'utilité prédictive.

---

## 16. Sélection Cohérente des Variables

La sélection ne consiste pas à garder mécaniquement les plus fortes corrélations. Elle combine disponibilité, sens métier, sécurité, statistiques et validation.

### 16.1 Entonnoir de sélection

```text
colonnes brutes
  → unité/granularité correcte
  → disponible à l'instant de prédiction
  → qualité et taux de manque acceptables
  → usage licite, éthique et soutenable
  → pas d'identifiant arbitraire ni fuite
  → association/redondance étudiée sur train
  → sélection intégrée dans validation croisée
  → stabilité et gain hors échantillon
```

### 16.2 Filtre métier avant filtre statistique

| Variable de crédit | Association possible | Décision raisonnée |
| :--- | :--- | :--- |
| `id_client` | peut mémoriser des individus | exclure comme feature, conserver comme clé |
| `date_cloture_dossier` | très corrélée au défaut final | exclure : indisponible à la demande |
| `revenu_annuel` | capacité de remboursement | conserver après contrôles et imputation |
| `montant_credit` | exposition demandée | conserver |
| `ratio_endettement` | relation montant/revenu | créer sans cible, gérer revenu nul |
| `score_credit` | historique synthétique | conserver si disponible/licite, documenter sa provenance |
| `incidents_12m` | historique de paiement | conserver si fenêtre strictement antérieure |
| `region` | proxy social ou géographique | auditer ; exclure ou justifier selon usage |
| `couleur_interface` | aucune hypothèse métier | baseline bruit ; ne garder que si gain stable et explicable |
| `defaut_paiement` | réponse | cible uniquement, jamais dans $X$ |

Le mot « explicative » peut signifier **associée au sens prédictif** ou **causale**. Ce chapitre établit des variables candidates pour prédire ; il ne prouve pas qu'agir sur elles changera la cible.

### 16.3 Méthodes statistiques et modèles

- **VarianceThreshold** : retire les colonnes constantes ou quasi constantes ;
- **corrélation/association** : filtre univarié rapide, aveugle aux interactions ;
- **information mutuelle** : capture certaines relations non linéaires ;
- **SelectKBest** : choisit selon un score univarié dans chaque pli ;
- **L1** : sélection intégrée à un modèle linéaire ;
- **arbres/boosting** : importance intégrée, pouvant favoriser certaines variables ;
- **permutation** : mesure la baisse de score quand une colonne est mélangée ;
- **RFE** : élimination récursive, plus coûteuse ;
- **sélection séquentielle** : ajoute/retire selon la validation.

Toute sélection utilisant $y$ doit être ajustée **dans chaque pli du train**. Sélectionner sur tout le dataset avant validation fuit l'information.

### 16.4 Comparer des jeux de variables

Construisez au moins :

1. baseline métier minimale ;
2. toutes les variables licites ;
3. sous-ensemble sélectionné ;
4. sous-ensemble avec/sans feature engineering.

Comparez sur les mêmes plis : métrique, dispersion, latence, nombre de variables, stabilité et performance par sous-groupe. Si deux jeux sont équivalents, le plus simple, fiable et disponible est souvent préférable.

---

## 17. Démonstration de Bout en Bout

Le notebook [02_eda_correlations_pandas_polars.ipynb](02_eda_correlations_pandas_polars.ipynb) suit un jeu synthétique de demandes de crédit. Le mécanisme de génération est connu, ce qui permet de comparer ce que l'EDA retrouve à la « vérité » simulée.

Le parcours est :

```text
jeu industriel mixte
  → schéma et rôles
  → audit Pandas
  → transformations équivalentes Polars
  → distributions et valeurs manquantes
  → split avant analyse de la cible
  → matrices Pearson et Spearman sur train
  → taux de cible par catégorie + V de Cramér
  → information mutuelle
  → exclusion IDs/fuites/proxies/bruit
  → comparaison en validation croisée
  → évaluation finale du sous-ensemble figé
```

Le notebook distingue explicitement :

- **observation** : « le score de crédit est associé négativement au défaut dans le train » ;
- **hypothèse** : « ce signal pourrait être utile au modèle » ;
- **preuve prédictive** : gain stable en validation puis résultat sur test intact ;
- **affirmation causale** : non démontrée par ce protocole.

---

## 18. Checklist et Questions de Compréhension

### Checklist avant modélisation

- [ ] L'unité d'observation, la population, la cible et l'horizon sont définis.
- [ ] Chaque variable est disponible au moment de l'inférence.
- [ ] Les clés, jointures, doublons et mécanismes de valeurs manquantes sont audités.
- [ ] Le découpage reproduit les contraintes temporelles, géographiques ou de groupe.
- [ ] Toutes les transformations apprises restent dans la pipeline d'entraînement.
- [ ] Les sous-groupes, données sensibles et règles de conservation sont documentés.
- [ ] Le schéma, la provenance et la version des données sont traçables.
- [ ] Des contrôles de qualité et un plan de réaction existent pour la production.

### Questions de compréhension

1. Pourquoi une séparation aléatoire est-elle trompeuse si un patient apparaît plusieurs fois ?
2. Quelle différence faites-vous entre une valeur extrême réelle et une erreur de saisie ?
3. Pourquoi un encodage de cible calculé sur tout le jeu provoque-t-il une fuite ?
4. Donnez un exemple de variable disponible à l'entraînement mais absente au moment de la décision.
5. Une dérive de $P(X)$ implique-t-elle nécessairement une baisse de performance ? Justifiez.
6. Pourquoi un identifiant entier reste-t-il une variable nominale ?
7. Quand préférer Spearman à Pearson ?
8. Quelle mesure utiliser pour deux variables catégorielles ?
9. Pourquoi une heatmap calculée avant le split peut-elle contaminer une sélection cible-aware ?
10. Dans quel cas l'API lazy de Polars peut-elle éviter de lire des données inutiles ?

**Mini-étude de cas.** Vous prédisez les pannes d'une machine dans les 7 jours. Dessinez la ligne du temps, choisissez l'unité d'observation, listez trois variables licites et une variable fuyante, proposez un découpage, une matrice d'associations adaptée aux types, trois graphiques et deux tests de qualité.
