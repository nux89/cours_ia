# Module 3 : Théorie et Architectures du Deep Learning

> "Le Deep Learning consiste à apprendre des représentations hiérarchiques de données à travers des couches successives de transformations non-linéaires." — Yann LeCun, Yoshua Bengio, Geoffrey Hinton (Nature, 2015)

**Objectifs du module.** À l'issue de ce chapitre, vous saurez suivre les formes d'un tenseur, expliquer la rétropropagation, construire une boucle d'entraînement correcte, sélectionner architecture et fonction de perte, diagnostiquer sous/surapprentissage, décrire précisément un Transformer et évaluer un réseau au-delà de la seule loss.

**Prérequis.** Modules 1 et 2, dérivée d'une fonction simple, produit matriciel et probabilités élémentaires. Les cinq notebooks associés couvrent MLP, CNN, RNN, LSTM, attention, Transformer et auto-encodeur.

---

## 📖 Le Dico du Débutant (Jargon Buster)
Le Deep Learning utilise son propre lexique. Voici votre traducteur de poche :
- **Tenseur (*Tensor*)** : Un conteneur de nombres multidimensionnel.
  - 0D = un simple nombre (scalaire, ex: `5`).
  - 1D = une liste de nombres (vecteur, ex: `[2.5, 4.1, -1.0]`).
  - 2D = un tableau (matrice, ex: un tableur Excel de $1000$ lignes et $10$ colonnes).
  - 3D = une image couleur (Hauteur $\times$ Largeur $\times$ 3 canaux RGB).
  - 4D = un paquet d'images (*batch*, ex: 32 images de taille $224 \times 224 \times 3$).
- **Neurone artificiel** : Une petite cellule mathématique qui réalise une addition pondérée de ses entrées, ajoute un biais, et passe le tout dans un interrupteur (fonction d'activation).
- **Poids ($w$) et Biais ($b$)** : Les "boutons de réglage" du réseau. L'entraînement consiste à tourner ces millions de boutons jusqu'à ce que le réseau ne fasse plus d'erreurs.
- **Fonction de perte (*Loss*)** : Le signal optimisé pendant l'entraînement. Une perte basse sur l'entraînement ne prouve pas à elle seule que le modèle généralise, est calibré ou répond au besoin réel.
- **Époque (*Epoch*)** : Un tour complet où le modèle a examiné la totalité des exemples du jeu de données une fois.
- **Lot (*Mini-batch*)** : Un petit paquet de données (ex: 32 ou 64 exemples) envoyé d'un coup à la carte graphique (GPU) pour calculer une mise à jour des poids.
- **Taux d'apprentissage (*Learning Rate* $\eta$)** : La taille des pas de correction effectués par le modèle à chaque mise à jour.

---

## Table des Matières
1. [Du Neurone Biologique au Neurone Formel (Exemple Chiffré Pas à Pas)](#1-du-neurone-biologique-au-neurone-formel)
2. [L'Algorithme d'Apprentissage : Descente de Gradient et Rétropropagation](#2-lalgorithme-dapprentissage--descente-de-gradient-et-rétropropagation)
3. [Les Fonctions d'Activation et Leurs Enjeux](#3-les-fonctions-dactivation-et-leurs-enjeux)
4. [Le Perceptron Multicouche (MLP)](#4-le-perceptron-multicouche-mlp)
5. [Les Réseaux Convolutifs (CNN) pour la Vision](#5-les-réseaux-convolutifs-cnn-pour-la-vision)
6. [Les Réseaux Récurrents (RNN) et la Problématique Temporelle](#6-les-réseaux-récurrents-rnn-et-la-problématique-temporelle)
7. [Les Cellules LSTM et GRU : Résolution du Vanishing Gradient](#7-les-cellules-lstm-et-gru--résolution-du-vanishing-gradient)
8. [La Révolution de l'Attention et les Transformers](#8-la-révolution-de-lattention-et-les-transformers)
9. [Panorama des Autres Architectures Modernes](#9-panorama-des-autres-architectures-modernes)
10. [Tenseurs, Formes et Graphe de Calcul](#10-tenseurs-formes-et-graphe-de-calcul)
11. [Fonctions de Perte par Tâche](#11-fonctions-de-perte-par-tâche)
12. [Boucle d'Entraînement et Optimisation Robuste](#12-boucle-dentraînement-et-optimisation-robuste)
13. [Généralisation et Régularisation](#13-généralisation-et-régularisation)
14. [Le Transformer Bloc par Bloc](#14-le-transformer-bloc-par-bloc)
15. [Préentraînement, Transfert et Adaptation](#15-préentraînement-transfert-et-adaptation)
16. [Encodeurs–Décodeurs et Architectures Multimodales](#16-encodeursdécodeurs-et-architectures-multimodales)
17. [Familles Génératives et Modèles de Diffusion](#17-familles-génératives-et-modèles-de-diffusion)
18. [Évaluation, Efficacité et Déploiement](#18-évaluation-efficacité-et-déploiement)
19. [Checklist et Questions de Compréhension](#19-checklist-et-questions-de-compréhension)

---

## 1. Du Neurone Biologique au Neurone Formel

À la fin des années 1950, Frank Rosenblatt formalise et expérimente le **perceptron**, un classifieur linéaire apprenant. Il s'inscrit dans une histoire plus large des neurones formels commencée auparavant.

```text
  Entrées x_j           Poids w_j
      x_1 ────────────► w_1 ──┐
      x_2 ────────────► w_2 ──┼──► Somme pondérée z = Σ w_j x_j + b ──► Activation σ(z) ──► Sortie ŷ
      ... ────────────► ...   │
      x_d ────────────► w_d ──┘
               Biais b ───────┘
```

### 🔢 Calcul Manuel d'un Neurone Pas à Pas (Démystification)
Prenons un neurone avec seulement 2 entrées réelles :
- **Entrées (le signal reçu)** : $x_1 = 2.0$ et $x_2 = 3.0$
- **Poids synaptiques (l'importance accordée)** : $w_1 = 0.5$ et $w_2 = -1.0$
- **Biais (la tendance intrinsèque)** : $b = 1.0$

#### Étape 1 : Le calcul de la somme pondérée $z$
$$z = (x_1 \cdot w_1) + (x_2 \cdot w_2) + b$$
$$z = (2.0 \times 0.5) + (3.0 \times -1.0) + 1.0 = 1.0 - 3.0 + 1.0 = -1.0$$

#### Étape 2 : Le passage dans la fonction d'activation
- Si on utilise la fonction **ReLU** ($\max(0, z)$) :  
  $$\hat{y} = \text{ReLU}(-1.0) = \max(0, -1.0) = 0$$  
  *(Le neurone ne s'allume pas, signal bloqué)*
- Si on utilise la fonction **Sigmoïde** ($\frac{1}{1 + e^{-z}}$) :  
  $$\hat{y} = \frac{1}{1 + e^{-(-1.0)}} = \frac{1}{1 + 2.718} \approx 0.269$$  
  *(Le neurone transmet une probabilité de $26.9\%$)*

### La Limite Historique du Perceptron Simple et le XOR
Un neurone unique ne peut tracer qu'une **ligne droite** pour séparer deux classes.  
Or, la fonction logique **XOR** (vrai si l'une OU l'autre des propositions est vraie, mais faux si les deux sont vraies en même temps) est géométriquement impossible à séparer avec une seule droite :
```text
  x2 ▲
   1 │   (0) VRAI        (1) FAUX
     │
   0 │   (0) FAUX        (1) VRAI
     └──────────────────────────────► x1
         0               1
  Impossible de tracer une seule ligne droite séparant les VRAI des FAUX !
```
Pour résoudre ce problème, il faut combiner plusieurs neurones en couches : c'est la naissance du **Perceptron Multicouche (MLP)**.

---

## 2. L'Algorithme d'Apprentissage : Descente de Gradient et Rétropropagation

Comment un réseau ajuste-t-il ses millions de poids automatiquement ?

### 🏔️ L'Analogie du Randonneur dans le Brouillard
Imaginez que vous êtes un randonneur au sommet d'une montagne accidentée. Un brouillard impénétrable tombe soudainement : **vous ne voyez strictement rien à plus de 50 cm**. Vous devez impérativement rejoindre le refuge situé au fond de la vallée (le minimum de la fonction de perte $\mathcal{L}$).  
Comment faites-vous ?
1. Vous ne voyez pas le refuge au loin, mais vous sentez avec vos chaussures **la pente du terrain immédiatement sous vos pieds** (c'est le **Gradient** $\nabla \mathcal{L}$).
2. Vous faites un pas dans la direction qui descend le plus fort (le sens opposé au gradient : $-\nabla \mathcal{L}$).
3. Vous répétez cette opération pas après pas jusqu'à atteindre le fond de la cuvette.

```text
Perte (Loss)
    ▲
    │   \
    │    \   <- Position actuelle du modèle (Erreur élevée)
    │     \
    │      \   Direction : - Gradient
    │       \  ────────►
    │        \_________
    │                  \
    │                   * <- Fond de la vallée (Erreur minimale = Modèle performant !)
    └────────────────────────────────────────► Valeur du Poids w
```

### Le Taux d'Apprentissage (*Learning Rate* $\eta$) : La taille de vos pas
- **Pas trop grands ($\eta$ trop élevé)** : Le randonneur saute de 50 mètres à la fois. Il saute par-dessus la vallée, percute la falaise d'en face et s'écrase dans le précipice (l'erreur diverge vers l'infini `NaN`).
- **Pas trop petits ($\eta$ minuscule)** : Le randonneur avance de 1 millimètre par heure. Il lui faudra 3 siècles pour descendre de la montagne, et il risque de rester bloqué dans une petite flaque d'eau (un minimum local).
- **Pas adapté** : Il n'existe pas de plage universelle. La valeur dépend de l'optimiseur, de la normalisation, de la taille des lots et de l'architecture ; elle se choisit par expérimentation sur la validation.

### 2.1 Les Optimiseurs Modernes
- **SGD (Stochastic Gradient Descent)** : La descente de base. Simple mais sensible aux bosses du terrain.
- **Adam (Adaptive Moment Estimation)** : Un optimiseur adaptatif très utilisé. Il combine notamment deux idées :
  1. *Momentum* : Donne une vitesse/inertie physique au modèle pour traverser les petites bosses sans ralentir.
  2. *Taux adaptatif* : Réduit automatiquement la vitesse sur les paramètres qui changent trop vite et accélère sur ceux qui stagnent.

### 2.2 La Rétropropagation (*Backpropagation*) : Remonter la chaîne des responsabilités
Quand le réseau prédit un "Chien" alors que l'image représentait un "Chat", qui est coupable ?  
Grâce à la **règle de dérivation en chaîne (*Chain Rule*)**, l'erreur commise en sortie est redistribuée à reculons, de la dernière couche vers la première. Chaque neurone se voit attribuer une fraction de la responsabilité et ses poids sont corrigés proportionnellement :
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} \cdot (\mathbf{a}^{(l-1)})^T$$

---

## 3. Les Fonctions d'Activation et Leurs Enjeux

> 💡 **L'Analogie du Mille-feuille** : Pourquoi les fonctions d'activation non-linéaires sont-elles vitales ?  
> Si vous empilez 100 couches linéaires sans activation, cela équivaut mathématiquement à une seule grande multiplication matricielle ($\mathbf{W}_3 \times \mathbf{W}_2 \times \mathbf{W}_1 = \mathbf{W}_{\text{total}}$). C'est comme empiler 100 vitres transparentes plates : vous obtenez une vitre plus épaisse, mais vous ne pourrez jamais lui donner la forme courbée d'un pare-brise !  
> **L'activation non-linéaire est ce qui permet au réseau de "plier" l'espace pour épouser des formes complexes.**

| Fonction | Allure & Formule | Où l'utilise-t-on ? | Points forts / Faibles |
| :--- | :---: | :--- | :--- |
| **ReLU** | $\max(0, z)$ | Nombreuses couches cachées | Rapide à calculer, évite la saturation positive. Risque de neurone "mort" si les entrées restent négatives. |
| **Sigmoïde** | $\frac{1}{1 + e^{-z}} \in [0, 1]$ | Sortie binaire et portes de LSTM | Interprétable comme probabilité dans un modèle correctement entraîné ; peut saturer dans les couches profondes. |
| **Softmax** | $\frac{e^{z_i}}{\sum e^{z_j}}$ | Sortie multi-classes exclusives | Normalise $C$ scores en valeurs positives de somme 1 ; leur calibration doit être vérifiée. |
| **GELU** | $z \cdot \Phi(z)$ | Dans de nombreux Transformers | Version lisse de la ReLU, fréquemment utilisée dans les modèles de langage. |

---

## 4. Le Perceptron Multicouche (MLP)

Le MLP connecte chaque neurone d'une couche à **absolument tous** les neurones de la couche suivante (*Fully Connected*).  
Un MLP peut modéliser des données tabulaires et des images aplaties, mais il ignore la structure spatiale des **images** :
- Une photo moderne de smartphone en couleur fait $1000 \times 1000$ pixels avec 3 canaux (RGB), soit **3 millions d'entrées numériques**.
- Si la première couche cachée compte seulement 1 000 neurones, cela ferait $3 \text{ millions} \times 1000 = \mathbf{3 \text{ milliards de poids}}$ rien que pour la couche 1, soit un coût prohibitif dans de nombreux contextes.
- Un déplacement de quelques pixels change de nombreuses entrées de l'image aplatie ; le MLP ne bénéficie pas directement du partage spatial des poids.  
Les **CNN** introduisent un biais inductif local mieux adapté aux images.

---

## 5. Les Réseaux Convolutifs (CNN) pour la Vision

Les CNN exploitent des idées liées aux champs récepteurs locaux et au partage des poids. Leur développement a aussi été influencé par les travaux sur le cortex visuel, sans reproduire fidèlement son fonctionnement.

### 🔍 L'Analogie du Tampon / Filtre Photo
Au lieu de regarder toute l'image d'un coup, on fait glisser un tout petit carré de calcul (par exemple $3 \times 3$ pixels), appelé **noyau ou filtre convolutionnel**, sur toute la surface de l'image :
```text
Image d'entrée (5x5)          Filtre Détecteur de Bords (3x3)
┌───┬───┬───┬───┬───┐               ┌───┬───┬───┐
│ 0 │ 0 │ 1 │ 1 │ 1 │               │-1 │ 0 │+1 │
├───┼───┼───┼───┼───┤               ├───┼───┼───┤
│ 0 │ 0 │ 1 │ 1 │ 1 │      *        │-1 │ 0 │+1 │  ──► Carte d'activation
├───┼───┼───┼───┼───┤               ├───┼───┼───┤      (Met en valeur les contours verticaux !)
│ 0 │ 0 │ 1 │ 1 │ 1 │               │-1 │ 0 │+1 │
└───┴───┴───┴───┴───┘               └───┴───┴───┘
```
1. **Champs récepteurs locaux & Partage des poids** : Le même petit filtre de $9$ chiffres parcourt toute la photo. Qu'un chat soit en haut à gauche ou en bas à droite de l'image, le même filtre reconnaîtra ses moustaches !
2. **Sous-échantillonnage (*Pooling / MaxPool*)** : Comme une photo miniature (*thumbnail*), on divise la taille de l'image par 2 à chaque étape en ne gardant que le pixel le plus brillant de chaque zone $2 \times 2$.
3. **Hiérarchie visuelle** :
   - Couches 1-2 : Détectent les bords, contrastes et textures simples.
   - Couches 3-4 : Assemblent les bords pour reconnaître des yeux, des roues, des fenêtres.
   - Couches finales : Reconnaissent le visage complet, la voiture ou le chien.

---

## 6. Les Réseaux Récurrents (RNN) et la Problématique Temporelle

Pour lire une phrase ou écouter un son, l'ordre temporel est capital : *"Le chien mange le chat"* ne signifie pas du tout la même chose que *"Le chat mange le chien"*.

Un **RNN (Recurrent Neural Network)** lit les mots les uns après les autres tout en conservant une mémoire interne appelée **état caché** ($\mathbf{h}_t$) :
```text
  Mot 1 ("Le") ──► [ RNN ] ──► Mémoire h_1
                     │
  Mot 2 ("chat") ─► [ RNN ] ──► Mémoire h_2 (combine "chat" et la mémoire précédente h_1)
                     │
  Mot 3 ("dort") ─► [ RNN ] ──► Mémoire h_3 (combine "dort" et toute l'histoire passée)
```

### 🗣️ La Pathologie du Téléphone Arabe (*Vanishing Gradient*)
Dans le jeu du téléphone arabe, quand un message est chuchoté d'oreille en oreille à travers 20 personnes, les premiers mots sont totalement déformés ou oubliés par la 20ème personne.  
Dans un RNN classique, la rétropropagation temporelle multiplie de nombreux Jacobiens successifs. Selon les poids et les activations, le gradient peut décroître ou exploser ; la distance à laquelle le problème devient critique n'est pas universelle.

---

## 7. Les Cellules LSTM et GRU : Résolution du Vanishing Gradient

Pour guérir l'amnésie des RNN, Hochreiter et Schmidhuber inventent le **LSTM (Long Short-Term Memory)** en 1997.

### 📝 L'Analogie du Secrétaire de Direction et son Carnet
Le LSTM introduit un carnet de notes infalsifiable qui traverse le temps sans s'effacer : l'**état cellulaire** $\mathbf{C}_t$.  
À chaque nouveau mot reçu, un secrétaire intelligent applique **trois portes de décision** :
1. **La Porte d'Oubli (*Forget Gate* $\mathbf{f}_t$)** : *"Quelles informations anciennes devenues obsolètes dois-je gommer du carnet ?"* (Ex: si la phrase change de sujet, on oublie le genre grammatical du sujet précédent).
2. **La Porte d'Entrée (*Input Gate* $\mathbf{i}_t$)** : *"Quelles nouvelles informations capitales apportées par le mot actuel dois-je inscrire au propre dans le carnet ?"*
3. **La Porte de Sortie (*Output Gate* $\mathbf{o}_t$)** : *"À partir de l'état actuel de mon carnet, quel résumé dois-je communiquer au patron à cet instant précis ?"*

Le chemin additif de l'état cellulaire et les portes apprises facilitent la propagation de l'information et du gradient. Ils atténuent le problème sans garantir une mémoire parfaite ni une distance fixe.

---

## 8. La Révolution de l'Attention et les Transformers

Les LSTM imposent une dépendance séquentielle entre pas de temps, ce qui limite la parallélisation sur la longueur de séquence. Les calculs d'un pas et les lots restent néanmoins accélérables sur GPU.

En 2017, les chercheurs de Google publient l'article fondateur *"Attention Is All You Need"* et créent le **Transformer**.

### 📚 L'Analogie du Moteur de Recherche pour Comprendre Q, K, V
Le cœur du Transformer est le mécanisme d'**Attention**. Pour chaque mot, le modèle calcule trois vecteurs :
- **$\mathbf{Q}$ (Query / Requête)** : Ce que vous tapez dans la barre de recherche Google (ex: *"Quel est le statut de l'animal ?"*).
- **$\mathbf{K}$ (Key / Clé)** : Les mots-clés et titres répertoriés dans l'index de toutes les pages web existantes.
- **$\mathbf{V}$ (Value / Valeur)** : Le contenu textuel intégral de chaque page web.

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$

1. **Le produit $\mathbf{Q} \mathbf{K}^T$** mesure la compatibilité : à quel point ma Requête correspond-elle à chaque Clé ?
2. **Le Softmax** transforme ces compatibilités en pourcentages d'attention (somme = $100\%$).
3. **La multiplication par $\mathbf{V}$** extrait et mélange les Valeurs les plus pertinentes.

Dans une couche d'auto-attention dense, les positions d'une séquence peuvent être traitées en parallèle et pondérer toutes les autres positions. Cette interaction a un coût mémoire et calcul typiquement quadratique en longueur de séquence, ce qui motive des variantes d'attention plus efficaces.

---

## 9. Panorama des Autres Architectures Modernes

1. **Auto-encodeurs (AE)** : Le modèle compresse une image en quelques dizaines de chiffres (espace latent), puis apprend à la reconstruire. Idéal pour débruiter une image ou détecter des anomalies.
2. **Modèles de Diffusion (DDPM)** : Une famille de modèles qui apprend un processus de débruitage ; à la génération, elle part d'un échantillon de bruit et le transforme progressivement, éventuellement sous condition textuelle.
3. **LLMs (Grands Modèles de Langage)** : De nombreux modèles de langage fondés sur les Transformers sont préentraînés notamment à prédire des **tokens** à partir de leur contexte, puis souvent adaptés avec d'autres objectifs et données. Leurs capacités doivent être évaluées par tâche plutôt que déduites du seul objectif de préentraînement.

Pour approfondir et vérifier ces notions, consultez les articles et documentations du module dans [REFERENCES.md](../REFERENCES.md).

---

## 10. Tenseurs, Formes et Graphe de Calcul

La majorité des erreurs en deep learning sont des erreurs de **forme**, de type, d'appareil ou d'échelle. Écrivez la forme attendue après chaque opération.

| Donnée | Forme PyTorch fréquente |
| :--- | :--- |
| Table de $B$ lignes et $D$ variables | `[B, D]` |
| Images | `[B, C, H, W]` |
| Séquences d'embeddings avec `batch_first=True` | `[B, T, D]` |
| Logits binaires, un par exemple | `[B]` ou `[B, 1]` |
| Logits multiclasse | `[B, K]` |

Une couche linéaire `Linear(D_in, D_out)` transforme seulement la dernière dimension : `[B, D_in] → [B, D_out]`. Une convolution 2D conserve la dimension de lot et transforme canaux et dimensions spatiales.

### 10.1 Types et appareils

- paramètres et entrées doivent être sur le même appareil (`cpu`, `cuda`, autre accélérateur) ;
- les entrées continues sont souvent en `float32` ;
- les cibles de `CrossEntropyLoss` sont des indices entiers `long` ;
- une différence entre `[B]` et `[B, 1]` peut déclencher une diffusion implicite (*broadcasting*) indésirable.

### 10.2 Autograd et règle de la chaîne

PyTorch construit un graphe des opérations lorsque les tenseurs nécessitent un gradient. Si

$$z = wx+b, \qquad \hat{y}=\sigma(z), \qquad L=\ell(y,\hat{y}),$$

alors la règle de la chaîne donne :

$$\frac{\partial L}{\partial w}=\frac{\partial L}{\partial \hat{y}}\frac{\partial \hat{y}}{\partial z}\frac{\partial z}{\partial w}.$$

`loss.backward()` calcule ces dérivées. Les gradients **s'accumulent** par défaut : il faut les remettre à zéro avant la nouvelle rétropropagation. `optimizer.step()` applique ensuite la mise à jour.

Utilisez `torch.no_grad()` ou un mode d'inférence pour l'évaluation afin de ne pas conserver le graphe. Appelez `model.train()` pendant l'entraînement et `model.eval()` pendant l'évaluation : Dropout et Batch Normalization changent de comportement.

---

## 11. Fonctions de Perte par Tâche

La dernière couche, la forme de la cible et la loss forment un contrat indivisible.

| Tâche | Sortie du réseau | Cible | Loss courante |
| :--- | :--- | :--- | :--- |
| Régression | `[B, 1]`, sans activation imposée | flottant `[B, 1]` | MSE, MAE, Huber |
| Binaire | un **logit** par exemple | 0/1 flottant | `BCEWithLogitsLoss` |
| Multiclasse exclusive | $K$ logits | indice entier de classe | `CrossEntropyLoss` |
| Multilabel | $K$ logits indépendants | vecteur 0/1 | `BCEWithLogitsLoss` |

Un **logit** est un score non borné. `BCEWithLogitsLoss` combine sigmoïde et entropie croisée de façon numériquement stable : n'ajoutez pas une sigmoïde avant cette loss. `CrossEntropyLoss` combine log-softmax et négative log-vraisemblance : n'ajoutez pas de softmax avant elle.

Pour produire des probabilités à l'inférence : sigmoïde en binaire ou multilabel, softmax en multiclasse. La calibration et le seuil restent à valider comme au module 2.

### 11.1 Pertes de séquence et masques

Pour des séquences de longueurs variables, remplissez les positions absentes avec un symbole de padding et **masquez-les** dans l'attention et/ou la loss. Sans masque, le réseau est récompensé pour prédire le remplissage, ce qui fausse les métriques.

---

## 12. Boucle d'Entraînement et Optimisation Robuste

Une boucle minimale correcte suit cet ordre :

```python
model.train()
for x_batch, y_batch in train_loader:
    optimizer.zero_grad()
    logits = model(x_batch)
    loss = criterion(logits, y_batch)
    loss.backward()
    optimizer.step()
```

La validation ne met pas à jour les poids :

```python
model.eval()
with torch.no_grad():
    for x_batch, y_batch in valid_loader:
        logits = model(x_batch)
```

### 12.1 Optimiseurs

- **SGD** met à jour dans la direction opposée au gradient ; le momentum lisse et accélère certaines directions.
- **Adam** adapte le pas par paramètre à partir de moments du gradient ; il converge souvent vite, sans garantir la meilleure généralisation.
- **AdamW** sépare la décroissance des poids de l'adaptation du gradient ; son `weight_decay` reste à régler.

### 12.2 Taux d'apprentissage et planification

Un taux trop haut diverge ; trop bas apprend lentement ou reste bloqué. Des *schedulers* le font varier : décroissance par paliers, cosinus, plateau ou période d'échauffement (*warmup*). Comparez les courbes train/validation et ne sélectionnez pas l'époque sur le test.

### 12.3 Stabilité des gradients

- normaliser les entrées ;
- choisir une initialisation adaptée à l'activation ;
- utiliser connexions résiduelles et normalisation dans les réseaux profonds ;
- surveiller les normes de gradient ;
- appliquer un **gradient clipping** pour limiter les explosions, notamment en séquence ;
- arrêter et diagnostiquer immédiatement une loss `NaN` ou infinie.

### 12.4 Accumulation et précision mixte

L'accumulation de gradients simule un lot plus grand sur plusieurs mini-lots, à condition de normaliser correctement la loss. La précision mixte accélère certains matériels et réduit la mémoire, mais peut nécessiter une mise à l'échelle des gradients. Vérifiez toujours que les métriques restent comparables.

---

## 13. Généralisation et Régularisation

### 13.1 Lire les courbes

| Observation | Diagnostic probable | Actions possibles |
| :--- | :--- | :--- |
| Loss train et validation élevées | sous-apprentissage | capacité, variables, optimisation, durée |
| Train baisse, validation remonte | surapprentissage | régularisation, données, early stopping |
| Courbes très instables | lots trop petits, taux trop haut ou données bruitées | ajuster lot/learning rate, auditer les données |

### 13.2 Outils de régularisation

- **weight decay** : limite les poids extrêmes ;
- **dropout** : neutralise aléatoirement des activations à l'entraînement ;
- **early stopping** : conserve l'état ayant la meilleure validation ;
- **data augmentation** : crée des variantes qui préservent l'étiquette ;
- **label smoothing** : réduit la confiance excessive dans certains cas multiclasse ;
- **réduction de capacité** : réseau moins large ou moins profond.

Une augmentation doit respecter la sémantique : retourner une photographie de chat peut être valide, retourner un chiffre 6 peut changer sa classe. Les augmentations sont appliquées au train, pas au test.

### 13.3 BatchNorm et LayerNorm

**Batch Normalization** normalise à partir du lot et maintient des statistiques pour l'inférence ; elle dépend donc du mode train/eval. **Layer Normalization** normalise des dimensions de caractéristiques au sein de chaque exemple et est centrale dans les Transformers. Elles stabilisent l'optimisation mais ne remplacent ni la qualité des données ni la validation.

---

## 14. Le Transformer Bloc par Bloc

Le Transformer ne se réduit pas à la formule d'attention. Un chemin simplifié est :

```text
tokens → embeddings + positions
       → [multi-head attention → résiduel → normalisation
       →  réseau feed-forward → résiduel → normalisation] × N
       → tête de sortie
```

### 14.1 Tokenisation et embeddings

Le tokenizer découpe le texte en unités d'un vocabulaire et produit des identifiants. Une table d'embeddings transforme chaque identifiant en vecteur. La tokenisation est une partie du système versionné : changer de tokenizer change les entrées du modèle.

L'attention seule ne connaît pas l'ordre ; on ajoute donc une information positionnelle, fixe ou apprise. La longueur maximale et la manière d'extrapoler au-delà de la longueur d'entraînement sont des contraintes concrètes.

### 14.2 Auto-attention multi-têtes

Pour une séquence $X$, chaque tête projette :

$$Q=XW_Q, \qquad K=XW_K, \qquad V=XW_V.$$

Les têtes apprennent des sous-espaces différents ; leurs sorties sont concaténées puis reprojetées. La division par $\sqrt{d_k}$ limite l'amplitude des produits scalaires avant softmax.

### 14.3 Masques

- **padding mask** : interdit de porter attention aux positions de remplissage ;
- **causal mask** : interdit à une position de voir les tokens futurs pendant la génération autorégressive ;
- d'autres masques représentent des contraintes de structure.

Une erreur de masque peut divulguer la réponse future pendant l'entraînement.

### 14.4 Résiduels et feed-forward

Chaque sous-couche est entourée d'une connexion résiduelle qui facilite la circulation de l'information et du gradient. Le réseau feed-forward applique la même transformation non linéaire à chaque position. La position de la normalisation (*pre-norm* ou *post-norm*) modifie la dynamique d'entraînement.

### 14.5 Encodeur, décodeur et variantes

- **encodeur seul** : représentation bidirectionnelle, souvent utile pour classification ou extraction ;
- **décodeur seul** : prédiction causale du prochain token, adaptée à la génération ;
- **encodeur–décodeur** : encode une entrée puis génère une sortie, utile en traduction ou résumé.

Cette taxonomie décrit l'architecture, pas automatiquement la qualité, la sûreté ou la factualité.

---

## 15. Préentraînement, Transfert et Adaptation

Le **préentraînement** apprend des représentations sur une tâche source et un grand corpus. Le **transfert d'apprentissage** réutilise ces représentations pour une tâche cible. Le **fine-tuning** poursuit l'entraînement d'une partie ou de la totalité du réseau sur les données cibles.

Cette approche est particulièrement utile lorsque les données cibles sont rares, mais elle fonctionne surtout si le domaine source partage des régularités avec le domaine cible. Des poids appris sur des photographies naturelles peuvent aider pour certaines images médicales, sans garantir que les textures, échelles ou biais transférés soient pertinents.

### 15.1 Anatomie d'un modèle transféré

```text
entrée → backbone préentraîné → représentation → tête spécifique → prédiction
          (CNN/ViT/Transformer)                  (Linear, CRF, décodeur…)
```

- le **backbone** extrait une représentation générale ;
- la **tête** traduit cette représentation vers les classes, valeurs ou tokens de la tâche cible ;
- le **checkpoint** contient les paramètres appris ;
- le **prétraitement associé** fixe taille, normalisation, tokenizer et conventions d'entrée.

Charger les poids sans le prétraitement exact est une erreur fréquente. Il faut aussi remplacer la tête lorsque le nombre de classes change.

### 15.2 Quatre stratégies de transfert

| Stratégie | Paramètres entraînés | Quand l'essayer ? | Risque principal |
| :--- | :--- | :--- | :--- |
| Extracteur figé | nouvelle tête uniquement | très peu de données, domaine proche | représentation trop rigide |
| Dégel progressif | tête puis blocs profonds | compromis courant | oubli si dégel trop rapide |
| Fine-tuning complet | tous les paramètres | assez de données, domaine différent | coût et surapprentissage |
| Adaptation efficace (PEFT) | petits modules ou sous-espace | très grand modèle, ressources limitées | capacité d'adaptation bornée |

Les adaptateurs, *prompt tuning* et méthodes de mise à jour de faible rang sont des familles de PEFT. Elles réduisent les paramètres entraînables, mais le backbone complet reste souvent nécessaire à l'inférence.

### 15.3 Procédure progressive recommandée

1. établir une baseline avec le backbone figé ;
2. entraîner la nouvelle tête jusqu'à stabilisation ;
3. dégeler le dernier bloc et utiliser un taux d'apprentissage plus faible ;
4. dégeler davantage seulement si la validation progresse ;
5. comparer à une initialisation aléatoire et à une baseline non neuronale ;
6. évaluer sur le test une seule fois, y compris par sous-groupe.

On utilise souvent des taux d'apprentissage **différentiels** : plus faible dans les premières couches, plus élevé dans la tête récente. Les premières couches contiennent souvent des motifs plus généraux ; les dernières sont plus liées à la tâche source.

```python
# Schéma conceptuel : les noms exacts dépendent de l'architecture.
for parameter in model.backbone.parameters():
    parameter.requires_grad = False

model.head = nn.Linear(model.feature_dim, n_classes)
optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
```

Après dégel :

```python
for parameter in model.backbone.last_block.parameters():
    parameter.requires_grad = True

optimizer = torch.optim.AdamW([
    {"params": model.backbone.last_block.parameters(), "lr": 1e-5},
    {"params": model.head.parameters(), "lr": 1e-4},
])
```

### 15.4 Oubli catastrophique et décalage de domaine

Un taux trop élevé ou un petit jeu trop homogène peut détruire rapidement les représentations utiles : c'est l'**oubli catastrophique**. Surveillez la validation, limitez le taux, employez early stopping et conservez le checkpoint initial.

Le **negative transfer** se produit lorsque le préentraînement dégrade la tâche cible. Détectez-le en comparant au même modèle initialisé aléatoirement et à une baseline simple. Examinez également : résolution, couleurs, vocabulaire, langue, capteurs, licence, population et période des données sources.

### 15.5 Adapter un modèle de langage

Plusieurs objectifs répondent à des besoins différents :

- **préentraînement continu** : poursuivre l'objectif de langage sur un domaine spécialisé ;
- **fine-tuning supervisé** : apprendre à suivre des exemples entrée–sortie ;
- **alignement par préférences** : favoriser certaines réponses à partir de comparaisons ou signaux de préférence ;
- **distillation** : transférer le comportement d'un modèle professeur vers un élève.

Un modèle adapté peut mémoriser des exemples sensibles ou perdre des capacités générales. Dédupliquez les données, séparez les évaluations, testez la mémorisation et versionnez jeu, recette, checkpoint et gabarit de prompt.

### 15.6 Quand préférer le RAG au fine-tuning ?

Le fine-tuning adapte surtout un **comportement**, un format ou une distribution de tâche ; il n'est pas une base documentaire fiable. Pour des connaissances privées, changeantes et devant être citées, la récupération documentaire (RAG, module 4) est souvent plus appropriée. Les deux approches peuvent être combinées.

| Besoin | RAG | Fine-tuning |
| :--- | :---: | :---: |
| Mettre à jour des faits fréquemment | ✓ | peu adapté seul |
| Citer les documents utilisés | ✓ | difficile |
| Apprendre un format de sortie stable | possible | ✓ |
| Adapter le ton ou une tâche | possible | ✓ |
| Retirer rapidement un document | ✓ | réentraînement parfois requis |
| Réduire le contexte par requête | parfois non | potentiellement |

La bonne question est souvent « quelle combinaison minimise les erreurs, le coût et le délai de mise à jour ? », pas « lequel remplace l'autre ? »

---

## 16. Encodeurs–Décodeurs et Architectures Multimodales

Une architecture **encodeur–décodeur** sépare deux responsabilités :

1. l'encodeur transforme une entrée en représentation ;
2. le décodeur produit la sortie conditionnée par cette représentation et par ses sorties précédentes.

```text
entrée x → ENCODEUR → représentation z → DÉCODEUR → sortie y₁, y₂, …, y_T
```

L'entrée et la sortie peuvent avoir des longueurs et même des modalités différentes : texte→texte, image→texte, audio→texte, texte→image ou vidéo→texte.

### 16.1 Séquence vers séquence : RNN/LSTM

Dans une architecture historique de traduction :

- un LSTM encode les tokens source en états cachés ;
- un autre LSTM génère les tokens cible un par un ;
- un symbole `<BOS>` amorce la génération et `<EOS>` l'arrête ;
- un masque ignore le padding dans la loss.

À l'entraînement, le **teacher forcing** fournit parfois le vrai token précédent au décodeur. À l'inférence, le décodeur reçoit ses propres prédictions : ce décalage peut accumuler les erreurs (*exposure bias*).

Sans attention, compresser toute une longue séquence dans un seul vecteur crée un goulot d'étranglement. Avec attention, le décodeur pondère les états de l'encodeur à chaque pas.

### 16.2 CNN + LSTM pour comprendre une séquence visuelle

Un CNN traite l'espace ; un LSTM traite une succession. Leur association est naturelle quand chaque instant contient une image : vidéo, gestes, imagerie médicale temporelle ou capteurs visuels.

```text
vidéo [B,T,C,H,W]
  └─ CNN partagé sur chaque image → caractéristiques [B,T,D]
       └─ LSTM sur T → états [B,T,H]
            └─ tête → classe, score ou séquence
```

Le même CNN doit être appliqué à chaque pas : on partage ses poids. On peut utiliser le dernier état du LSTM pour classifier toute la vidéo, ou tous les états pour prédire une étiquette par image.

Alternatives :

- convolution 3D pour apprendre conjointement temps et espace ;
- Transformer temporel sur les caractéristiques visuelles ;
- pooling temporel simple comme baseline ;
- ConvLSTM lorsque l'état caché doit conserver une organisation spatiale.

Le choix dépend de la longueur, du mouvement, des données et du coût. Un LSTM n'est pas automatiquement meilleur qu'une moyenne temporelle : comparez-les.

### 16.3 Image-to-caption : CNN/ViT vers LSTM

Le **captioning d'image** génère une phrase décrivant une image. Une architecture classique associe :

```text
image [B,C,H,W]
  → CNN préentraîné
  → vecteur visuel [B,D]
  → projection vers l'état initial du LSTM
  → LSTM + tête vocabulaire
  → « un chien court dans l'herbe »
```

À l'entraînement, pour une légende tokenisée $w_1,\ldots,w_T$, le modèle minimise :

$$
\mathcal{L}=-\sum_{t=1}^{T}\log p(w_t\mid w_{<t},\mathrm{image}).
$$

Formes typiques :

- caractéristiques globales : `[B, D_img]` ;
- tokens d'entrée : `[B, T]` ;
- embeddings texte : `[B, T, D_txt]` ;
- logits de vocabulaire : `[B, T, V]` ;
- cible : `[B, T]`, décalée d'un token.

Une seule représentation globale perd les détails spatiaux. Une version avec **attention visuelle** conserve une grille `[B, N_regions, D]` ; à chaque mot, le décodeur choisit les régions pertinentes. Cela permet d'associer « chien » à une région et « herbe » à une autre.

### 16.4 Image-to-caption moderne : encodeur visuel + Transformer

Un CNN ou Vision Transformer encode l'image en une séquence de *visual tokens*. Un décodeur Transformer utilise :

- une **self-attention causale** entre les tokens textuels déjà générés ;
- une **cross-attention** où les requêtes viennent du texte et les clés/valeurs de l'image ;
- une tête de langage qui prédit le token suivant.

```text
image → encodeur visuel → tokens visuels ───────────────┐
                                                        ▼ K,V
tokens texte → self-attention causale → cross-attention → FFN → prochain token
                                              Q
```

Dans la cross-attention :

$$Q=H_{texte}W_Q, \qquad K=H_{image}W_K, \qquad V=H_{image}W_V.$$

Le mécanisme aligne dynamiquement les mots en cours de génération avec les régions visuelles.

### 16.5 Alignement texte–image par apprentissage contrastif

Une autre famille n'est pas générative : elle apprend un espace partagé.

```text
image → encodeur image → vecteur v
texte → encodeur texte → vecteur t
objectif : rapprocher les paires correspondantes, éloigner les autres
```

Pour un lot de paires image–texte, on calcule les similarités cosinus entre tous les vecteurs et on optimise une classification contrastive dans les deux directions. Ce type de modèle permet : recherche texte→image, image→texte, classification *zero-shot* par descriptions et initialisation de systèmes multimodaux.

**Différence essentielle :** un modèle contrastif retrouve ou classe ; un modèle de captioning génère une séquence. On peut combiner représentation contrastive, décodeur génératif et récupération d'exemples.

### 16.6 Fusion multimodale

Trois stratégies générales :

- **fusion précoce** : concaténer les représentations avant les couches principales ;
- **fusion tardive** : combiner les scores de modèles séparés ;
- **fusion par attention** : laisser une modalité interroger l'autre.

Avant la fusion, projetez les modalités vers des dimensions compatibles et indiquez clairement masques, positions et segments. Les données doivent être alignées : une mauvaise correspondance image–légende apprend des associations erronées.

#### Trois architectures texte–image à ne pas confondre

| Architecture | Circulation de l'information | Atout | Usage typique |
| :--- | :--- | :--- | :--- |
| **Double encodeur** | image et texte encodés séparément, comparaison finale | indexation rapide, embeddings pré-calculables | recherche image–texte, zero-shot |
| **Cross-encodeur / fusion** | tokens visuels et textuels interagissent dans un même réseau | interactions fines | classement de paires, VQA, classification multimodale |
| **Encodeur–décodeur** | entrée encodée, autre modalité générée séquentiellement | production de contenu | image-to-caption, texte-to-image |

Un double encodeur est efficace pour chercher parmi un million d'images : on pré-calcule les vecteurs image puis on compare la requête textuelle. Un cross-encodeur est plus précis pour départager quelques candidats, mais il doit recalculer les interactions pour chaque paire. On les combine souvent en récupération puis reranking.

#### Exemple 1 : classifier une annonce avec photo et texte

On veut prédire la catégorie d'un produit à partir d'une image et de sa description.

```text
image [B,C,H,W] ─► encodeur visuel ─► v_img [B,D_img] ─► projection [B,D]
                                                                  ├► concaténation [B,2D]
texte [B,T] ─────► encodeur texte ──► v_txt [B,D_txt] ─► projection [B,D]
                                                                  └► MLP → logits [B,K]
```

La loss est une entropie croisée multiclasse. Une expérience complète compare : image seule, texte seul, fusion des deux. Cette **ablation** révèle si les deux modalités apportent vraiment une information complémentaire. Il faut aussi tester image manquante, texte vide et contradiction entre les modalités.

#### Exemple 2 : répondre à une question sur une image

Pour la question « combien de vélos sont visibles ? » :

1. l'encodeur visuel produit des tokens correspondant à des régions ou patches ;
2. l'encodeur textuel représente la question ;
3. la cross-attention relie notamment « combien » et « vélos » aux régions pertinentes ;
4. une tête de classification ou un décodeur génère la réponse.

Cette tâche de **Visual Question Answering** demande un alignement plus fin qu'une simple similarité globale. Elle peut échouer par mauvaise perception, mauvaise compréhension de la question, mauvais comptage ou biais appris ; l'analyse d'erreur doit distinguer ces causes.

#### Fine-tuning d'un système multimodal

Une recette prudente est :

1. charger les deux encodeurs avec leurs prétraitements officiels ;
2. figer les encodeurs et entraîner projections, fusion et tête ;
3. mesurer les baselines unimodales ;
4. dégeler les derniers blocs avec un taux plus faible ;
5. surveiller séparément qualité visuelle, qualité textuelle et performance jointe ;
6. tester les modalités absentes, bruitées ou contradictoires ;
7. vérifier que les paires train/test ne sont pas des quasi-doublons.

Si un encodeur est beaucoup plus puissant, le modèle peut ignorer l'autre modalité. La suppression aléatoire contrôlée d'une modalité à l'entraînement, des pertes auxiliaires ou un objectif contrastif peuvent encourager l'utilisation des deux, mais doivent être validés sur le cas d'usage.

### 16.7 Décodage d'une légende

- **greedy decoding** : choisit le token le plus probable ; rapide mais myope ;
- **beam search** : conserve plusieurs séquences candidates ; plus coûteux et parfois répétitif ;
- **échantillonnage** : augmente la diversité avec température, top-k ou noyau ;
- pénalités de répétition et contraintes peuvent corriger certains défauts.

Évaluez avec plusieurs références lorsque possible. BLEU, ROUGE, METEOR, CIDEr ou scores sémantiques capturent des aspects différents et ne remplacent pas une revue humaine de la fidélité visuelle. Vérifiez particulièrement objets inventés, relations spatiales, comptage, texte présent dans l'image et biais sociaux.

### 16.8 Associations fréquentes au-delà de CNN–LSTM

| Entrée → sortie | Encodeur | Décodeur/tête | Exemple |
| :--- | :--- | :--- | :--- |
| image → masque | CNN/ViT | décodeur convolutionnel avec skip connections | segmentation |
| image → texte | CNN/ViT | LSTM ou Transformer | captioning, OCR raisonné |
| audio → texte | convolutions/audio Transformer | Transformer causal | transcription |
| texte → texte | Transformer encodeur | Transformer décodeur | traduction, résumé |
| texte → image | encodeur texte | débruiteur conditionné + décodeur latent | diffusion conditionnelle |
| vidéo → texte | CNN/ViT par frame + encodeur temporel | décodeur de langage | résumé vidéo |

Dans tous les cas, documentez où se trouve le goulot d'information, quels blocs sont préentraînés, lesquels sont figés, comment les modalités sont alignées et quelle loss entraîne chaque bloc.

---

## 17. Familles Génératives et Modèles de Diffusion

| Famille | Idée centrale | Usages typiques |
| :--- | :--- | :--- |
| Autorégressive | factorise une séquence et prédit le prochain élément | texte, code, audio |
| Auto-encodeur variationnel (VAE) | apprend une distribution latente régularisée | génération, représentation |
| GAN | oppose générateur et discriminateur | images, traduction de domaine |
| Diffusion | apprend à inverser progressivement un processus de bruitage | image, audio, vidéo |

Une génération est un échantillon conditionné, pas une récupération garantie d'un fait. Température, stratégie d'échantillonnage et graine modifient la diversité. Évaluez qualité, diversité, fidélité à la consigne, mémorisation, contenus indésirables et droits d'utilisation.

### 17.1 Auto-encodeur et VAE

Un auto-encodeur déterministe apprend $z=E(x)$ puis $\hat{x}=D(z)$. Il sert à compresser, débruiter ou détecter des anomalies, mais son espace latent n'est pas nécessairement facile à échantillonner.

Un VAE apprend une distribution latente, généralement paramétrée par $\mu(x)$ et $\sigma(x)$. Sa loss combine reconstruction et divergence vers une distribution de référence. Le tour de **reparamétrisation** permet d'échantillonner tout en propageant le gradient :

$$z=\mu+\sigma\odot\varepsilon, \qquad \varepsilon\sim\mathcal{N}(0,I).$$

### 17.2 GAN

Le générateur transforme un bruit en exemple ; le discriminateur tente de distinguer vraies et fausses données. L'entraînement est un jeu adversarial difficile à équilibrer. Les risques incluent instabilité et **mode collapse**, lorsque la diversité produite diminue fortement.

### 17.3 Diffusion : intuition

Un modèle de diffusion définit :

1. un processus avant qui ajoute graduellement du bruit à une donnée $x_0$ jusqu'à obtenir presque un bruit gaussien $x_T$ ;
2. un processus appris qui retire le bruit étape par étape pour générer un nouvel échantillon.

```text
entraînement : image x₀ → + bruit → xₜ → réseau prédit le bruit ajouté
génération   : bruit x_T → débruitages successifs → image x₀
```

Une écriture courante du bruitage direct permet d'échantillonner directement un niveau $t$ :

$$x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\,\varepsilon,
\qquad \varepsilon\sim\mathcal{N}(0,I).$$

Le réseau $\varepsilon_\theta(x_t,t,c)$ reçoit l'exemple bruité, le pas de temps et éventuellement une condition $c$. Un objectif simplifié minimise :

$$\mathbb{E}_{x_0,t,\varepsilon}\left[\|\varepsilon-\varepsilon_\theta(x_t,t,c)\|_2^2\right].$$

Le réseau de débruitage est souvent de type U-Net pour les images, avec connexions résiduelles, embeddings temporels et blocs d'attention.

### 17.4 Diffusion conditionnelle texte→image

Un encodeur texte produit des représentations de la consigne. Le débruiteur y accède généralement via cross-attention : les caractéristiques visuelles bruitées interrogent les tokens textuels.

```text
prompt → encodeur texte → condition c ───────────────┐
                                                     ▼
bruit latent z_T → U-Net/Transformer débruiteur × T → z_0 → décodeur → image
```

La **classifier-free guidance** combine typiquement une prédiction conditionnelle et une prédiction sans condition. Une guidance plus forte suit souvent davantage le prompt mais peut réduire diversité ou naturalité ; c'est un compromis à évaluer.

### 17.5 Diffusion dans l'espace latent

Débruiter directement des pixels haute résolution est coûteux. Une diffusion latente utilise :

1. un encodeur qui compresse l'image vers $z$ ;
2. un modèle de diffusion qui agit dans cet espace ;
3. un décodeur qui reconstruit les pixels.

La compression accélère l'entraînement et l'échantillonnage, au prix d'une limite imposée par l'auto-encodeur. Les petits textes et détails fins peuvent être perdus.

### 17.6 Entraînement, sampling et évaluation

Le planning de bruit fixe la quantité ajoutée à chaque pas. Le nombre de pas d'échantillonnage échange vitesse contre qualité ; des solveurs et méthodes de distillation peuvent le réduire. Pour une comparaison loyale, fixez prompt, résolution, sampler, nombre de pas, guidance et graine.

Évaluez :

- fidélité au texte et relations entre objets ;
- qualité et diversité visuelles ;
- visages, mains, texte rendu et détails fins ;
- biais, contenus sensibles, mémorisation et proximité aux données ;
- latence, mémoire et coût énergétique ;
- préférences humaines avec protocole en aveugle lorsque pertinent.

Une mesure distributionnelle ou un score texte–image ne suffit pas seul. Une image esthétique peut contredire le prompt ; une image fidèle peut reproduire un stéréotype ou un élément mémorisé.

---

## 18. Évaluation, Efficacité et Déploiement

### 18.1 Évaluer le système complet

Conservez la métrique métier du module 2 et ajoutez :

- robustesse aux perturbations et au changement de domaine ;
- calibration et abstention ;
- latence médiane et percentiles élevés ;
- mémoire, débit, énergie ou coût par requête ;
- performance par sous-groupe ;
- reproductibilité et sensibilité aux graines.

Pour la génération, une métrique automatique unique est rarement suffisante : combinez tests déterministes, jeux d'évaluation représentatifs, comparaison humaine définie par une grille et analyse des erreurs.

### 18.2 Compression et service

- **quantification** : représente poids/activations avec moins de bits ;
- **pruning** : retire des paramètres ou structures ;
- **distillation** : entraîne un modèle élève à reproduire un professeur ;
- **batching et cache** : améliorent le débit mais modifient latence et mémoire.

Toute optimisation peut dégrader certaines classes ou entrées : revalidez le modèle compressé, pas seulement l'original.

### 18.3 Artefacts à versionner

Poids, architecture, tokenizer, normalisation, mapping de classes, seuil, dépendances, code d'inférence, configuration matérielle, données et rapport d'évaluation doivent être traçables. Un modèle chargé avec le mauvais prétraitement peut s'exécuter sans erreur tout en produisant des résultats faux.

---

## 19. Checklist et Questions de Compréhension

### Checklist d'entraînement

- [ ] Les formes, types et appareils sont vérifiés sur un mini-lot.
- [ ] La dernière couche, la cible et la loss sont compatibles.
- [ ] Les gradients sont remis à zéro et l'évaluation désactive leur calcul.
- [ ] Les modes `train()` et `eval()` sont correctement utilisés.
- [ ] Train, validation et test restent séparés.
- [ ] Les courbes, gradients et erreurs par segment sont inspectés.
- [ ] Le meilleur état est sauvegardé avec sa configuration.
- [ ] Une baseline et une expérience d'ablation sont disponibles.
- [ ] Le modèle final est réévalué après compression ou conversion.

### Questions de compréhension

1. Pourquoi ne faut-il pas appliquer une sigmoïde avant `BCEWithLogitsLoss` ?
2. Que se passe-t-il si l'on oublie `optimizer.zero_grad()` ?
3. Quelle différence entre padding mask et causal mask ?
4. Pourquoi `model.eval()` est-il nécessaire même avec `torch.no_grad()` ?
5. Dans quel cas un RAG est-il préférable à un fine-tuning pour ajouter des connaissances ?
6. Comment circulent les tenseurs dans une architecture CNN–LSTM appliquée à une vidéo ?
7. Quelle différence entre un modèle contrastif texte–image et un générateur de légendes ?
8. Quel rôle joue la cross-attention dans un modèle image-to-caption ou texte-to-image ?
9. Pourquoi un modèle de diffusion latent est-il généralement moins coûteux qu'une diffusion en pixels ?

**Mini-étude de cas 1.** Pour classifier dix catégories d'images, donnez les formes à l'entrée et à la sortie, la loss, le type de cible, deux augmentations valides, une baseline et quatre métriques ou contraintes de déploiement.

**Mini-étude de cas 2.** Concevez un système image-to-caption. Choisissez l'encodeur et le décodeur, indiquez les formes intermédiaires, le décalage entrée/cible, les masques, le mode de décodage, une stratégie de transfert et un protocole d'évaluation humaine et automatique.
