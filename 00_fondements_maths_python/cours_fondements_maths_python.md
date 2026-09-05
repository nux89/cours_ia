# Module 0 : Fondements Mathématiques et Python pour l'IA

> Les mathématiques de l'IA ne sont pas un rite de passage : ce sont les quelques outils qui rendent le reste **compréhensible plutôt que magique**. Trois idées reviennent partout : *empiler des nombres* (algèbre linéaire), *descendre une pente* (calcul différentiel) et *raisonner sous incertitude* (probabilités). Ce module installe ces intuitions et l'outillage Python associé.

**Objectifs du module.** À l'issue de ce chapitre, vous saurez lire et manipuler vecteurs, matrices et tenseurs, interpréter un produit scalaire et une norme, comprendre ce que calcule un gradient et pourquoi on le suit, raisonner avec des probabilités et une vraisemblance, distinguer estimation et vérité, reconnaître les pièges numériques du calcul flottant, et écrire du NumPy vectorisé reproductible.

**Prérequis.** Mathématiques de fin de lycée (fonctions, dérivée d'une fonction simple, pourcentages) et Python élémentaire (variables, fonctions, boucles, listes). Rien de plus : chaque notion est réintroduite avec une analogie et un exemple chiffré. Ce module est **transversal** ; les modules 1 à 8 y renvoient dès qu'une notion mathématique apparaît.

---

## 📖 Le Dico du Débutant (Jargon Buster)
- **Scalaire / Vecteur / Matrice / Tenseur** : un nombre seul ; une liste de nombres ; un tableau de nombres ; un tableau à plus de deux dimensions.
- **Gradient** : la « pente » d'une fonction à plusieurs variables. Il indique la direction de plus forte montée ; on va dans le sens opposé pour minimiser.
- **Probabilité** : un nombre entre 0 et 1 qui quantifie à quel point on croit qu'un événement se produira.
- **Distribution** : la façon dont les valeurs possibles d'une grandeur se répartissent (où elles se concentrent, comment elles s'étalent).
- **Espérance (*moyenne théorique*)** : la valeur moyenne qu'on obtiendrait en répétant une expérience une infinité de fois.
- **Vraisemblance (*likelihood*)** : à quel point des paramètres de modèle rendent les données observées « plausibles ».
- **Vectorisation** : remplacer une boucle Python lente par une opération sur des tableaux entiers, calculée en bloc et bien plus rapidement.

---

## Table des Matières
1. [Pourquoi (un peu) de maths pour l'IA ?](#1-pourquoi-un-peu-de-maths-pour-lia)
2. [Algèbre Linéaire : le Langage des Données](#2-algèbre-linéaire--le-langage-des-données)
3. [Calcul Différentiel : Descendre la Pente](#3-calcul-différentiel--descendre-la-pente)
4. [Probabilités : Raisonner sous Incertitude](#4-probabilités--raisonner-sous-incertitude)
5. [Statistiques : de l'Échantillon à la Conclusion](#5-statistiques--de-léchantillon-à-la-conclusion)
6. [Optimisation : Trouver un Bon Minimum](#6-optimisation--trouver-un-bon-minimum)
7. [Théorie de l'Information : Surprise, Entropie et Entropie Croisée](#7-théorie-de-linformation--surprise-entropie-et-entropie-croisée)
8. [Python Scientifique : NumPy, Vectorisation et Pièges Numériques](#8-python-scientifique--numpy-vectorisation-et-pièges-numériques)
9. [Checklist et Questions de Compréhension](#9-checklist-et-questions-de-compréhension)

---

## 1. Pourquoi (un peu) de maths pour l'IA

On peut appeler des bibliothèques sans jamais écrire une dérivée à la main. Mais dès qu'un modèle diverge, qu'une perte reste bloquée, qu'une métrique semble « trop belle » ou qu'un tenseur a la mauvaise forme, la compréhension mathématique fait la différence entre **corriger** et **deviner**.

Trois blocs suffisent pour l'essentiel du cursus :

| Bloc | Question à laquelle il répond | Où il resurgit |
| :--- | :--- | :--- |
| Algèbre linéaire | Comment représenter et transformer des données en masse ? | $X$, embeddings, couches, attention (modules 1, 2, 3) |
| Calcul différentiel | Dans quelle direction ajuster les paramètres ? | descente de gradient, rétropropagation (modules 2, 3) |
| Probabilités & statistiques | Que puis-je affirmer, et avec quelle confiance ? | métriques, incertitude, calibration (modules 1, 2) |

> 💡 **Le bon niveau d'exigence.** Vous n'avez pas besoin de *démontrer* les théorèmes, mais de *lire* les formules et d'en avoir l'intuition. Objectif : qu'une équation du cours vous dise « ah, c'est juste une moyenne pondérée » plutôt que de vous bloquer.

---

## 2. Algèbre Linéaire : le Langage des Données

### 2.1 Scalaires, vecteurs, matrices, tenseurs

Tout jeu de données du cursus est un empilement de nombres :

```text
scalaire  5                      → 0 dimension
vecteur   [2.5, 4.1, -1.0]       → 1 dimension (un individu, un embedding)
matrice   [[25, 32],             → 2 dimensions (un tableau X : lignes × colonnes)
           [45, 65],
           [38, 48]]
tenseur   images [B, C, H, W]    → 3+ dimensions (module 3)
```

Notations usuelles : un vecteur $\mathbf{x}\in\mathbb{R}^d$ (d nombres), une matrice $\mathbf{X}\in\mathbb{R}^{n\times d}$ ($n$ lignes, $d$ colonnes). C'est exactement le $\mathbf{X}$ et le $\mathbf{y}$ du module 1.

### 2.2 Les opérations qui comptent

**Addition et multiplication par un scalaire** (élément par élément) :
$$2\cdot[1, 3] + [0, 4] = [2, 10].$$

**Produit scalaire (*dot product*)** — l'opération la plus importante de tout le cursus. Il combine deux vecteurs en **un seul nombre** :
$$\mathbf{a}\cdot\mathbf{b}=\sum_{j=1}^d a_j b_j.$$

*Exemple chiffré* : poids d'un neurone $\mathbf{w}=[0.5, -1.0]$, entrée $\mathbf{x}=[2.0, 3.0]$ :
$$\mathbf{w}\cdot\mathbf{x}=(0.5\times 2.0)+(-1.0\times 3.0)=1.0-3.0=-2.0.$$
C'est littéralement la somme pondérée d'un neurone (module 3) et le cœur d'une régression linéaire (module 2).

**Produit matriciel** — appliquer la même combinaison à toutes les lignes d'un coup. Si $\mathbf{X}$ est $[n\times d]$ et $\mathbf{w}$ est $[d]$, alors $\mathbf{X}\mathbf{w}$ est $[n]$ : une prédiction par ligne. La règle des formes est **incontournable** : le nombre de colonnes de gauche doit égaler le nombre de lignes de droite.

$$[n\times d]\cdot[d\times k]=[n\times k].$$

> ⚠️ **La majorité des bugs de deep learning sont des erreurs de forme.** Écrivez la forme attendue après chaque opération (conseil repris au module 3).

**Transposée** $\mathbf{X}^T$ : échange lignes et colonnes ($[n\times d]\to[d\times n]$).

### 2.3 Norme, distance et similarité

La **norme** mesure la « longueur » d'un vecteur :
$$\|\mathbf{x}\|_2=\sqrt{\textstyle\sum_j x_j^2}\quad(\text{norme euclidienne, }L_2).$$

La **distance euclidienne** entre deux points est la norme de leur différence — c'est ce que calculent k-NN et K-Means (modules 1 et 2) :
$$d(\mathbf{a},\mathbf{b})=\|\mathbf{a}-\mathbf{b}\|_2.$$

La **similarité cosinus** mesure l'angle, en ignorant les longueurs — centrale pour comparer des embeddings et pour le RAG (modules 3 et 4) :
$$\cos(\mathbf{a},\mathbf{b})=\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}\in[-1, 1].$$

> 💡 **Pourquoi la mise à l'échelle compte.** Dans $d(\mathbf{a},\mathbf{b})$, une variable exprimée en dizaines de milliers (un salaire) écrase une variable exprimée en dizaines (un âge). C'est la justification mathématique du *scaling* vu au module 1.

### 2.4 Notions à connaître de nom (intuition suffit)

| Notion | Intuition | Usage dans le cursus |
| :--- | :--- | :--- |
| Rang | nombre de directions vraiment indépendantes | colinéarité, redondance (module 1) |
| Inverse $\mathbf{A}^{-1}$ | « annuler » une transformation | solution analytique de la régression |
| Valeurs/vecteurs propres | directions que la matrice se contente d'étirer | PCA (module 2) |
| SVD | décomposer une matrice en facteurs | PCA, réduction, systèmes de recommandation |
| Matrice orthogonale | rotation/réflexion qui préserve les longueurs | stabilité numérique |

La **PCA** du module 2 revient à trouver les directions de plus grande variance : ce sont les vecteurs propres de la matrice de covariance, ou les composantes d'une SVD des données centrées.

---

## 3. Calcul Différentiel : Descendre la Pente

### 3.1 La dérivée, c'est une pente

La dérivée $f'(x)$ répond à : « si j'augmente $x$ d'un tout petit peu, de combien change $f(x)$, et dans quel sens ? »

- $f'(x)>0$ : la fonction monte ;
- $f'(x)<0$ : la fonction descend ;
- $f'(x)=0$ : terrain plat (sommet, creux ou palier).

*Exemple chiffré* : pour $f(x)=x^2$, on a $f'(x)=2x$. En $x=3$, $f'(3)=6$ : la courbe monte fort. En $x=0$, $f'(0)=0$ : c'est le fond de la cuvette, le minimum.

### 3.2 Le gradient : une pente par variable

Quand une fonction dépend de plusieurs variables (des millions de poids !), sa dérivée devient un **vecteur de dérivées partielles**, le **gradient** $\nabla f$. Chaque composante dit comment bouge $f$ quand on tourne **un seul** bouton en gelant les autres.

$$\nabla f=\left[\frac{\partial f}{\partial w_1}, \frac{\partial f}{\partial w_2}, \ldots, \frac{\partial f}{\partial w_d}\right].$$

Le gradient pointe vers la plus forte **montée**. Pour **minimiser** une perte, on va donc dans le sens opposé : c'est **la descente de gradient**, moteur des modules 2 et 3.

$$\mathbf{w}_{\text{nouveau}}=\mathbf{w}_{\text{ancien}}-\eta\,\nabla \mathcal{L}(\mathbf{w}),$$

où $\eta$ (le *taux d'apprentissage*) est la taille du pas. C'est l'analogie du randonneur dans le brouillard du module 3, formalisée.

### 3.3 La règle de la chaîne : le cœur de la rétropropagation

Si une sortie dépend de $z$, qui dépend de $w$, alors :
$$\frac{\partial \mathcal{L}}{\partial w}=\frac{\partial \mathcal{L}}{\partial z}\cdot\frac{\partial z}{\partial w}.$$

On multiplie les pentes le long de la chaîne. Un réseau de neurones n'est qu'une **longue composition de fonctions** ; la rétropropagation (module 3) applique cette règle en remontant de la sortie vers l'entrée pour attribuer à chaque poids sa part de responsabilité dans l'erreur.

*Exemple chiffré minimal.* Soit $z=wx$ avec $x=2$, et $\mathcal{L}=z^2$. Alors $\frac{\partial \mathcal{L}}{\partial z}=2z$ et $\frac{\partial z}{\partial w}=x=2$. Si $w=3$, alors $z=6$, donc $\frac{\partial \mathcal{L}}{\partial w}=2\times 6\times 2=24$. Un pas de descente avec $\eta=0.01$ donne $w\leftarrow 3-0.01\times 24=2.76$ : on a réduit $w$, ce qui réduit $\mathcal{L}$. ✅

> 💡 **Convexe ou non ?** Une fonction convexe (« une seule cuvette ») garantit qu'un minimum trouvé est *le* minimum. Les réseaux profonds sont non convexes (plein de creux), mais la descente de gradient y trouve en pratique des solutions utiles — sans garantie d'optimum global (voir §6).

---

## 4. Probabilités : Raisonner sous Incertitude

### 4.1 Vocabulaire de base

- **Probabilité** $P(A)\in[0, 1]$ : 0 = impossible, 1 = certain.
- **Événements complémentaires** : $P(\text{non }A)=1-P(A)$.
- **Union / intersection** : $P(A\text{ ou }B)=P(A)+P(B)-P(A\text{ et }B)$.
- **Indépendance** : $A$ et $B$ sont indépendants si $P(A\text{ et }B)=P(A)P(B)$.
- **Probabilité conditionnelle** : $P(A\mid B)=\dfrac{P(A\text{ et }B)}{P(B)}$ — « la probabilité de $A$ **sachant** que $B$ est arrivé ».

### 4.2 Variables aléatoires et distributions

Une **variable aléatoire** associe un nombre à un résultat incertain (le résultat d'un dé, la taille d'une personne tirée au hasard). Sa **distribution** décrit comment ses valeurs se répartissent.

| Distribution | Décrit | Exemple |
| :--- | :--- | :--- |
| Bernoulli | un tirage oui/non | clic / pas de clic |
| Binomiale | nombre de succès sur $n$ tirages | 7 conversions sur 100 |
| Poisson | comptage d'événements rares | appels par heure |
| Uniforme | toutes les valeurs équiprobables | initialisation aléatoire |
| Normale (gaussienne) | valeurs groupées autour d'une moyenne | bruit de mesure, erreurs |

La loi **normale** $\mathcal{N}(\mu,\sigma^2)$ revient partout : moyenne $\mu$ (le centre), écart-type $\sigma$ (l'étalement). Environ 68 % de la masse tombe dans $[\mu-\sigma, \mu+\sigma]$ et 95 % dans $[\mu-2\sigma, \mu+2\sigma]$ — d'où la plage $[-3, +3]$ typique après standardisation (module 1).

### 4.3 Espérance et variance

L'**espérance** $\mathbb{E}[X]$ est la moyenne théorique ; la **variance** $\operatorname{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]$ mesure la dispersion, et $\sigma=\sqrt{\operatorname{Var}(X)}$ l'exprime dans l'unité d'origine.

*Exemple chiffré* : un dé équilibré. $\mathbb{E}[X]=\frac{1+2+3+4+5+6}{6}=3.5$. On ne fera jamais « 3,5 » sur un lancer : l'espérance est une **moyenne de long terme**, pas une prédiction ponctuelle.

### 4.4 Le théorème de Bayes

Il relie $P(A\mid B)$ et $P(B\mid A)$ — indispensable pour raisonner sur des tests et des diagnostics :
$$P(A\mid B)=\frac{P(B\mid A)\,P(A)}{P(B)}.$$

> 🔍 **Le piège du test médical (à méditer avant le module 2).** Une maladie touche 1 personne sur 1 000 ($P(M)=0{,}001$). Un test détecte 99 % des malades et se trompe sur 5 % des sains. Vous êtes testé positif : quelle est la probabilité d'être réellement malade ?
>
> Sur 100 000 personnes : 100 malades, dont 99 positifs ; 99 900 sains, dont $\approx 4\,995$ faux positifs. Parmi les $99+4\,995=5\,094$ positifs, seuls 99 sont malades :
> $$P(M\mid +)=\frac{99}{5\,094}\approx 1{,}9\%.$$
> Un test « à 99 % » laisse une probabilité réelle de **moins de 2 %** ! C'est exactement pourquoi le module 1 insiste : **l'accuracy seule trompe quand une classe est rare**, et pourquoi la précision et le rappel du module 2 sont nécessaires.

### 4.5 Vraisemblance et maximum de vraisemblance

La **vraisemblance** mesure à quel point un modèle de paramètres $\theta$ rend les données observées plausibles : $\mathcal{L}(\theta)=P(\text{données}\mid\theta)$. On cherche souvent le $\theta$ qui la **maximise** (maximum de vraisemblance).

Comme un produit de nombreuses probabilités devient minuscule, on maximise la **log-vraisemblance** (transformer un produit en somme, plus stable numériquement). Minimiser la **log-loss / entropie croisée** du module 2 revient exactement à maximiser la log-vraisemblance d'un modèle probabiliste — les deux points de vue coïncident.

---

## 5. Statistiques : de l'Échantillon à la Conclusion

Les probabilités partent du modèle vers les données ; les statistiques font le chemin inverse : **des données observées vers ce qu'on peut en conclure**.

### 5.1 Population, échantillon, estimation

On observe rarement toute la population ; on travaille sur un **échantillon**. Une statistique calculée dessus (moyenne, proportion) est une **estimation** de la vraie valeur, entachée d'incertitude. Le module 1 rappelle qu'un échantillon est le produit d'un mécanisme de collecte, pas « la réalité ».

### 5.2 Deux théorèmes qui justifient tout le reste

- **Loi des grands nombres** : plus l'échantillon grandit, plus la moyenne observée s'approche de l'espérance. C'est pourquoi plus de données (représentatives) aident.
- **Théorème central limite (TCL)** : la moyenne de nombreuses variables indépendantes suit approximativement une loi normale, *quelle que soit* la distribution d'origine. C'est ce qui rend les intervalles de confiance possibles.

### 5.3 Incertitude : écart-type, erreur standard, intervalle

Ne jamais rapporter une moyenne seule. L'**erreur standard** de la moyenne décroît comme $\sigma/\sqrt{n}$ : quadrupler les données divise l'incertitude par deux seulement. Un **intervalle de confiance** exprime une plage plausible ; il ne dit pas « la vraie valeur est ici à 95 % » au sens naïf, mais « une procédure comme celle-ci contient la vraie valeur 95 % du temps ».

Le module 2 applique directement ceci : rapporter *moyenne ± dispersion* d'une validation croisée, et se méfier d'un écart de 0,2 point qui n'est pas significatif.

### 5.4 Tests d'hypothèse et p-value, sans superstition

Une **p-value** est la probabilité d'observer un effet au moins aussi extrême *si l'hypothèse nulle était vraie*. Ce **n'est pas** la probabilité que l'hypothèse soit fausse, ni une mesure de la taille de l'effet.

> ⚠️ **Sur un très grand jeu, une p-value minuscule peut accompagner un effet sans importance pratique.** Rapportez toujours la **taille d'effet** et les **effectifs** (rappel du V de Cramér, module 1).

### 5.5 Corrélation : puissante mais piégeuse

La corrélation (module 1) résume une relation en un nombre. Rappels essentiels : elle mesure surtout le **linéaire** (Pearson) ou le **monotone** (Spearman), un coefficient nul n'implique pas l'indépendance, et **corrélation n'est pas causalité** (voir modules 1 et 2). Le quartet d'Anscombe illustre que des jeux très différents partagent le même coefficient : **toujours regarder le nuage de points**.

---

## 6. Optimisation : Trouver un Bon Minimum

Entraîner un modèle, c'est minimiser une fonction de perte. Quelques notions transversales :

- **Minimum global vs local** : le point le plus bas partout, versus un creux dont on ne peut sortir par petits pas. Les modèles linéaires réguliers ont souvent un paysage convexe (un seul creux) ; les réseaux profonds non.
- **Point-selle** : plat dans une direction, en pente dans une autre — fréquent en haute dimension et plus problématique que les minima locaux.
- **Taux d'apprentissage** : trop grand, on diverge (perte `NaN`) ; trop petit, on rampe (module 3).
- **Stochastique vs batch** : calculer le gradient sur un **mini-lot** (module 3) est bruité mais rapide, et le bruit aide parfois à s'échapper d'un mauvais creux.
- **Convexité** : si la fonction est convexe, tout minimum local est global — confort théorique rare en deep learning mais fréquent en ML classique régularisé.

```text
Perte
  ▲
  │\            /\
  │ \          /  \        ← minimum local (piège possible)
  │  \        /    \______
  │   \______/            \
  │   local                \____ ← minimum global
  └──────────────────────────────► paramètre
```

La régularisation (modules 1 et 2) modifie ce paysage en pénalisant la complexité, ce qui stabilise la solution.

---

## 7. Théorie de l'Information : Surprise, Entropie et Entropie Croisée

Ces notions expliquent pourquoi la log-loss est *la* perte de classification.

- **Information / surprise** d'un événement de probabilité $p$ : $-\log p$. Un événement certain ($p=1$) n'apporte aucune surprise ($-\log 1=0$) ; un événement rare en apporte beaucoup.
- **Entropie** $H(p)=-\sum_i p_i\log p_i$ : la surprise moyenne d'une distribution, donc son « incertitude ». Maximale quand tout est équiprobable.
- **Entropie croisée** $H(p, q)=-\sum_i p_i\log q_i$ : le coût moyen quand on prédit avec $q$ alors que la réalité suit $p$. C'est exactement la **`CrossEntropyLoss`** du module 3 : la cible réelle est $p$, la prédiction du modèle est $q$.
- **Divergence de Kullback–Leibler** $D_{KL}(p\,\|\,q)=\sum_i p_i\log\frac{p_i}{q_i}\ge 0$ : l'écart entre deux distributions ; nulle seulement si elles coïncident. Elle apparaît dans les VAE (module 3).
- **Information mutuelle** : réduction d'incertitude sur une variable quand on en connaît une autre ; utilisée pour la sélection de variables non linéaire (module 1).

> 💡 **Le fil rouge.** Minimiser l'entropie croisée = rendre la distribution prédite proche de la vraie = maximiser la vraisemblance. Trois vocabulaires, une seule idée.

---

## 8. Python Scientifique : NumPy, Vectorisation et Pièges Numériques

### 8.1 NumPy : le tableau qui remplace les boucles

NumPy fournit le `ndarray`, un tableau homogène et efficace. C'est la brique sous Pandas, Scikit-Learn et, conceptuellement, sous les tenseurs PyTorch.

```python
import numpy as np

X = np.array([[25, 32], [45, 65], [38, 48]], dtype=np.float64)  # (3, 2)
w = np.array([0.5, -1.0])

X.shape          # (3, 2) : toujours vérifier la forme
X.mean(axis=0)   # moyenne par colonne : [36., 48.33]
X @ w            # produit matriciel : une valeur par ligne
```

### 8.2 Vectorisation : penser en tableaux, pas en boucles

```python
# ❌ Lent : boucle Python explicite
scores = []
for i in range(len(X)):
    total = 0.0
    for j in range(X.shape[1]):
        total += X[i, j] * w[j]
    scores.append(total)

# ✅ Rapide, lisible : une seule expression vectorisée
scores = X @ w
```

La version vectorisée est souvent des dizaines à des centaines de fois plus rapide : les opérations s'exécutent en code compilé sur le tableau entier. **Réflexe à acquérir : si vous écrivez une boucle sur les lignes d'un tableau, demandez-vous s'il existe une opération vectorisée.**

### 8.3 Broadcasting : combiner des formes différentes

NumPy « étire » automatiquement les dimensions compatibles. C'est puissant et dangereux :

```python
X = np.array([[1., 2.], [3., 4.], [5., 6.]])   # (3, 2)
mu = X.mean(axis=0)                             # (2,)
X_centre = X - mu                               # (3,2) - (2,) → (3,2) : chaque ligne centrée
```

> ⚠️ **Le piège `(n,)` vs `(n, 1)`.** Un vecteur de forme `(3,)` et un de forme `(3, 1)` ne se combinent pas de la même façon : `(3,) - (1,3)` peut produire silencieusement une matrice `(3,3)` au lieu d'une erreur. Le module 3 rappelle le même piège entre `[B]` et `[B,1]`. Vérifiez `.shape`.

### 8.4 Reproductibilité et pièges numériques

- **Graine aléatoire** : `rng = np.random.default_rng(42)` rend un tirage reproductible. Fixer la graine facilite le débogage sans garantir une reproductibilité parfaite entre machines/versions (rappel modules 2 et 3).
- **Flottants** : `0.1 + 0.2 == 0.3` renvoie `False` ! Les nombres à virgule flottante sont approximés. Comparez avec `np.isclose(a, b)`, jamais `==`.
- **Stabilité numérique** : calculer un softmax ou un `log(somme d'exponentielles)` naïvement peut déborder (`inf`) ou sous-déborder (`0`). Les bibliothèques utilisent des versions stables (soustraction du max, `logsumexp`) — c'est pourquoi le module 3 recommande `BCEWithLogitsLoss` plutôt que sigmoïde suivie de log.
- **`NaN` se propage** : une seule valeur `NaN` contamine toute une somme. Traquez son origine (division par zéro, `log(0)`, données manquantes non traitées) au lieu de la masquer.

```python
def softmax_stable(z):
    z = z - z.max()          # décale sans changer le résultat, évite l'overflow
    e = np.exp(z)
    return e / e.sum()
```

### 8.5 Où va-t-on ensuite ?

- Manipuler des **tableaux étiquetés** (colonnes nommées, types mixtes) → **Pandas / Polars**, module 1.
- Enchaîner prétraitement et modèles proprement → **Scikit-Learn**, module 2.
- Passer aux **tenseurs** dérivables sur GPU → **PyTorch**, module 3.

Ces bibliothèques réutilisent toutes les concepts de ce module : formes, vectorisation, produit matriciel, gradient et probabilités.

---

## 9. Checklist et Questions de Compréhension

### Checklist des fondements

- [ ] Je sais dire la forme d'un vecteur, d'une matrice et d'un tenseur, et vérifier la compatibilité d'un produit matriciel.
- [ ] Je peux calculer un produit scalaire et l'interpréter comme une somme pondérée.
- [ ] Je sais ce qu'indique un gradient et pourquoi on avance dans le sens opposé.
- [ ] Je peux appliquer la règle de la chaîne sur une composition simple.
- [ ] Je distingue probabilité, probabilité conditionnelle et vraisemblance.
- [ ] Je sais pourquoi un test « à 99 % » peut donner moins de 2 % de vrais positifs sur une maladie rare.
- [ ] Je rapporte une incertitude (dispersion, intervalle), pas seulement une moyenne.
- [ ] Je relie entropie croisée, log-loss et maximum de vraisemblance.
- [ ] J'écris du NumPy vectorisé et je vérifie mes formes plutôt que d'empiler des boucles.
- [ ] Je compare des flottants avec `np.isclose` et je sais d'où vient un `NaN`.

### Questions de compréhension

1. Pourquoi le produit scalaire de deux vecteurs est-il « la » brique d'un neurone et d'une régression linéaire ?
2. Une similarité cosinus de 0,99 entre deux embeddings garantit-elle qu'ils ont la même norme ? Pourquoi est-ce utile en RAG ?
3. Le gradient pointe vers la montée ou la descente ? Que fait-on de ce fait pour minimiser une perte ?
4. Reprenez l'exemple du test médical avec une prévalence de 1 % : la probabilité d'être malade après un test positif augmente-t-elle beaucoup ? Justifiez.
5. Pourquoi maximiser la log-vraisemblance équivaut-il à minimiser l'entropie croisée ?
6. Une p-value de 0,001 sur un million de lignes prouve-t-elle un effet important ? Que faut-il rapporter en plus ?
7. Quelle différence entre `(n,)` et `(n, 1)` en NumPy, et pourquoi peut-elle créer un bug silencieux ?
8. Pourquoi `0.1 + 0.2 == 0.3` est-il faux, et comment tester une égalité de flottants ?

**Mini-étude de cas.** On vous donne une matrice `X` de forme `(1000, 5)` (1 000 clients, 5 caractéristiques) et un vecteur de poids `w`. (a) Quelle forme doit avoir `w` pour calculer un score par client, et quelle est la forme du résultat ? (b) Écrivez en une ligne vectorisée la standardisation (z-score) de `X` colonne par colonne. (c) Vous obtenez des `NaN` dans les scores : listez trois causes possibles et comment les diagnostiquer. (d) On vous annonce « précision moyenne 0,90 » : quelles deux informations réclamez-vous avant d'y croire ?

---

Pour approfondir et vérifier ces notions, consultez les documentations et ouvrages de référence regroupés dans [REFERENCES.md](../REFERENCES.md).
