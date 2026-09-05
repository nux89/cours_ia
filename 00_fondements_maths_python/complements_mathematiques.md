# Calculs guidés : des notations aux modèles

Ce complément du [module 0](cours_fondements_maths_python.md) se lit avant les formules des modules 2, 3 et 6, puis se consulte au besoin. Objectif : savoir refaire un calcul, annoncer ses dimensions et vérifier ses hypothèses. Prévoir 2–3 heures, exercices compris. Il ne remplace pas un cours complet d'analyse ou de probabilités.

## 1. Lire les symboles et suivre les formes

| Symbole | Sens et convention locale |
|---|---|
| $n,d,k$ | nombre d'exemples, de caractéristiques, de sorties |
| $X\in\mathbb R^{n\times d}$ | exemples en lignes, caractéristiques en colonnes |
| $w\in\mathbb R^d$, $b\in\mathbb R$ | poids et biais d'une sortie scalaire |
| $\hat y=Xw+b$ | prédictions ; $b$ est ajouté à chaque ligne |
| $\theta$, $\eta$ | ensemble des paramètres, taux d'apprentissage |
| $\sum_i$, $\prod_i$ | somme, produit sur l'indice indiqué |
| $\partial L/\partial w_j$, $\nabla_w L$ | dérivée partielle, vecteur des dérivées |
| $\mathbb E$, $\operatorname{Var}$ | espérance, variance ; préciser la variable aléatoire concernée |
| $\arg\min_\theta L$ | paramètre minimisant $L$, distinct de la valeur $\min L$ |
| $\odot$, $A^T$ | produit élément par élément, transposée |

Exemple : $X=\begin{pmatrix}1&2\\3&4\end{pmatrix}$ et $w=(2,-1)^T$. Alors $Xw=(0,2)^T$ ; avec $b=1$, les prédictions sont $(1,3)^T$. En NumPy, `X @ w` effectue ce produit ; `X * w` multiplie les colonnes par diffusion (*broadcasting*) et conserve une matrice. Les deux opérations sont valides mais ne répondent pas à la même question.

Exercice : pour $X$ de forme `[8,3]` et $W$ de forme `[3,5]`, quelle forme a $XW$ ? Correction : `[8,5]`, avec $(XW)_{ij}=\sum_{r=1}^3 X_{ir}W_{rj}$.

## 2. Dériver une perte et vérifier un pas

Pour une régression scalaire, $L(w)=\frac1n\sum_i(x_iw-y_i)^2$. La règle de la chaîne donne :

$$\frac{dL}{dw}=\frac2n\sum_i x_i(x_iw-y_i).$$

Prenons $x=(1,2)$, $y=(2,4)$ et $w=1$. Les résidus valent $(-1,-2)$, $L=2,5$ et le gradient vaut $-5$. Avec $\eta=0,1$, $w'=1-0,1(-5)=1,5$, et $L(w')=0,625$. Le gradient négatif conduit bien à augmenter le poids. Avec $\eta=1$, le poids devient 6 et la perte 40 : suivre la direction de descente avec un pas arbitrairement grand ne suffit pas.

Version matricielle, avec résidu $r=Xw+b-y$ :

$$\nabla_w L=\frac2nX^Tr,\qquad \frac{\partial L}{\partial b}=\frac2n\sum_i r_i.$$

Les dimensions sont `[d,n] @ [n] → [d]`. Une pénalité $\lambda\|w\|^2$ ajoute $2\lambda w$ ; avec $\lambda\|w\|^2/2$, elle ajoute $\lambda w$. Toujours lire la convention de la perte avant de comparer deux implémentations.

Contrôle numérique : $(L(w+h)-L(w-h))/(2h)$ approche la dérivée pour un petit $h$, par exemple $10^{-5}$ en double précision. Trop petit, $h$ amplifie les erreurs d'arrondi. Aux points non dérivables comme ReLU en zéro, cette vérification demande une convention particulière.

## 3. Softmax, entropie croisée et gradient

Pour des logits $z=(0,\ln2,0)$, les exponentielles valent $(1,2,1)$ : les probabilités softmax sont $p=(0,25;0,5;0,25)$. Si la deuxième classe est correcte, la cible one-hot vaut $y=(0,1,0)$ et :

$$L=-\sum_j y_j\ln p_j=-\ln0,5\simeq0,6931,\qquad \frac{\partial L}{\partial z_j}=p_j-y_j.$$

Le gradient est donc $(0,25;-0,5;0,25)$ : la descente augmente le score correct et diminue les autres. Cette formule vaut pour une cible de somme 1, sans pondération supplémentaire, et une observation ; la moyenne du lot ajoute son facteur $1/n$.

Pour éviter le débordement, calculer softmax sur $z-\max(z)$ : cela ne change pas le résultat. `torch.nn.CrossEntropyLoss` reçoit des **logits**, pas un softmax déjà appliqué. Une petite perte ne prouve pas la calibration ni l'absence de biais.

L'entropie $H(p)=-\sum p_j\ln p_j$ mesure l'incertitude d'une distribution sur un support fini fixé ; elle est maximale à l'uniforme. La divergence $D_{KL}(p\|q)=\sum p_j\ln(p_j/q_j)$ n'est pas une distance métrique : elle n'est pas symétrique. Elle est infinie si $p_j>0$ mais $q_j=0$ pour une classe. Utiliser la convention $0\ln0=0$ par passage à la limite.

## 4. Covariance, corrélation et PCA avec trois observations

Prenons $x=(1,2,3)$ et $y=(2,4,6)$. Les moyennes sont 2 et 4. Les versions centrées sont $(-1,0,1)$ et $(-2,0,2)$. Avec le diviseur d'échantillon $n-1=2$ :

$$s_x^2=1,\quad s_y^2=4,\quad s_{xy}=2,\quad r=\frac{s_{xy}}{s_xs_y}=1.$$

La matrice de covariance est $C=\begin{pmatrix}1&2\\2&4\end{pmatrix}$. Ses valeurs propres sont 5 et 0. La première direction propre est $(1,2)^T/\sqrt5$ : toutes les observations centrées sont sur cette droite. Projeter dessus conserve ici toute la variance. Pour la PCA d'une table réelle, centrer et éventuellement standardiser sur le train, puis transformer validation et test avec ces paramètres.

La covariance a des unités composées ; la corrélation est sans unité. Si une colonne est constante, son écart-type est nul et Pearson est indéfini. Avec $x=(-1,0,1)$ et $y=x^2$, la covariance est nulle malgré une dépendance parfaite : une heatmap ne suffit pas à sélectionner les variables. Visualiser aussi les nuages, distributions par groupe et relations non linéaires ; vérifier que chaque variable existe à l'instant de prédiction.

## 5. Ce que dit réellement biais–variance

En régression à perte quadratique, fixons une entrée $x$, $Y=f(x)+\varepsilon$ avec $\mathbb E[\varepsilon\mid x]=0$ et variance $\sigma^2(x)$. Le jeu d'apprentissage aléatoire $D$ est indépendant du bruit de test. Alors :

$$\mathbb E_{D,\varepsilon}[(Y-\hat f_D(x))^2]=\sigma^2(x)+\left(f(x)-\mathbb E_D[\hat f_D(x)]\right)^2+\operatorname{Var}_D[\hat f_D(x)].$$

Le biais est l'écart de la prédiction moyenne à la vraie fonction ; la variance mesure la sensibilité à un nouveau jeu d'entraînement. Ce n'est pas la variance des entrées ni une identité générale pour l'accuracy en classification. La courbe en U est une intuition, pas une loi universelle imposant une forme à toute expérience.

## 6. Attention : les dimensions avant la formule

Avec $m$ requêtes et $n$ clés, $Q$ a la forme $[m,d_k]$, $K$ `[n,d_k]` et $V$ `[n,d_v]` :

$$A=\operatorname{softmax}_{\text{clés}}(QK^T/\sqrt{d_k}+M),\qquad O=AV.$$

Les scores et le masque $M$ sont `[m,n]`, la sortie `[m,d_v]`. Les entrées interdites reçoivent conceptuellement $-\infty$ avant softmax ; ne jamais laisser une ligne entièrement masquée sans traitement explicite. En self-attention, requêtes, clés et valeurs proviennent d'une même séquence ; en cross-attention, les requêtes peuvent être textuelles et les clés/valeurs visuelles. Elles n'ont pas besoin du même nombre de tokens.

Exemple à $d_k=1$ : $Q=(1)$, $K=(0,\ln3)^T$, $V=(2,10)^T$. Les poids sont $(1/4,3/4)$ et la sortie vaut 8. En masquant la deuxième clé, la sortie devient 2. Sous l'approximation de composantes indépendantes, centrées et de variance 1, le produit scalaire a une variance $d_k$ ; le diviseur $\sqrt{d_k}$ contrôle cette échelle. Ce n'est pas une hypothèse toujours exacte après apprentissage.

## 7. Adam sans métaphore trompeuse

À l'itération $t\ge1$, avec gradient $g_t$, moments initialisés à zéro et opérations élément par élément :

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,$$
$$\hat m_t=m_t/(1-\beta_1^t),\quad \hat v_t=v_t/(1-\beta_2^t),\quad \theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

Les $\beta$ règlent la mémoire des moyennes exponentielles ; $\epsilon>0$ stabilise le dénominateur. Le premier pas avec gradient scalaire 2 a $\hat m_1=2$, $\hat v_1=4$, donc une correction proche de $\eta$, et non $2\eta$. AdamW ajoute une décroissance des poids découplée. Aucun de ces mécanismes ne garantit la généralisation.

## 8. Savoir ce qui reste à approfondir

Ce support explicite les calculs utilisés ici, sans démontrer tous les théorèmes. Pour aller plus loin : Jacobiennes et Hessiennes, conditionnement et SVD, convergence stochastique, inférence sous dépendance, théorie de l'information et équations différentielles. Refaire les exemples dans le [TP NumPy](01_maths_numpy_pratique.ipynb), puis relier la règle de chaîne au module 3 et l'espérance conditionnelle aux équations de Bellman du module 6.
