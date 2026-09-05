# Ouverture : architectures et méthodes prometteuses, sans promesse universelle

Objectif : comprendre ce qui change par rapport aux CNN, RNN et Transformers, puis savoir concevoir une comparaison honnête. Prérequis : module 3, attention et apprentissage auto-supervisé ; les équations différentielles peuvent rester une ouverture. Prévoir 1–2 heures. Sélection de travaux fondateurs consultés le 5 septembre 2026 : ce chapitre n'est ni un classement des derniers modèles ni un état de l'art exhaustif.

« Prometteur » signifie ici qu'un mécanisme répond à une limite identifiable et possède des résultats expérimentaux publiés. Cela ne prouve pas sa supériorité sur votre tâche. Les MoE et certaines méthodes de génération sont déjà des familles établies ; leurs nouvelles variantes, hybridations et usages restent des sujets de recherche.

## 1. États structurés sélectifs : Mamba et les hybrides

Une représentation simplifiée d'un modèle à espace d'état est $h_t=\bar A_t h_{t-1}+\bar B_t x_t$, puis $y_t=C_t h_t$. Ici $x_t$ est l'entrée, $h_t$ un état mémoire de taille fixée et les matrices assurent la transition et la lecture. Mamba rend des paramètres de cette dynamique dépendants de l'entrée : le système sélectionne ce qu'il retient. Pour une taille d'état fixée, le coût peut croître linéairement avec la longueur de séquence, avec une implémentation adaptée. [Article Mamba](https://arxiv.org/abs/2312.00752).

L'intérêt est une mémoire séquentielle compacte ; le compromis est de compresser le passé, là où l'attention permet un accès explicite à des représentations précédentes. Mamba-2 étudie une relation structurée entre modèles à états et attention ; ce n'est pas l'affirmation que tous les Transformers sont équivalents à n'importe quel système récurrent. [Article Mamba-2 / SSD](https://arxiv.org/abs/2405.21060).

Expérience proposée : comparer un modèle récurrent, un Transformer et un hybride sur classification de séries puis récupération d'une information ancienne. Fixer qualité cible, matériel, longueur et taille de lot ; mesurer mémoire, latence de premier résultat et débit. Une complexité asymptotique favorable ne garantit pas une accélération à petite longueur.

## 2. Mixture of Experts : capacité conditionnelle

Un mélange d'experts parcimonieux active une partie des sous-réseaux pour chaque token. Schématiquement, $y=\sum_{i\in S(x)}g_i(x)E_i(x)$ : le routeur choisit l'ensemble $S(x)$, les $E_i$ sont les experts, et les $g_i$ leurs poids. Switch Transformers explore notamment le routage vers un expert. Cela augmente la capacité totale sans calculer tous les experts pour chaque entrée. [Article Switch Transformers](https://arxiv.org/abs/2101.03961).

Les paramètres actifs ne sont pas les paramètres totaux à stocker. Le routage pose des problèmes d'équilibrage, de capacité et de communication entre appareils. Des experts ne correspondent pas nécessairement à des compétences humaines lisibles. Surtout, **un MoE n'est pas un système multi-agent** : les experts sont des composants entraînés du réseau, pas des acteurs avec outils, messages et autorisations.

Expérience : comparer à un réseau dense à budget de calcul actif comparable, puis à budget mémoire comparable ; ce sont deux questions différentes. Contrôler la répartition des tokens entre experts et le débit réel, pas seulement la loss.

## 3. JEPA : prédire des représentations plutôt que des pixels

I-JEPA apprend à prédire la représentation de régions masquées d'une image à partir d'un contexte visible. Au lieu de reconstruire chaque pixel, un encodeur de contexte et un prédicteur visent les représentations produites par un encodeur cible. Les choix de masquage et de mise à jour de la cible sont importants. [Article I-JEPA](https://arxiv.org/abs/2301.08243).

Intuition : apprendre une information utile à la compréhension sans devoir reproduire toutes les textures. Une simple proximité entre deux encodeurs entraînés sans précautions admettrait une solution effondrée où tout devient constant ; l'objectif seul ne suffit pas. Une représentation prédictive n'est pas automatiquement un modèle causal du monde, et I-JEPA ne génère pas par lui-même des descriptions textuelles.

Évaluer avec encodeur gelé + sonde linéaire, puis fine-tuning à différents budgets de labels ; vérifier transfert hors domaine et coût du préentraînement. Une extension vidéo doit en plus tester la compréhension temporelle, pas seulement le décor.

## 4. Flow matching : apprendre un champ de vitesse

Il s'agit d'une méthode de modélisation générative, pas d'une unique architecture. Dans un chemin d'interpolation simple, $x_0$ est un bruit, $x_1$ une donnée et $x_t=(1-t)x_0+tx_1$. Un réseau apprend $v_\theta(x_t,t)$ en visant la vitesse $x_1-x_0$ pour les paires échantillonnées. À l'inférence, un solveur intègre $dx/dt=v_\theta(x,t)$ de 0 vers 1. Le champ optimal représente une moyenne conditionnelle des vitesses, pas la connaissance d'une cible unique pour chaque bruit. [Article Flow Matching](https://arxiv.org/abs/2210.02747).

Le chemin, le couplage bruit–donnée et le solveur influencent le résultat. Le temps est ici orienté **bruit vers donnée**, contrairement à l'indice de bruitage direct du [TP DDPM](08_diffusion_2d.ipynb). Ne pas mélanger leurs équations. Comparer nombre d'évaluations du réseau, coût réel et couverture des modes ; peu de pas ne garantit pas une génération correcte.

## 5. KAN : apprendre des fonctions sur les connexions

Dans une couche KAN simplifiée, $y_j=\sum_i\phi_{ji}(x_i)$, où les fonctions scalaires $\phi_{ji}$ sont apprises, par exemple via des splines. Un MLP usuel combine plutôt poids scalaires et activations fixées. Les travaux KAN explorent notamment l'approximation de fonctions et des problèmes scientifiques. [Article KAN](https://arxiv.org/abs/2404.19756).

Le théorème de représentation qui motive la construction ne garantit pas une optimisation facile, une bonne extrapolation ou un avantage sur des images naturelles. La visualisation des fonctions peut aider l'analyse, sans constituer une preuve d'interprétabilité globale. Comparer à MLP, splines et modèles spécialisés, en tenant compte des degrés de liberté, du temps et de la sensibilité au bruit.

## 6. Lire un résultat de recherche avant de l'adopter

| Question | Preuve attendue |
|---|---|
| Quel goulot d'étranglement est visé ? | mémoire, longueur, labels, calcul ou structure de la tâche explicités |
| Le gain vient-il vraiment du mécanisme ? | ablation et contrôle des données, du préentraînement et du réglage |
| À budget équitable ? | paramètres totaux/actifs, calcul train, latence, mémoire, énergie si mesurable |
| La comparaison généralise-t-elle ? | plusieurs graines, hors distribution, modalités absentes, incertitude |
| Peut-on reproduire ? | version du code et des poids, dépendances, protocole, accès/licences |
| Peut-on exploiter ? | maturité des kernels, surveillance, risques, solution de repli |

Exercice corrigé : « Le modèle A a dix fois plus de paramètres mais deux fois moins de paramètres actifs ; il est donc plus léger. » Réponse : indécidable sans préciser léger en calcul, mémoire de stockage, mémoire à l'exécution ou communication. Mesurer chaque coût dans le contexte de déploiement.

Prolongement de projet : choisir **une** famille et formuler une hypothèse réfutable, une baseline, un budget et un critère d'abandon. L'objectif pédagogique est une expérience interprétable, pas une collection de noms d'architectures.
