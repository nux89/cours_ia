# Module 6 : L'Apprentissage par Renforcement (Reinforcement Learning)

> Là où l'apprentissage supervisé reçoit la bonne réponse à chaque exemple, l'apprentissage par renforcement (RL) n'a qu'un **signal de récompense** différé et partiel. L'agent doit **découvrir par essais et erreurs** quelles suites d'actions maximisent une récompense cumulée. C'est le troisième grand paradigme annoncé dans la taxonomie du module 2, et le mécanisme derrière l'alignement des LLMs vu au module 5.

**Objectifs du module.** À l'issue de ce chapitre, vous saurez formaliser un problème de décision séquentielle en processus de décision markovien (MDP), distinguer politique, valeur et récompense, comprendre le dilemme exploration/exploitation, expliquer Q-learning et gradient de politique, situer le Deep RL et le RLHF, et reconnaître les pièges de spécification de récompense et de sécurité avant tout déploiement.

**Prérequis.** Modules 0 (probabilités, espérance, gradient), 2 (généralisation, validation) et 3 (réseaux, descente de gradient). Le module 5 relie ce chapitre à l'alignement des modèles de langage ; le module 4 en discute l'usage prudent dans les agents.

---

## 📖 Le Dico du Débutant (Jargon Buster)
- **Agent** : l'entité qui décide et agit (un robot, un joueur, un système de recommandation).
- **Environnement** : le monde avec lequel l'agent interagit et qui renvoie des observations et des récompenses.
- **État (*state*)** : la description de la situation à un instant donné.
- **Action** : ce que l'agent peut faire dans un état.
- **Récompense (*reward*)** : un nombre renvoyé après une action, qui dit si c'était bon (+) ou mauvais (−).
- **Politique (*policy* $\pi$)** : la stratégie de l'agent — quelle action choisir dans chaque état.
- **Retour (*return*)** : la somme (souvent actualisée) des récompenses futures ; c'est **lui** qu'on cherche à maximiser, pas la récompense immédiate.
- **Épisode** : une trajectoire complète, du début à un état terminal (une partie, une mission).

---

## Table des Matières
1. [Le Paradigme : Apprendre par Essais et Erreurs](#1-le-paradigme--apprendre-par-essais-et-erreurs)
2. [Formalisation : le Processus de Décision Markovien (MDP)](#2-formalisation--le-processus-de-décision-markovien-mdp)
3. [Politiques, Valeurs et Équations de Bellman](#3-politiques-valeurs-et-équations-de-bellman)
4. [Exploration vs Exploitation et les Bandits](#4-exploration-vs-exploitation-et-les-bandits)
5. [Méthodes sans Modèle : Monte Carlo, TD et Q-Learning](#5-méthodes-sans-modèle--monte-carlo-td-et-q-learning)
6. [Deep RL : Quand les Réseaux Approximent Valeurs et Politiques](#6-deep-rl--quand-les-réseaux-approximent-valeurs-et-politiques)
7. [RLHF : Aligner un Modèle par les Préférences Humaines](#7-rlhf--aligner-un-modèle-par-les-préférences-humaines)
8. [Spécification de Récompense, Sécurité et Évaluation](#8-spécification-de-récompense-sécurité-et-évaluation)
9. [Checklist et Questions de Compréhension](#9-checklist-et-questions-de-compréhension)

---

## 1. Le Paradigme : Apprendre par Essais et Erreurs

### 🐕 L'Analogie du Dressage (revisitée)

Le module 2 comparait le RL au dressage d'un chiot. Précisons la boucle. À chaque instant $t$ :

```text
   ┌─────────────────────────────────────────────┐
   │                  AGENT                        │
   │        choisit une action a_t = π(s_t)        │
   └───────────────┬───────────────▲──────────────┘
                   │ action a_t     │ état s_{t+1}, récompense r_{t+1}
                   ▼                │
   ┌─────────────────────────────────────────────┐
   │               ENVIRONNEMENT                   │
   └─────────────────────────────────────────────┘
```

L'agent observe un état, agit, reçoit une récompense et un nouvel état, et recommence. Trois différences fondamentales avec le supervisé :

1. **Pas de corrigé** : personne ne dit « la bonne action était X ». Seule une récompense, souvent **différée** (on ne sait qu'à la fin qu'on a gagné).
2. **Les données dépendent des actions** : la politique influence ce que l'agent verra ensuite (contrairement à un jeu de données fixe). C'est une boucle de rétroaction (module 1).
3. **Objectif de long terme** : maximiser le **retour** cumulé, pas la récompense immédiate — accepter une perte maintenant pour un gain plus tard (sacrifier un pion aux échecs).

> 💡 **Le problème du crédit temporel (*credit assignment*).** Si l'on gagne au bout de 100 coups, quels coups méritent le crédit ? Attribuer la récompense finale aux bonnes décisions passées est le cœur technique du RL.

---

## 2. Formalisation : le Processus de Décision Markovien (MDP)

Un **MDP** est le cadre standard du RL. Il est défini par $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ :

- $\mathcal{S}$ : l'ensemble des **états** ;
- $\mathcal{A}$ : l'ensemble des **actions** ;
- $P(s'\mid s, a)$ : la **dynamique** — probabilité d'arriver en $s'$ en faisant $a$ depuis $s$ ;
- $R(s, a)$ : la **récompense** attendue ;
- $\gamma\in[0, 1]$ : le **facteur d'actualisation**.

**Propriété de Markov** : le futur ne dépend que de l'**état présent**, pas de tout l'historique. Bien choisir l'état pour qu'il résume l'information utile est une décision de conception majeure (un seul instantané d'un jeu ne dit pas la vitesse d'une balle — d'où l'empilement d'images en Deep RL).

### 2.1 Le retour et l'actualisation

Le **retour** à partir de l'instant $t$ est la somme actualisée des récompenses futures :
$$G_t=r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\cdots=\sum_{k=0}^{\infty}\gamma^k r_{t+k+1}.$$

Le facteur $\gamma$ règle l'**horizon** :
- $\gamma\to 0$ : agent « myope », ne regarde que l'immédiat ;
- $\gamma\to 1$ : agent « prévoyant », valorise le long terme ;
- $\gamma<1$ garantit une somme absolument finie si les récompenses sont bornées : $|G_t|\le R_{\max}/(1-\gamma)$. Pour un épisode fini, $\gamma=1$ est possible.

*Exemple chiffré* : récompenses $r=[0, 0, 10]$ sur trois pas, $\gamma=0.9$. Le retour initial vaut $0+0.9\cdot 0+0.9^2\cdot 10=8.1$. La même récompense obtenue plus tôt vaudrait davantage : l'actualisation encode « un gain proche vaut mieux qu'un gain lointain ».

---

## 3. Politiques, Valeurs et Équations de Bellman

### 3.1 La politique

Une **politique** $\pi(a\mid s)$ dit quelle action prendre (ou avec quelle probabilité) dans chaque état. Elle peut être **déterministe** ($a=\pi(s)$) ou **stochastique**. Le but du RL : trouver une politique **optimale** $\pi^\*$ qui maximise le retour attendu.

### 3.2 Fonctions de valeur

Deux fonctions résument « à quel point une situation est bonne » sous une politique $\pi$ :

- **Valeur d'état** $V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]$ : le retour attendu si l'on part de $s$ et suit $\pi$.
- **Valeur d'action (Q)** $Q^\pi(s, a)=\mathbb{E}_\pi[G_t\mid s_t=s, a_t=a]$ : le retour attendu si l'on fait $a$ en $s$ puis suit $\pi$.

$Q$ est très pratique : choisir une action maximisant **$Q^*$**, la valeur optimale, définit une politique optimale. Être glouton selon une estimation quelconque de $Q$ ou selon $Q^\pi$ ne l'assure pas.

### 3.3 Les équations de Bellman

Elles expriment une idée récursive simple : **la valeur d'un état = récompense immédiate + valeur actualisée de la suite**.

$$Q^\*(s, a)=\mathbb{E}\!\left[r+\gamma\max_{a'}Q^\*(s', a')\;\middle|\;s, a\right].$$

Cette cohérence entre valeur présente et valeur future est le levier de **presque tous** les algorithmes : on ajuste les estimations pour qu'elles respectent Bellman.

---

## 4. Exploration vs Exploitation et les Bandits

### 🎰 L'Analogie des Machines à Sous

Face à plusieurs machines à sous (*bandits*) aux gains inconnus :
- **Exploiter** : rejouer la machine qui a le mieux payé jusqu'ici.
- **Explorer** : essayer une machine peu testée qui pourrait être meilleure.

Trop exploiter fige sur un choix médiocre ; trop explorer gaspille. Ce **dilemme exploration/exploitation** est propre au RL (absent du supervisé). Stratégies courantes :

- **ε-greedy** : agir au mieux avec probabilité $1-\varepsilon$, au hasard avec probabilité $\varepsilon$ (souvent décroissante).
- **Optimisme / bornes de confiance (UCB)** : préférer les options incertaines mais potentiellement bonnes.
- **Échantillonnage de Thompson** : tirer selon la probabilité d'être la meilleure.

Le **bandit** (un seul état, ou contextuel) est le RL minimal : il éclaire par exemple l'optimisation d'annonces ou de recommandations, où chaque choix influence les données futures (biais d'exposition, module 2, §10.9).

---

## 5. Méthodes sans Modèle : Monte Carlo, TD et Q-Learning

Quand la dynamique $P$ est **inconnue** (le cas usuel), on apprend directement de l'expérience — c'est le RL **sans modèle** (*model-free*).

### 5.1 Monte Carlo

Jouer des **épisodes complets**, puis mettre à jour les valeurs avec le retour observé. Pour une politique fixée et un retour intégrable, ce retour constitue une cible non biaisée de sa valeur conditionnelle, sans approximation par une valeur future (*bootstrap*). Il faut attendre la fin de l'épisode et la variance peut être élevée ; une politique changeante ou un estimateur hors politique demande une analyse supplémentaire.

### 5.2 Différence temporelle (TD)

Mettre à jour **à chaque pas** à partir d'une estimation de la suite, sans attendre la fin (*bootstrapping*). La mise à jour TD(0) de la valeur :
$$V(s_t)\leftarrow V(s_t)+\alpha\big[\underbrace{r_{t+1}+\gamma V(s_{t+1})}_{\text{cible TD}}-V(s_t)\big].$$
Le terme entre crochets est l'**erreur TD** : l'écart entre ce qu'on attendait et ce qu'on observe + estime. C'est un compromis biais/variance entre Monte Carlo et une estimation pure.

### 5.3 Q-Learning : apprendre à agir sans modèle

Le **Q-learning** vise $Q^\*$ par la mise à jour :
$$Q(s, a)\leftarrow Q(s, a)+\alpha\big[r+\gamma\max_{a'}Q(s', a')-Q(s, a)\big].$$

C'est une méthode **hors politique** (*off-policy*) : elle apprend la valeur de la meilleure action même en explorant autrement (ε-greedy). Sa cousine **SARSA** est **sur politique** (*on-policy*) : elle évalue la politique réellement suivie, exploration comprise, ce qui la rend souvent plus prudente près des dangers.

*Exemple chiffré d'un pas.* $Q(s,a)=2$, récompense $r=1$, $\gamma=0.9$, meilleure valeur suivante $\max_{a'}Q(s',a')=5$, taux $\alpha=0.1$. Cible = $1+0.9\times 5=5.5$ ; erreur = $5.5-2=3.5$ ; mise à jour : $Q\leftarrow 2+0.1\times 3.5=2.35$. La valeur monte vers la cible, sans la rejoindre d'un coup.

Pour un état terminal, la cible est $r$, sans terme de valeur future. La convergence tabulaire classique demande notamment un MDP fini à récompenses bornées, une exploration suffisante de chaque paire état–action et des pas vérifiant $\sum_t\alpha_t=\infty$ et $\sum_t\alpha_t^2<\infty$ par paire, avec actualisation stricte ou conditions épisodiques adaptées. Un taux constant et un nombre fini d'épisodes ne constituent pas une preuve de convergence. Ces garanties ne se transfèrent pas automatiquement à un réseau profond.

### 5.4 Tabulaire vs réaliste

Ces méthodes « tabulaires » supposent qu'on peut stocker une valeur par état (ou par paire état–action). Impossible dès que les états sont nombreux ou continus (pixels d'un jeu, capteurs d'un robot). Il faut **approximer** — d'où le Deep RL.

---

## 6. Deep RL : Quand les Réseaux Approximent Valeurs et Politiques

On remplace la table par un **réseau de neurones** (module 3) qui généralise entre états semblables.

### 6.1 Méthodes par valeur : DQN

Le **Deep Q-Network (DQN)** approxime $Q(s, a)$ par un réseau. Il a rendu possible l'apprentissage à partir de pixels, grâce à deux stabilisateurs clés :

- **experience replay** : mémoriser des transitions et les rejouer en lots aléatoires, ce qui décorrèle les données (les transitions consécutives sont très corrélées) ;
- **réseau cible** : une copie figée périodiquement pour calculer la cible, évitant une cible qui bouge sans cesse.

> ⚠️ **Instabilité.** Combiner approximation par fonction, bootstrapping et apprentissage hors politique (la « triade mortelle ») peut faire diverger l'entraînement. Le RL est **notoirement sensible** aux graines, hyperparamètres et détails d'implémentation : rapportez plusieurs graines et un intervalle (rappel modules 2 et 3).

### 6.2 Méthodes par politique : gradient de politique

Plutôt qu'estimer des valeurs, on **paramètre directement la politique** $\pi_\theta(a\mid s)$ et on monte le long du gradient du retour attendu (**REINFORCE**) :
$$\nabla_\theta J(\theta)=\mathbb{E}_\pi\big[\nabla_\theta\log\pi_\theta(a\mid s)\,G_t\big].$$
Intuition : augmenter la probabilité des actions suivies de bons retours, diminuer celle des mauvaises. Ces méthodes gèrent naturellement les **actions continues** (couple d'un moteur, dose) et les politiques stochastiques, mais souffrent d'une forte variance.

### 6.3 Acteur-critique

Combiner les deux : un **acteur** (politique) agit, un **critique** (valeur) juge, réduisant la variance du gradient. Des algorithmes répandus (A2C/A3C, PPO, SAC, DDPG) déclinent cette idée. **PPO** est particulièrement utilisé, y compris pour le RLHF (§7), car il limite l'ampleur des mises à jour pour rester stable.

| Famille | Apprend | Actions | Atout | Vigilance |
| :--- | :--- | :--- | :--- | :--- |
| Par valeur (Q, DQN) | $Q(s,a)$ | discrètes surtout | efficace en échantillons | instable, actions continues difficiles |
| Par politique (REINFORCE) | $\pi_\theta$ | discrètes/continues | politiques stochastiques | variance élevée |
| Acteur-critique (PPO, SAC) | les deux | discrètes/continues | compromis stabilité/flexibilité | plus d'hyperparamètres |

---

## 7. RLHF : Aligner un Modèle par les Préférences Humaines

Le RL ne sert pas qu'aux jeux et à la robotique : il **aligne** les modèles de langage (module 5, §6.2).

Le problème : « une bonne réponse » n'a pas de récompense numérique évidente. Le **RLHF (Reinforcement Learning from Human Feedback)** contourne cela :

```text
1. Collecter des comparaisons humaines : "réponse A meilleure que réponse B".
2. Entraîner un modèle de récompense qui prédit la préférence humaine.
3. Optimiser le LLM (souvent par PPO) pour maximiser cette récompense,
   avec une pénalité l'empêchant de trop s'éloigner du modèle initial.
```

La pénalité (souvent une divergence KL, module 0) évite que le modèle « triche » en dérivant vers un langage dégénéré qui plaît au modèle de récompense mais pas aux humains.

> 💡 **Alternatives et nuances.** Des méthodes plus directes d'optimisation de préférences (sans boucle RL explicite) existent et sont largement utilisées. Dans tous les cas, l'alignement oriente le comportement **sans** garantir la vérité, l'équité ou l'absence de biais : les préférences des annotateurs et l'imperfection du modèle de récompense se propagent au modèle final. C'est le lien direct avec le module 8 (éthique).

---

## 8. Spécification de Récompense, Sécurité et Évaluation

### 8.1 Le piège de la récompense mal spécifiée (*reward hacking*)

Un agent optimise **exactement** ce que vous récompensez, pas ce que vous vouliez (rappel module 2, §10.11).

> 🔍 **Exemples classiques.** Un bateau de course virtuel récompensé sur des points intermédiaires tourne en rond pour collecter des bonus au lieu de finir la course. Un robot récompensé pour « tenir un objet en hauteur » apprend à se placer devant la caméra plutôt qu'à soulever. Un agent de nettoyage récompensé pour « ne plus voir de saleté » apprend à fermer les yeux.

La récompense est une **spécification** : toute faille sera exploitée. Bonnes pratiques :
- récompenser le **résultat voulu**, pas un proxy facile à contourner ;
- ajouter des **pénalités** pour comportements indésirables et effets de bord ;
- tester la robustesse de la récompense **avant** de passer à l'échelle.

### 8.2 Sécurité et exploration risquée

Explorer par essais et erreurs est dangereux dans le monde réel (un robot qui « essaie » de tomber, un traitement médical). Approches :
- **simulation** d'abord, puis transfert prudent (*sim-to-real*), en gardant à l'esprit l'écart de réalité ;
- **RL hors ligne (*offline RL*)** : apprendre d'un journal de données déjà collecté, sans interagir — utile quand l'exploration en ligne est trop risquée, mais limité par la couverture des données (rappel module 1) ;
- **contraintes de sécurité** et **supervision humaine** pour borner les actions (module 4, §7).

### 8.3 Évaluer un agent RL

- comparer à des **baselines** : politique aléatoire, règle métier, comportement humain (rappel module 2, §7) ;
- rapporter **plusieurs graines**, moyenne **et** dispersion : le RL est très variable ;
- évaluer sur des **conditions non vues** (généralisation), pas seulement l'environnement d'entraînement ;
- distinguer **récompense optimisée** et **objectif réel** : un score élevé peut cacher un *reward hacking*.

> ⚠️ **Quand (ne pas) utiliser le RL.** Le RL brille sur les problèmes **séquentiels** où les décisions influencent les états futurs et où l'on dispose d'un simulateur ou de nombreuses interactions bon marché. Si un modèle supervisé, un bandit ou une simple règle suffit, préférez-les : ils sont plus stables, plus faciles à valider et moins coûteux (même esprit de sobriété qu'au module 4).

---

## 9. Checklist et Questions de Compréhension

### Checklist d'un projet RL

- [ ] Le problème est réellement séquentiel (les actions changent les états futurs).
- [ ] L'état respecte (approximativement) la propriété de Markov et résume l'information utile.
- [ ] La récompense encode l'objectif réel, pas un proxy facile à détourner.
- [ ] Le facteur $\gamma$ et l'horizon correspondent à la décision visée.
- [ ] Exploration et exploitation sont explicitement gérées.
- [ ] L'entraînement est répété sur plusieurs graines, avec dispersion rapportée.
- [ ] L'exploration risquée est simulée ou bornée ; l'humain garde le contrôle des actions sensibles.
- [ ] L'évaluation compare à une baseline et teste des conditions non vues.
- [ ] Le *reward hacking* a été recherché avant tout passage à l'échelle.

### Questions de compréhension

1. Quelles sont les trois différences majeures entre RL et apprentissage supervisé ?
2. À quoi sert le facteur d'actualisation $\gamma$, et que se passe-t-il quand il tend vers 0 ou vers 1 ?
3. Différence entre $V(s)$ et $Q(s, a)$ ; pourquoi $Q$ facilite-t-elle le choix d'une action ?
4. Expliquez le dilemme exploration/exploitation et une stratégie pour l'arbitrer.
5. En quoi le Q-learning est-il « hors politique » et SARSA « sur politique » ?
6. Pourquoi DQN a-t-il besoin d'un *experience replay* et d'un réseau cible ?
7. Quand préférer une méthode par politique à une méthode par valeur ?
8. Décrivez les trois étapes du RLHF et le rôle de la pénalité KL.
9. Donnez un exemple de *reward hacking* et une parade.

**Mini-étude de cas.** Vous concevez un agent qui gère le chauffage d'un bâtiment pour minimiser la facture tout en gardant le confort. (a) Définissez états, actions, récompense (avec pénalités) et $\gamma$. (b) Choisissez une famille d'algorithmes et justifiez. (c) Identifiez un risque de *reward hacking* et comment l'empêcher. (d) Expliquez pourquoi vous entraînez d'abord en simulation et comment vous évaluez avant tout déploiement réel.

---

Pour approfondir et vérifier ces notions, consultez les articles fondateurs et documentations regroupés dans [REFERENCES.md](../REFERENCES.md).
