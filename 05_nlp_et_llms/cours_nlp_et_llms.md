# Module 5 : Traitement du Langage Naturel (NLP) et Grands Modèles de Langage

> **Étape 6/11 du parcours** · [← Précédent](../03_deep_learning/cours_deep_learning.md) · [Progression et ordre des sections](../PARCOURS_ET_EVALUATION.md) · [Suivant →](../10_methodes_specialisees/cours_methodes_specialisees.md)

> Le langage est la donnée la plus abondante et la plus ambiguë. Une même phrase peut être ironique, contextuelle ou dépendante du monde extérieur. Ce module relie les représentations classiques du texte (sacs de mots, TF-IDF, embeddings) aux modèles séquentiels et aux Transformers du module 3, puis aux grands modèles de langage (LLMs) qui alimentent l'IA agentique du module 4.

**Objectifs du module.** À l'issue de ce chapitre, vous saurez préparer du texte (normalisation, tokenisation), le représenter (one-hot, TF-IDF, embeddings), choisir une baseline solide, situer les grandes tâches du NLP, expliquer le passage du seq2seq au Transformer pour le langage, décrire ce qu'un LLM apprend réellement en préentraînement, piloter la génération (décodage, prompting, in-context learning), et évaluer un système de langage en tenant compte de l'hallucination, du coût et des biais.

**Prérequis.** Modules 1 à 3 (surtout attention et Transformers), et notions du module 0 (produit scalaire, similarité cosinus, entropie croisée). Le module 4 prolonge ce chapitre avec le RAG, les outils et la sécurité.

---

## 📖 Le Dico du Débutant (Jargon Buster)
- **Token** : l'unité de base manipulée par le modèle — un mot, un sous-mot (`re`, `##chercher`) ou un caractère. Le texte est d'abord découpé en tokens.
- **Vocabulaire** : l'ensemble fini des tokens connus, chacun associé à un identifiant entier.
- **Embedding** : un vecteur de nombres qui représente un token ou un texte, tel que la proximité géométrique reflète une proximité de sens.
- **Corpus** : un grand ensemble de textes utilisé pour entraîner ou évaluer un modèle.
- **Modèle de langage (*LM*)** : un modèle qui attribue des probabilités à des suites de tokens, donc sait prédire le token suivant.
- **LLM (*Large Language Model*)** : un modèle de langage à grand nombre de paramètres, préentraîné sur de vastes corpus, souvent fondé sur un Transformer décodeur.
- **Prompt** : le texte d'entrée fourni au modèle, qui conditionne sa sortie.
- **Hallucination** : une sortie fluide et plausible mais fausse ou non fondée sur une source.

---

## Table des Matières
1. [Pourquoi le Langage est Difficile](#1-pourquoi-le-langage-est-difficile)
2. [Préparer le Texte : Normalisation et Tokenisation](#2-préparer-le-texte--normalisation-et-tokenisation)
3. [Représenter le Texte : du Sac de Mots aux Embeddings](#3-représenter-le-texte--du-sac-de-mots-aux-embeddings)
4. [Les Grandes Tâches du NLP](#4-les-grandes-tâches-du-nlp)
5. [Des RNN aux Transformers pour le Langage](#5-des-rnn-aux-transformers-pour-le-langage)
6. [Les Grands Modèles de Langage (LLMs)](#6-les-grands-modèles-de-langage-llms)
7. [Piloter un LLM : Prompting et In-Context Learning](#7-piloter-un-llm--prompting-et-in-context-learning)
8. [Décodage : Comment le Texte est Généré](#8-décodage--comment-le-texte-est-généré)
9. [Évaluation, Hallucination et Limites](#9-évaluation-hallucination-et-limites)
10. [Checklist et Questions de Compréhension](#10-checklist-et-questions-de-compréhension)

---

## 1. Pourquoi le Langage est Difficile

Une image de chat reste un chat sous plusieurs angles ; une phrase, elle, change de sens avec un mot, un contexte ou un ton.

- **Ambiguïté lexicale** : « avocat » (fruit ou juriste ?), « la petite brise la glace » (le vent ? une enfant ?).
- **Dépendance au contexte** : « il » renvoie à quoi ? Résoudre la **coréférence** demande de la mémoire.
- **Ordre et structure** : « le chien mord l'homme » ≠ « l'homme mord le chien » (déjà souligné au module 3).
- **Créativité et rareté** : de nouveaux mots, fautes, argot, emojis, mélanges de langues.
- **Monde extérieur** : « quel temps fait-il ? » exige une information que le texte seul ne contient pas — d'où les outils et le RAG (module 4).

> 💡 **Conséquence pratique.** Le texte est une donnée **non structurée** (module 1) : avant tout modèle, il faut le transformer en nombres — sans détruire l'information d'ordre et de sens.

---

## 2. Préparer le Texte : Normalisation et Tokenisation

### 2.1 Normalisation

Selon la tâche, on peut : passer en minuscules, retirer/uniformiser la ponctuation, gérer les accents et l'unicode, remplacer chiffres et URLs par des marqueurs. Ces choix ne sont **pas neutres** : mettre en minuscules fusionne « Pomme » (marque) et « pomme » (fruit) ; retirer la ponctuation détruit « 3,14 » ou l'ironie d'un « super... ».

Deux opérations classiques, aujourd'hui souvent inutiles avec les modèles modernes :
- **Stop-words** : retirer les mots très fréquents (« le », « de ») — utile pour TF-IDF, souvent nuisible pour un Transformer qui utilise ces mots grammaticaux.
- **Racinisation / lemmatisation** : ramener « chevaux » à « cheval », « mangeait » à « manger » — utile pour des méthodes lexicales, rarement pour des modèles à sous-mots.

> ⚠️ **Toute normalisation est une transformation apprise ou fixée qui fait partie du système versionné** (module 1). Le texte d'entraînement et d'inférence doit être traité **identiquement**.

### 2.2 Tokenisation

Découper le texte en unités. Trois familles :

| Granularité | Avantage | Inconvénient |
| :--- | :--- | :--- |
| Mots | intuitif | vocabulaire immense, mots inconnus (*OOV*) |
| Caractères | vocabulaire minuscule, aucun OOV | séquences très longues, sens dilué |
| Sous-mots (BPE, WordPiece, Unigram) | compromis : mots rares décomposés, pas d'OOV | découpage parfois contre-intuitif |

Les modèles modernes utilisent des **sous-mots** : « tokenisation » peut devenir `token` + `isation`. Un mot inconnu est reconstruit à partir de morceaux connus.

> 💡 **Un token n'est pas un mot.** En anglais, ~1 token ≈ 0,75 mot ; en français, l'estimation varie. Les coûts et limites d'un LLM se comptent **en tokens**, pas en mots (rappel module 4). Changer de tokenizer change les entrées du modèle (module 3, §14.1).

---

## 3. Représenter le Texte : du Sac de Mots aux Embeddings

### 3.1 One-hot et sac de mots (*Bag-of-Words*)

Chaque mot devient une dimension. Un document est le vecteur des **comptes** de ses mots. Simple, interprétable, mais : très grande dimension, très **creux** (sparse, module 1), et **aucun ordre** (« le chat mange » = « mange le chat »).

### 3.2 TF-IDF : pondérer l'importance

Le **TF-IDF** corrige le sac de mots en abaissant le poids des mots omniprésents et en relevant celui des mots discriminants :

$$\text{tfidf}(t, d)=\underbrace{\text{tf}(t, d)}_{\text{fréquence dans }d}\times\underbrace{\log\frac{N}{1+\text{df}(t)}}_{\text{rareté dans le corpus}}.$$

Un mot fréquent dans **ce** document mais rare **ailleurs** obtient un poids élevé. **TF-IDF + régression logistique ou SVM linéaire reste une baseline redoutable** en classification de texte (rappel module 2, §10.10) : à battre avant de sortir l'artillerie lourde.

### 3.3 Embeddings : la sémantique distributionnelle

Idée fondatrice (Firth) : *« on connaît un mot par la compagnie qu'il fréquente »*. En observant les contextes, on apprend pour chaque mot un **vecteur dense** (ex. 300 dimensions) où la proximité géométrique reflète la proximité de sens.

- **word2vec** apprend à prédire un mot à partir de son contexte (ou l'inverse) ;
- **GloVe** factorise des statistiques de co-occurrence globales.

La géométrie devient porteuse de sens ; l'exemple célèbre :
$$\text{vec}(\text{roi})-\text{vec}(\text{homme})+\text{vec}(\text{femme})\approx\text{vec}(\text{reine}).$$

On compare deux embeddings par **similarité cosinus** (module 0). Limite majeure de word2vec/GloVe : un mot = **un seul** vecteur, donc « avocat » n'a qu'une représentation malgré ses deux sens.

### 3.4 Embeddings contextuels

Les modèles à base de Transformers produisent un vecteur **qui dépend de la phrase** : « avocat » n'a pas le même vecteur dans « un avocat mûr » et « mon avocat plaide ». Ces embeddings contextuels (et les embeddings de **phrases** entières) sont la base de la recherche sémantique et du RAG (module 4, §8.3).

| Représentation | Contexte | Ordre | Cas d'usage |
| :--- | :---: | :---: | :--- |
| One-hot / sac de mots | non | non | baseline simple, features lexicales |
| TF-IDF | non | non | classification, recherche lexicale |
| word2vec / GloVe | non | non | similarité de mots, features |
| Embeddings contextuels | oui | oui | recherche sémantique, transfert, RAG |

---

## 4. Les Grandes Tâches du NLP

| Tâche | Entrée → sortie | Exemple | Métriques typiques |
| :--- | :--- | :--- | :--- |
| Classification de texte | document → classe | spam, sentiment, thème | accuracy, F1, PR-AUC (module 2) |
| Étiquetage de séquence | tokens → étiquette par token | NER, POS-tagging | F1 par entité, exactitude |
| Similarité / recherche | requête + docs → classement | recherche sémantique | Recall@k, nDCG (module 4) |
| Question-réponse extractive | question + contexte → empan | trouver la réponse dans un texte | Exact Match, F1 |
| Traduction | texte source → texte cible | fr → en | BLEU, COMET, humain |
| Résumé | long texte → court texte | article → résumé | ROUGE, fidélité, humain |
| Génération / dialogue | prompt → texte | assistant, rédaction | tâche-dépendant, humain |

> 💡 **Le protocole du module 2 reste obligatoire.** Split réaliste (attention aux quasi-doublons de documents, module 1, §8.3), baseline, métrique cohérente avec le coût des erreurs, incertitude, test intact. Une F1 « 0,84 » sans classe positive ni seuil reste incomplète.

Pour les tâches génératives (traduction, résumé), les métriques automatiques (BLEU, ROUGE, METEOR, CIDEr, ou scores sémantiques) capturent des aspects partiels et **ne remplacent pas une revue humaine** (rappel module 3, §16.7).

---

## 5. Des RNN aux Transformers pour le Langage

Le module 3 a posé les architectures ; voici leur lecture « langage ».

1. **RNN / LSTM / GRU** lisent les tokens un à un en maintenant un état caché. Ils souffrent du gradient qui s'atténue sur les longues dépendances et se parallélisent mal sur la longueur.
2. **Seq2seq (encodeur–décodeur)** : un LSTM encode la phrase source, un autre génère la cible token par token (`<BOS>` … `<EOS>`), avec **teacher forcing** à l'entraînement et risque d'*exposure bias* à l'inférence (module 3, §16.1). Sans attention, tout doit passer par un unique vecteur : goulot d'étranglement.
3. **Attention puis Transformer** : chaque position peut pondérer toutes les autres (module 3, §8 et §14). Gain décisif pour le langage : dépendances longues capturées et calcul parallélisable, au prix d'un coût mémoire quadratique en longueur de séquence.

Trois familles de Transformers, trois usages (module 3, §14.5) :

- **encodeur seul** (bidirectionnel) → comprendre : classification, NER, recherche ;
- **décodeur seul** (causal) → générer : complétion, dialogue, la plupart des LLMs ;
- **encodeur–décodeur** → transformer une séquence en une autre : traduction, résumé.

---

## 6. Les Grands Modèles de Langage (LLMs)

### 6.1 Ce qu'un LLM apprend vraiment

La plupart des LLMs sont des Transformers **décodeurs** préentraînés à une tâche apparemment banale : **prédire le token suivant**. En minimisant l'entropie croisée sur d'immenses corpus, le modèle apprend grammaire, faits fréquents, styles et régularités de raisonnement — comme effets **secondaires** de cet objectif.

$$\mathcal{L}=-\sum_{t}\log p_\theta(w_t\mid w_{<t}).$$

> ⚠️ **L'objectif n'est pas la capacité.** « Prédire le mot suivant » ne garantit ni vérité, ni calcul exact, ni absence de biais. Les capacités réelles doivent être **évaluées par tâche**, pas déduites de l'objectif de préentraînement (rappel module 3, §9.3).

### 6.2 Le pipeline d'adaptation

Après le préentraînement, plusieurs étapes (module 3, §15.5) façonnent le comportement :

```text
préentraînement (prédire le token suivant, corpus massif)
   → fine-tuning supervisé sur des exemples instruction→réponse
   → alignement par préférences (favoriser certaines réponses)
   → [éventuellement] adaptation métier : RAG, outils, fine-tuning ciblé
```

L'**alignement par préférences** (dont le RLHF, détaillé au module 6) oriente le style et la prudence des réponses ; il ne rend pas le modèle infaillible.

### 6.3 Fenêtre de contexte et lois d'échelle

- **Fenêtre de contexte** : le nombre maximal de tokens que le modèle peut lire d'un coup (rappel module 4, §11). Au-delà, il faut résumer ou récupérer (RAG).
- **Lois d'échelle** : augmenter paramètres, données et calcul améliore en général les performances de façon régulière, avec des rendements décroissants — une **tendance empirique**, pas une garantie sur une tâche précise.
- Des **capacités inattendues** apparaissent parfois avec l'échelle ; leur mesure dépend fortement du protocole d'évaluation, à interpréter avec prudence.

---

## 7. Piloter un LLM : Prompting et In-Context Learning

### 7.1 In-context learning

Un LLM peut adapter son comportement **sans réentraînement**, uniquement d'après les exemples présents dans le prompt :

- **zero-shot** : la consigne seule (« Classe ce message : positif ou négatif »).
- **few-shot** : quelques exemples résolus avant la vraie question.

```text
Exemple few-shot :
Avis : "Livraison rapide, produit conforme."  → positif
Avis : "Cassé à l'arrivée, service injoignable." → négatif
Avis : "Correct sans plus, un peu cher."        → ?
```

Le modèle ne « mémorise » pas ces exemples au-delà de la requête : ils vivent dans le contexte, consomment des tokens, et disparaissent ensuite.

### 7.2 Techniques de prompting utiles

- **Consigne explicite** : rôle, format de sortie attendu, contraintes.
- **Décomposition (*chain-of-thought* / étapes)** : demander un raisonnement par étapes peut aider sur les tâches à plusieurs pas — l'effet **dépend du modèle et de la tâche** (rappel module 4, §3.1) et n'est pas une garantie.
- **Format structuré** : exiger du JSON validable plutôt que du texte libre facilite le contrôle en aval (module 4, §9).
- **Sortie contrainte** : fournir un schéma, des options fermées, ou demander « je ne sais pas » plutôt que d'inventer.

> 💡 **Le prompt système n'est pas une frontière de sécurité** (rappel module 4, §1 du dico). Une consigne « ne révèle jamais X » peut être contournée ; les vraies garanties reposent sur du code et des autorisations externes.

### 7.3 Prompting, RAG ou fine-tuning ?

| Besoin | Prompting / few-shot | RAG (module 4) | Fine-tuning (module 3) |
| :--- | :---: | :---: | :---: |
| Ajuster un format ou un ton ponctuel | ✓ | possible | ✓ (stable) |
| Injecter des connaissances privées et citables | limité | ✓ | difficile |
| Mettre à jour des faits fréquemment | limité | ✓ | réentraînement |
| Réduire le coût par requête | variable | parfois non | potentiellement |

La bonne question reste « quelle combinaison minimise erreurs, coût et délai de mise à jour ? » (rappel module 3, §15.6).

---

## 8. Décodage : Comment le Texte est Généré

À chaque étape, le modèle produit une distribution sur le vocabulaire ; la **stratégie de décodage** choisit le token suivant (rappel module 3, §16.7).

| Stratégie | Principe | Effet |
| :--- | :--- | :--- |
| Greedy | prendre le token le plus probable | rapide, répétitif, myope |
| Beam search | garder plusieurs séquences candidates | meilleur pour traduction, parfois fade |
| Échantillonnage + température | tirer au hasard selon les probabilités | contrôle la diversité |
| Top-k | limiter aux $k$ tokens les plus probables | évite les tokens absurdes |
| Top-p (noyau) | garder la masse de probabilité cumulée $p$ | seuil adaptatif |

La **température** $T$ aplatit ($T>1$, plus créatif/risqué) ou pique ($T<1$, plus déterministe) la distribution avant tirage. Des **pénalités de répétition** limitent les boucles.

> 💡 **Déterminisme.** À température 0 (ou greedy), la sortie est quasi déterministe à graine et version fixées ; dès qu'on échantillonne, deux appels identiques peuvent différer. Pour une **comparaison loyale**, fixez prompt, température, top-p/k et graine (rappel module 3, §17.6).

---

## 9. Évaluation, Hallucination et Limites

### 9.1 Évaluer un système de langage

- **Tâches fermées** (classification, NER, extraction) : métriques du module 2, avec test intact et incertitude.
- **Génération** : combiner tests déterministes, jeux d'évaluation représentatifs, métriques automatiques *et* revue humaine sur grille (rappel module 3, §18.1). Pour le RAG, mesurer séparément récupération et génération (module 4, §8.7).
- **Contamination** : si des exemples de test figuraient dans le corpus de préentraînement, le score est optimiste. Documentez la provenance et cherchez les fuites (rappel module 1).

### 9.2 Hallucination

Un LLM produit un texte **plausible**, pas nécessairement **vrai**. Il génère un échantillon conditionné, pas une récupération garantie d'un fait (rappel module 3, §17). Réduire l'hallucination :

- **ancrer** la réponse sur des sources récupérées et **exiger des citations** (RAG, module 4) ;
- autoriser l'**abstention** (« information insuffisante ») plutôt que l'invention ;
- **vérifier** les faits critiques par un outil déterministe (calcul, base de données) ;
- garder un **humain dans la boucle** pour les décisions sensibles (module 4, §7).

### 9.3 Biais, sécurité et coût

- **Biais** : les corpus reflètent des stéréotypes ; le modèle peut les reproduire ou les amplifier. Évaluez par sous-groupe et documentez les limites (modules 1 et 8).
- **Sécurité** : injection de prompt directe et indirecte, exfiltration, usage abusif — traités en profondeur au module 4 (§13) et au module 8.
- **Vie privée** : un modèle peut mémoriser des exemples sensibles ; dédupliquez et testez la mémorisation (module 3, §15.5).
- **Coût et impact** : latence, tokens, mémoire, énergie ; le module 7 (MLOps) et le module 8 (impact environnemental) en traitent.

> 🔍 **Étude de cas éclair.** Un assistant juridique interne répond en citant des « articles de loi » inexistants mais crédibles. Diagnostic : hallucination non ancrée. Correctifs : RAG sur un corpus juridique gouverné avec citations obligatoires, abstention si aucune source, vérification des références par un outil, et validation humaine avant tout envoi. C'est le pont direct vers le module 4.

---

## 10. Checklist et Questions de Compréhension

### Checklist d'un projet NLP

- [ ] La normalisation et la tokenisation sont identiques à l'entraînement et à l'inférence, et versionnées.
- [ ] Une baseline TF-IDF + modèle linéaire est mesurée avant tout modèle lourd.
- [ ] Le split gère les quasi-doublons de documents et la contamination test/préentraînement.
- [ ] La métrique correspond à la tâche et au coût des erreurs ; l'incertitude est rapportée.
- [ ] Les représentations choisies (lexicales vs contextuelles) sont justifiées par le besoin.
- [ ] Pour la génération : température, top-p/k, graine et prompt sont fixés pour comparer.
- [ ] L'hallucination est traitée (ancrage, citations, abstention, vérification).
- [ ] Biais, vie privée, sécurité et coût sont évalués, pas seulement la qualité.

### Questions de compréhension

1. Pourquoi un token n'est-il pas un mot, et pourquoi cela change-t-il le calcul du coût d'un LLM ?
2. En quoi TF-IDF corrige-t-il le principal défaut du sac de mots ? Que ne corrige-t-il pas ?
3. Quelle limite de word2vec les embeddings contextuels lèvent-ils ?
4. Un LLM entraîné à « prédire le token suivant » sait-il pour autant calculer ou dire le vrai ? Justifiez.
5. Différence entre zero-shot et few-shot ; où « vivent » les exemples few-shot ?
6. Quel effet la température a-t-elle sur le décodage, et pourquoi la fixer pour une comparaison ?
7. Pourquoi le prompt système ne suffit-il pas à garantir une règle de sécurité ?
8. Citez trois leviers concrets pour réduire l'hallucination sur une question factuelle.

**Mini-étude de cas.** Vous construisez un classifieur de tickets support en 5 catégories, puis un assistant qui rédige une réponse. (a) Décrivez la préparation du texte et une baseline. (b) Choisissez une métrique adaptée à un fort déséquilibre de catégories et justifiez le seuil. (c) Pour la partie générative, indiquez la stratégie de décodage, comment vous ancrez et citez les sources, et deux tests d'hallucination. (d) Listez deux risques de biais et comment les mesurer.

---

Pour approfondir et vérifier ces notions, consultez les articles fondateurs et documentations regroupés dans [REFERENCES.md](../REFERENCES.md).
