# 📚 Glossaire transversal du cursus

Index unique de tout le vocabulaire du cursus. Chaque entrée donne une définition courte, le terme anglais usuel et le(s) module(s) où la notion est développée. Les « Dico du Débutant » en tête de chaque module restent le point d'entrée pédagogique ; ce glossaire sert de référence rapide et de table de correspondance français ↔ anglais.

Modules : **[0]** Fondements · **[1]** Données · **[2]** Machine Learning · **[3]** Deep Learning · **[4]** IA Agentique · **[5]** NLP & LLMs · **[6]** Reinforcement Learning · **[7]** MLOps · **[8]** Éthique

---

## A

- **Accuracy (Exactitude)** — proportion de prédictions correctes. Trompeuse en cas de classes déséquilibrées. **[1, 2]**
- **Actor-Critic (Acteur-critique)** — famille de RL combinant une politique (acteur) et une fonction de valeur (critique). **[6]**
- **AI Act** — régulation européenne classant les systèmes d'IA par niveau de risque. **[8]**
- **Anomalie (détection d')** — repérage d'observations rares ou atypiques (Isolation Forest, LOF, One-Class SVM). **[2]**
- **Anonymisation / Pseudonymisation** — rendre des données non attribuables (irréversible) / remplacer les identifiants par des codes (réversible). **[1, 8]**
- **Apprentissage supervisé / non supervisé / par renforcement** — les trois grands paradigmes : avec corrigé / sans corrigé / par récompense. **[2, 6]**
- **Attention** — mécanisme pondérant l'importance des positions d'une séquence via Q, K, V. **[3, 5]**
- **AUC (ROC-AUC, PR-AUC)** — aire sous une courbe ROC ou précision-rappel ; à interpréter avec la prévalence. **[2]**
- **Autograd** — différentiation automatique construisant le graphe de calcul pour la rétropropagation. **[3]**
- **Auto-encodeur (AE, VAE)** — réseau qui compresse puis reconstruit ; le VAE apprend une distribution latente. **[2, 3]**

## B

- **Backpropagation (Rétropropagation)** — calcul des gradients par la règle de la chaîne, de la sortie vers l'entrée. **[0, 3]**
- **Bagging** — agrégation de modèles entraînés sur des échantillons différents ; réduit la variance. **[2]**
- **Bandit** — RL à décision unique arbitrant exploration et exploitation. **[6]**
- **Baseline** — niveau de référence simple à dépasser (moyenne, classe majoritaire, dernière valeur). **[2]**
- **Batch / Mini-batch** — paquet d'exemples traités ensemble pour une mise à jour. **[3]**
- **BatchNorm / LayerNorm** — normalisations stabilisant l'entraînement ; LayerNorm est centrale dans les Transformers. **[3]**
- **Bellman (équations de)** — cohérence récursive valeur présente = récompense + valeur future actualisée. **[6]**
- **Biais (statistique)** — erreur de simplification d'un modèle trop pauvre (sous-apprentissage). **[2]**
- **Biais (équité)** — traitement systématiquement défavorable à un groupe. **[1, 8]**
- **Biais-variance (compromis)** — arbitrage entre simplification et sensibilité aux données. **[2]**
- **Boosting / Gradient Boosting** — ajout séquentiel de modèles corrigeant les erreurs (XGBoost, LightGBM, CatBoost). **[2]**
- **Broadcasting** — extension automatique des formes compatibles en NumPy/tenseurs ; source de bugs silencieux. **[0, 3]**

## C

- **Calibration** — adéquation entre probabilités annoncées et fréquences réelles (score de Brier). **[2]**
- **Canary / Shadow deployment** — déploiement progressif : petit % du trafic / exécution parallèle sans agir. **[7]**
- **Causalité vs corrélation** — prédire n'est pas expliquer ; agir sur une variable ne garantit pas de changer la cible. **[1, 2]**
- **Chain-of-thought** — invite à raisonner par étapes ; effet variable selon modèle et tâche. **[5]**
- **Chunking (Découpage)** — segmentation de documents en passages pour le RAG. **[4]**
- **CI/CD** — intégration et livraison continues, étendues aux tests de données et gardes de métriques. **[7]**
- **Classification** — prédire une catégorie discrète. **[2]**
- **Clustering (Partitionnement)** — regrouper sans étiquette (K-means, GMM, DBSCAN). **[2]**
- **CNN (Réseau convolutif)** — réseau exploitant champs récepteurs locaux et partage de poids pour la vision. **[3]**
- **Confidentialité différentielle** — bruit calibré masquant la présence d'un individu. **[8]**
- **Contrat de données** — attentes testables sur schéma, complétude, domaine, unicité, fraîcheur. **[1, 7]**
- **Corrélation (Pearson, Spearman, Kendall)** — mesure d'association linéaire/monotone ; ≠ causalité. **[1]**
- **Cross-attention** — attention où requêtes et clés/valeurs viennent de sources différentes (texte ↔ image). **[3]**
- **Cross-entropy (Entropie croisée)** — perte de classification ; équivaut au maximum de vraisemblance. **[0, 2, 3]**
- **Cross-validation (Validation croisée)** — estimation par plis multiples ; imbriquée pour le réglage. **[2]**

## D

- **Data Leakage (Fuite de données)** — laisser filtrer l'information de test/du futur dans l'entraînement. **[1]**
- **Datasheet (Fiche de données)** — documentation de motivation, collecte, usages et limites d'un jeu. **[1, 8]**
- **Décodage (greedy, beam, top-k, top-p, température)** — stratégies de génération de séquence. **[3, 5]**
- **Descente de gradient (SGD, Adam, AdamW)** — optimisation par pas opposés au gradient. **[0, 3]**
- **Diffusion (modèle de)** — génération par débruitage progressif ; latente pour réduire le coût. **[3]**
- **Dérive (drift : entrée, cible, concept)** — évolution de $P(X)$, $P(y)$ ou $P(y|X)$ en production. **[1, 7]**
- **Dropout** — neutralisation aléatoire d'activations pour régulariser. **[3]**

## E

- **Embedding** — vecteur dense représentant un token, un texte, une image ; proximité = similarité. **[3, 5]**
- **Encodeur–Décodeur** — architecture qui encode une entrée puis génère une sortie conditionnée. **[3]**
- **Ensemble** — combinaison de modèles (bagging, boosting, voting, stacking). **[2]**
- **Entropie** — incertitude moyenne d'une distribution. **[0]**
- **Époque (Epoch)** — un passage complet sur le jeu d'entraînement. **[3]**
- **Équité (fairness)** — non-discrimination ; plusieurs définitions parfois incompatibles. **[8]**
- **Espérance / Variance** — moyenne théorique / dispersion d'une variable aléatoire. **[0]**
- **Exploration vs Exploitation** — dilemme du RL : essayer du nouveau vs exploiter le connu. **[6]**

## F

- **Feature (Caractéristique)** — colonne/variable d'entrée. **[1]**
- **Feature engineering** — création de variables plus proches du mécanisme étudié. **[1]**
- **Feature store** — magasin de features garantissant la cohérence entraînement/service. **[7]**
- **Few-shot / Zero-shot** — fournir quelques exemples / aucun exemple dans le prompt. **[5]**
- **Fine-tuning** — poursuite de l'entraînement d'un modèle préentraîné sur une tâche cible. **[3, 5]**
- **Fonction d'activation (ReLU, Sigmoïde, Softmax, GELU)** — non-linéarité d'un neurone. **[3]**
- **F1 (score)** — moyenne harmonique de précision et rappel. **[2]**

## G

- **GAN** — générateur vs discriminateur en jeu adversarial. **[3]**
- **Garde-fou (Guardrail)** — contrôle préventif/détectif/correctif hors modèle. **[4]**
- **Gradient** — vecteur des dérivées partielles ; direction de plus forte montée. **[0]**
- **Gradient clipping** — bornage des gradients pour éviter l'explosion. **[3]**
- **GridSearchCV** — recherche exhaustive d'hyperparamètres sur une grille, par validation croisée. **[2]**

## H

- **Hallucination** — sortie fluide mais fausse ou non fondée. **[4, 5]**
- **Human-in-the-Loop (HITL)** — validation humaine des actions/décisions sensibles. **[4, 8]**
- **Hyperparamètre** — réglage choisi avant l'entraînement (vs paramètre appris). **[2]**

## I – K

- **Idempotence** — propriété d'une opération dont la répétition ne change pas le résultat (évite un double paiement). **[4]**
- **Imputation** — remplacement des valeurs manquantes (médiane, mode, k-NN, itérative). **[1]**
- **In-context learning** — adaptation d'un LLM via les seuls exemples du prompt. **[5]**
- **Information mutuelle** — réduction d'incertitude sur une variable connaissant une autre. **[0, 1]**
- **Injection de prompt (directe/indirecte)** — détournement d'un système LLM par du texte. **[4, 5, 8]**
- **Interprétabilité / Explicabilité** — capacité à comprendre/justifier une décision. **[2, 8]**
- **KL (divergence de Kullback–Leibler)** — écart entre deux distributions ; pénalité en RLHF/VAE. **[0, 6]**
- **k-NN** — prédiction par les $k$ voisins les plus proches ; sensible à l'échelle. **[2]**
- **K-Means** — partitionnement en $K$ groupes autour de centroïdes. **[2]**

## L

- **LLM (Grand modèle de langage)** — modèle de langage massif préentraîné, souvent Transformer décodeur. **[4, 5]**
- **Logit** — score non borné avant activation ; entrée de `BCEWithLogitsLoss`/`CrossEntropyLoss`. **[3]**
- **Loss (Fonction de perte)** — signal optimisé à l'entraînement (MSE, MAE, log-loss…). **[2, 3]**
- **LSTM / GRU** — cellules récurrentes à portes atténuant le vanishing gradient. **[3]**

## M

- **MAE / RMSE / R²** — métriques de régression. **[2]**
- **MCAR / MAR / MNAR** — mécanismes de valeurs manquantes. **[1]**
- **MDP (Processus de décision markovien)** — cadre du RL : états, actions, dynamique, récompense, $\gamma$. **[6]**
- **Mémoire (agentique)** — travail, conversation, épisodique, sémantique, procédurale. **[4]**
- **Model card (Fiche modèle)** — documentation d'usages, performances, sous-groupes et limites. **[2, 8]**
- **Model registry (Registre de modèles)** — catalogue versionné avec stades de déploiement. **[7]**
- **Multicolinéarité** — variables fortement corrélées rendant les coefficients instables. **[1]**
- **Multimodal** — modalités alignées (image–texte, vidéo–audio). **[1, 3]**

## N – O

- **NER (Reconnaissance d'entités)** — étiquetage des entités d'un texte (personnes, lieux…). **[5]**
- **Normalisation (Min-Max, Z-score, Robust)** — mise à l'échelle des variables. **[1]**
- **One-Hot Encoding** — colonne binaire par modalité nominale. **[1]**
- **Optimiseur** — algorithme de mise à jour des poids (SGD, Adam, AdamW). **[3]**
- **Outlier (Valeur aberrante)** — mesure extrême, erreur ou événement rare réel. **[1]**
- **Overfitting / Underfitting (Sur/Sous-apprentissage)** — mémorisation du bruit / modèle trop pauvre. **[2]**

## P

- **PCA (ACP)** — réduction de dimension par directions de variance maximale. **[2]**
- **PEFT** — adaptation efficace en paramètres d'un grand modèle (adaptateurs, LoRA, prompt tuning). **[3]**
- **Pipeline (Scikit-Learn)** — chaînage transformations + modèle empêchant les fuites. **[1, 2]**
- **Policy (Politique $\pi$)** — stratégie de choix d'action en RL. **[6]**
- **Policy gradient (REINFORCE)** — optimisation directe de la politique. **[6]**
- **Précision / Rappel (Precision / Recall)** — fiabilité des alertes / couverture des positifs. **[2]**
- **Préentraînement** — apprentissage de représentations sur une tâche source et un grand corpus. **[3, 5]**
- **Prompt / Prompt système** — texte d'entrée ; instruction de haut niveau (pas une frontière de sécurité). **[4, 5]**
- **p-value** — probabilité d'un effet aussi extrême sous l'hypothèse nulle ; ≠ taille d'effet. **[0]**

## Q – R

- **Q-learning / SARSA** — apprentissage de la valeur d'action, hors politique / sur politique. **[6]**
- **RAG (Génération augmentée par récupération)** — récupérer des passages pour ancrer et citer une réponse. **[4]**
- **Random Forest** — bagging d'arbres avec sous-échantillonnage des variables. **[2]**
- **ReAct** — patron d'orchestration alternant raisonnement, action et observation. **[4]**
- **Récompense (Reward)** — signal scalaire guidant le RL ; sa mauvaise spécification mène au *reward hacking*. **[6]**
- **Régression** — prédire une quantité continue. **[2]**
- **Régularisation (L1/Lasso, L2/Ridge, weight decay, early stopping)** — limiter la complexité. **[1, 2, 3]**
- **Reproductibilité** — capacité à reconstruire un résultat (code, données, env, modèle versionnés). **[2, 7]**
- **Rollback** — retour à une version stable antérieure. **[7]**
- **RLHF** — alignement d'un modèle par préférences humaines (souvent via PPO). **[5, 6]**
- **RNN** — réseau récurrent à état caché pour les séquences. **[3]**

## S

- **Sac de mots (Bag-of-Words) / TF-IDF** — représentations lexicales du texte. **[5]**
- **Scaling (Mise à l'échelle)** — voir Normalisation. **[1]**
- **Seq2seq** — encodeur–décodeur pour transformer une séquence en une autre. **[3, 5]**
- **Similarité cosinus** — comparaison d'angle entre vecteurs ; cœur de la recherche sémantique. **[0, 4]**
- **Skill (Compétence)** — module procédural versionné indiquant comment accomplir une famille de tâches. **[4]**
- **SVM** — séparateur à marge maximale, avec astuce du noyau. **[2]**
- **Split (Train/Validation/Test)** — découpage aux rôles distincts ; test intact. **[1, 2]**
- **Stacking** — méta-modèle combinant des prédictions hors pli. **[2]**

## T

- **Teacher forcing** — fournir le vrai token précédent au décodeur à l'entraînement. **[3]**
- **Température** — paramètre aplatissant/piquant la distribution avant échantillonnage. **[3, 5]**
- **Tenseur** — tableau de nombres multidimensionnel. **[0, 3]**
- **TD (Différence temporelle)** — mise à jour de valeur par bootstrapping à chaque pas. **[6]**
- **Théorème central limite** — la moyenne de nombreuses variables tend vers une loi normale. **[0]**
- **Token / Tokenisation** — unité de texte / découpage en unités. **[3, 5]**
- **Tool calling (Appel d'outils)** — le modèle propose un appel, l'orchestrateur valide et exécute. **[4]**
- **Train/serving skew** — décalage entre données d'entraînement et de service. **[7]**
- **Transfer learning** — réutilisation de représentations préentraînées. **[3]**
- **Transformer** — architecture à attention, parallélisable ; encodeur/décodeur/les deux. **[3, 5]**

## U – Z

- **UMAP / t-SNE** — réduction de dimension pour la visualisation ; distances globales trompeuses. **[2]**
- **Validation croisée** — voir Cross-validation. **[2]**
- **Vanishing/Exploding gradient** — gradient qui s'atténue/explose sur les longues séquences. **[3]**
- **Variance (compromis biais-variance)** — sensibilité excessive aux données d'entraînement. **[2]**
- **Vectorisation** — remplacer les boucles par des opérations sur tableaux entiers. **[0]**
- **V de Cramér** — association entre deux variables catégorielles (normalise le χ²). **[1]**
- **Vraisemblance (Likelihood)** — plausibilité des données sous des paramètres ; maximisée en apprentissage. **[0]**
- **Vecteur (base vectorielle / index)** — structure d'index d'embeddings ; ≠ base de connaissances complète. **[4]**
- **Zero-shot** — voir Few-shot. **[5]**

---

*Pour les références bibliographiques associées à chaque notion, voir [REFERENCES.md](REFERENCES.md). Pour la vue d'ensemble et l'ordre conseillé, voir [README.md](README.md).*
