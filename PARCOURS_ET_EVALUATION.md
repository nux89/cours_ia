# Parcours, couverture et preuves d'apprentissage

Ce document distingue un sujet introduit, un mécanisme pratiqué et une compétence évaluée. Un notebook fonctionnel ne démontre pas à lui seul la capacité à résoudre un problème nouveau. Les sorties enregistrées permettent la lecture ; l'apprenant doit aussi exécuter, expliquer, modifier et analyser les erreurs.

## 1. Progression

Ce document est la référence pour l'ordre pédagogique. Les numéros des dossiers sont des identifiants historiques, **pas l'ordre à suivre**. Parcours complet : **0 → 1 → 2 → 9 → 3 → 5 → 10 → 6 → 4 → 8 → 7**, puis projet final. Les liens « précédent / suivant » en tête de chaque cours suivent cet ordre.

| Étape | Module | Pourquoi à ce stade ? | Avant de poursuivre, savoir… |
|---|---|---|---|
| 1 | 0 — Mathématiques et Python | installer les outils communs | suivre les formes, calculer une moyenne, une probabilité et une dérivée simple |
| 2 | 1 — Données | comprendre ce qui sera appris et évalué | définir cible, unité d'observation et split sans fuite |
| 3 | 2 — Machine learning | apprendre la démarche avant les architectures | construire une baseline et distinguer entraînement, validation et test |
| 4 | 9 — Recherche, logique et probabilités | comparer apprentissage, règles et recherche | distinguer prédiction, preuve, recherche de chemin et inférence |
| 5 | 3 — Deep learning | approfondir les modèles avec les bases d'évaluation acquises | suivre une boucle d'entraînement et les formes CNN/RNN/attention |
| 6 | 5 — NLP et LLMs | appliquer les représentations et Transformers au langage | expliquer tokenisation, embeddings, décodage et hallucination |
| 7 | 10 — Méthodes spécialisées | disposer des bases ML, des graphes et des réseaux | choisir hypothèses et protocole selon causalité, temps, ranking ou graphe |
| 8 | 6 — Renforcement | passer de prédictions à des décisions séquentielles | formuler état, action, retour et évaluation ; distinguer RLHF et prompting |
| 9 | 4 — Agentique | assembler langage, recherche, outils et contrôle | expliquer RAG, droits, arrêt, évaluation et intérêt éventuel du multi-agent |
| 10 | 8 — Éthique, sécurité et gouvernance | décider des usages acceptables avant déploiement | expliciter risques, arbitrages, responsabilités et critères de non-déploiement |
| 11 | 7 — MLOps et production | industrialiser un système déjà évalué et cadré | versionner, surveiller, gérer les incidents et revenir en arrière |

Le module 9 ne nécessite pas le ML : sa place après le module 2 permet une comparaison concrète des paradigmes sans interrompre le premier parcours données→modèle. Le module 10 est une branche de spécialisation, pas un prérequis du RL ou du RAG. Le RL précède les agents pour distinguer politique apprise et orchestration d'un LLM ; il n'est pas indispensable pour construire une boucle d'outils simple.

**Éthique dès le début.** Lire les sections 1 et 4 du module 8 avec la collecte du module 1, puis ses sections 2–3 avec les métriques du module 2. L'étape 10 est la synthèse complète, pas le premier contact avec les risques. Le module 7 prolonge ensuite les décisions de gouvernance par des mécanismes opérationnels.

### Progression à l'intérieur des supports

Les cours historiques alternent panorama et approfondissements. Les numéros de sections restent stables pour préserver les renvois ; pour une première lecture accompagnée, suivre ces séquences :

- **Module 0** : sections 1–2, puis 8 (NumPy), 3–7 et 9. Dans les calculs guidés, commencer par les sections 1–4 ; reprendre biais–variance avec le ML, attention et Adam avec le deep learning. Ces derniers calculs ne sont pas un prérequis pour préparer une table.
- **Module 1** : cadrage (§7), nature et formes (§1–2), outils (§13), qualité (§3), fuite et split (§5 et §8), EDA/corrélations (§14–15), préparation (§4), construction et sélection (§9 et §16), contrats (§10), démonstration (§17), puis gouvernance/production et bilan (§6, §11–12, §18). L'EDA guidant des choix de modèle porte sur le train. Les comparaisons prédictives des TP de données se reprennent après le module 2 : au premier passage, se concentrer sur les tables, graphiques et transformations.
- **Médias** : lire les sections 1–4 du complément avec le module 1 (formats et traitement), puis reprendre sa section 5 sur les mécanismes de fusion après les encodeurs–décodeurs du module 3. Les contrôles et exercices des sections 6–7 accompagnent les deux passages.
- **Module 3** : neurone/activations/MLP (§1, §3–4), tenseurs (§10), pertes (§11), gradient et entraînement (§2, §12–13), CNN et séquences (§5–7), attention/Transformer (§8, §14), transfert (§15), encodeurs–décodeurs (§16), génératif (§9, §17), évaluation (§18–19). Réaliser les TP 01 à 05 avec ces blocs, puis 06–08. Les paragraphes d'adaptation des LLM et de comparaison au RAG (§15.5–15.6) sont une annonce à reprendre après les modules 5 et 4, pas des prérequis cachés.
- **Module 5** : suivre les sections 1–10 ; le RLHF est d'abord situé dans le cycle d'un LLM, puis expliqué techniquement au module 6. Pas besoin de connaître le RL pour comprendre tokenisation ou décodage.
- **Module 10** : suivre causalité, temps, recommandation, puis graphes. Le bloc séries (§2) peut être lu plus tôt, juste avant le TP RNN du module 3 ; le GCN attend la maîtrise de PyTorch et de la rétropropagation.
- **Module 4** : panorama (§1–2), outils et contrats (§4, §9), récupération documentaire (§8), mémoire (§5, §11), skills (§10), planification (§3), garde-fous (§7, §13), puis multi-agents (§6, §12), évaluation/cas/bilan (§14–16). Faire le TP d'orchestration après les contrats et garde-fous ; le TP documentaire après le RAG. Ne pas multiplier les agents avant de savoir contrôler un agent unique.
- **Architectures émergentes** : ouverture optionnelle après les modules 3 et 5 et les ateliers avancés ; elle n'est pas requise pour réussir les chapitres suivants.

Dans les modules 2, 6, 8 et 7, suivre l'ordre interne, avec retour ponctuel aux rappels mathématiques. Un renvoi vers un chapitre ultérieur est un approfondissement annoncé, pas une notion supposée déjà maîtrisée.

Chaque séance suit : question → hypothèses → modèle mental/formule → exemple → exécution → variation → conclusion. Avant d'ouvrir une correction, rédiger une prédiction du résultat. Après exécution, expliquer un écart plutôt que changer arbitrairement la graine.

## 2. Matrice des objectifs et évaluations

| Bloc et support | Compétence observable | Exemple et pratique | Preuve attendue |
|---|---|---|---|
| [0 — Fondements](00_fondements_maths_python/cours_fondements_maths_python.md) | vérifier formes, gradient et probabilités | [NumPy et Bayes](00_fondements_maths_python/01_maths_numpy_pratique.ipynb) | refaire un calcul puis expliquer broadcasting et gradient |
| [1 — Données](01_nature_et_preparation_des_donnees/cours_nature_et_preparation_donnees.md) | préparer sans fuite et sélectionner avec justification | [Préparation](01_nature_et_preparation_des_donnees/01_preparation_donnees_pratique.ipynb), [EDA](01_nature_et_preparation_des_donnees/02_eda_correlations_pandas_polars.ipynb) | dictionnaire, matrice expliquée, exclusions justifiées, split intact |
| [2 — ML](02_machine_learning/cours_machine_learning.md) | choisir modèle, métrique et protocole | [Comparaisons scikit-learn](02_machine_learning/02_machine_learning_scikit_learn.ipynb) | baseline, validation, coût d'erreur et conclusion hors test |
| [9 — Raisonnement](09_raisonnement_symbolique_probabiliste/cours_raisonnement_symbolique_probabiliste.md) | choisir recherche/règles/CSP/Bayes | [A*, CSP et inférence](09_raisonnement_symbolique_probabiliste/01_recherche_contraintes_bayes.ipynb) | chemin optimal, preuve de règle, cas impossible et posterior normalisé |
| [3 — Réseaux](03_deep_learning/cours_deep_learning.md) | entraîner et interpréter architectures | [MLP](03_deep_learning/01_perceptron_et_mlp.ipynb), [CNN](03_deep_learning/02_cnn_vision.ipynb), [RNN](03_deep_learning/03_rnn_series_temporelles.ipynb), [LSTM](03_deep_learning/04_lstm_sequences.ipynb), [autres architectures](03_deep_learning/05_decouverte_autres_architectures.ipynb) | formes, losses, courbes train/test et comparaison à baseline |
| [3 — Ateliers](03_deep_learning/ateliers_avances.md) | observer transfert, décodage et débruitage | [Transfert](03_deep_learning/06_transfert_fine_tuning.ipynb), [captioning](03_deep_learning/07_captioning_cnn_gru.ipynb), [DDPM](03_deep_learning/08_diffusion_2d.ipynb) | poids figés inchangés, génération sans vrais préfixes, sampling indépendant |
| [5 — NLP](05_nlp_et_llms/cours_nlp_et_llms.md) | représenter et évaluer du texte | [TF-IDF et embeddings](05_nlp_et_llms/01_nlp_tfidf_embeddings.ipynb) | comparer représentations et expliquer décodage et hallucination |
| [10 — Causalité et temps](10_methodes_specialisees/cours_methodes_specialisees.md) | identifier un effet et prévoir sans futur caché | [Causalité et prévision](10_methodes_specialisees/01_causalite_prevision.ipynb) | expliciter DAG, confusion, origine, horizon et baseline saisonnière |
| [10 — Recommandation et graphes](10_methodes_specialisees/cours_methodes_specialisees.md) | construire classement et propagation | [Ranking et GCN](10_methodes_specialisees/02_recommandation_graphes.ipynb) | filtrer items vus, calculer NDCG et distinguer inductif/transductif |
| [6 — RL](06_apprentissage_par_renforcement/cours_apprentissage_par_renforcement.md) | formaliser état/action/récompense | [Q-learning](06_apprentissage_par_renforcement/01_qlearning_gridworld.ipynb) | politique apprise, comparaison et limite de récompense |
| [4 — Agents](04_ia_agentique/cours_ia_agentique.md) | contrôler une boucle et sa récupération | [Orchestration](04_ia_agentique/01_tp_agent_autonome.ipynb), [documents et citations](04_ia_agentique/02_rag_documents_citations.ipynb) | erreurs d'outils, droits, arrêt, provenance et abstention |
| [8 — Responsabilité](08_ethique_securite_regulation/cours_ethique_securite_regulation.md) | mesurer et discuter les arbitrages | [Équité](08_ethique_securite_regulation/01_equite_metriques.ipynb) | métriques par groupe, incertitude et choix explicite |
| [7 — Production](07_mlops_production/cours_mlops_production.md) | conserver pipeline et surveiller | [Persistance et dérive](07_mlops_production/01_pipeline_persistance_derive.ipynb) | parité avant/après chargement et diagnostic de changement |

## 3. Barème d'une séance sur 10

Lectures complémentaires évaluables sans nouveau notebook : [calculs mathématiques](00_fondements_maths_python/complements_mathematiques.md) (refaire les exemples et annoncer les hypothèses), [médias et multimodal](01_nature_et_preparation_des_donnees/traitement_images_video_audio_multimodal.md) (définir manifeste, alignement et split), [architectures émergentes](03_deep_learning/architectures_emergentes.md) (proposer baseline, ablation et budget équitable). Prévoir respectivement 2–3 h, 3–4 h et 1–2 h.

- 2 points : question, population et hypothèses correctement formulées.
- 2 points : mécanisme expliqué par un calcul ou un schéma des formes.
- 2 points : code exécuté et contrôles compris.
- 2 points : variation ou ablation menée sans fuite d'information.
- 2 points : conclusion appuyée sur les sorties, avec une limite pertinente.

Une copie des sorties sans justification ne valide pas les deux derniers critères. Pour l'auto-évaluation, reformuler les corrections avec ses propres mots puis refaire l'expérience sur un autre cas.

## 4. Ce que le parcours ne prétend pas couvrir intégralement

Les ateliers rendent exécutables des mécanismes réduits : transfert sur chiffres, légendes de formes synthétiques, diffusion 2D, recherche documentaire extractive. Ils n'entraînent pas un grand modèle de fondation et ne déploient pas de service génératif distant. La planification robotique physique, la logique avancée, l'inférence causale avancée, les grands systèmes de recommandation, la vision dense, la parole, l'entraînement distribué et les preuves formelles de sécurité restent des spécialisations.

Le [projet final](PROJET_FINAL.md) vérifie une démarche complète dans une voie choisie ; il ne certifie pas la maîtrise de toutes ces spécialisations.
