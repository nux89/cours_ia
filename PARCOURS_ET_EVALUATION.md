# Parcours, couverture et preuves d'apprentissage

Ce document distingue un sujet introduit, un mécanisme pratiqué et une compétence évaluée. Un notebook fonctionnel ne démontre pas à lui seul la capacité à résoudre un problème nouveau. Les sorties enregistrées permettent la lecture ; l'apprenant doit aussi exécuter, expliquer, modifier et analyser les erreurs.

## 1. Progression

Parcours conseillé : **0 → 9 → 1 → 2 → 3 → 10 → 5 → 4 → 6 → 7 → 8**, puis projet final. Le module 9 peut être lu en parallèle des premiers TP de données. Les blocs de causalité et de séries du module 10 suivent le ML ; sa partie GCN requiert les bases de PyTorch du module 3.

Chaque séance suit : question → hypothèses → modèle mental/formule → exemple → exécution → variation → conclusion. Avant d'ouvrir une correction, rédiger une prédiction du résultat. Après exécution, expliquer un écart plutôt que changer arbitrairement la graine.

## 2. Matrice des objectifs et évaluations

| Bloc et support | Compétence observable | Exemple et pratique | Preuve attendue |
|---|---|---|---|
| [0 — Fondements](00_fondements_maths_python/cours_fondements_maths_python.md) | vérifier formes, gradient et probabilités | [NumPy et Bayes](00_fondements_maths_python/01_maths_numpy_pratique.ipynb) | refaire un calcul puis expliquer broadcasting et gradient |
| [9 — Raisonnement](09_raisonnement_symbolique_probabiliste/cours_raisonnement_symbolique_probabiliste.md) | choisir recherche/règles/CSP/Bayes | [A*, CSP et inférence](09_raisonnement_symbolique_probabiliste/01_recherche_contraintes_bayes.ipynb) | chemin optimal, preuve de règle, cas impossible et posterior normalisé |
| [1 — Données](01_nature_et_preparation_des_donnees/cours_nature_et_preparation_donnees.md) | préparer sans fuite et sélectionner avec justification | [Préparation](01_nature_et_preparation_des_donnees/01_preparation_donnees_pratique.ipynb), [EDA](01_nature_et_preparation_des_donnees/02_eda_correlations_pandas_polars.ipynb) | dictionnaire, matrice expliquée, exclusions justifiées, split intact |
| [2 — ML](02_machine_learning/cours_machine_learning.md) | choisir modèle, métrique et protocole | [Comparaisons scikit-learn](02_machine_learning/02_machine_learning_scikit_learn.ipynb) | baseline, validation, coût d'erreur et conclusion hors test |
| [3 — Réseaux](03_deep_learning/cours_deep_learning.md) | entraîner et interpréter architectures | [MLP](03_deep_learning/01_perceptron_et_mlp.ipynb), [CNN](03_deep_learning/02_cnn_vision.ipynb), [RNN](03_deep_learning/03_rnn_series_temporelles.ipynb), [LSTM](03_deep_learning/04_lstm_sequences.ipynb), [autres architectures](03_deep_learning/05_decouverte_autres_architectures.ipynb) | formes, losses, courbes train/test et comparaison à baseline |
| [3 — Ateliers](03_deep_learning/ateliers_avances.md) | observer transfert, décodage et débruitage | [Transfert](03_deep_learning/06_transfert_fine_tuning.ipynb), [captioning](03_deep_learning/07_captioning_cnn_gru.ipynb), [DDPM](03_deep_learning/08_diffusion_2d.ipynb) | poids figés inchangés, génération sans vrais préfixes, sampling indépendant |
| [10 — Causalité et temps](10_methodes_specialisees/cours_methodes_specialisees.md) | identifier un effet et prévoir sans futur caché | [Causalité et prévision](10_methodes_specialisees/01_causalite_prevision.ipynb) | expliciter DAG, confusion, origine, horizon et baseline saisonnière |
| [10 — Recommandation et graphes](10_methodes_specialisees/cours_methodes_specialisees.md) | construire classement et propagation | [Ranking et GCN](10_methodes_specialisees/02_recommandation_graphes.ipynb) | filtrer items vus, calculer NDCG et distinguer inductif/transductif |
| [5 — NLP](05_nlp_et_llms/cours_nlp_et_llms.md) | représenter et évaluer du texte | [TF-IDF et embeddings](05_nlp_et_llms/01_nlp_tfidf_embeddings.ipynb) | comparer représentations et expliquer décodage et hallucination |
| [4 — Agents](04_ia_agentique/cours_ia_agentique.md) | contrôler une boucle et sa récupération | [Orchestration](04_ia_agentique/01_tp_agent_autonome.ipynb), [documents et citations](04_ia_agentique/02_rag_documents_citations.ipynb) | erreurs d'outils, droits, arrêt, provenance et abstention |
| [6 — RL](06_apprentissage_par_renforcement/cours_apprentissage_par_renforcement.md) | formaliser état/action/récompense | [Q-learning](06_apprentissage_par_renforcement/01_qlearning_gridworld.ipynb) | politique apprise, comparaison et limite de récompense |
| [7 — Production](07_mlops_production/cours_mlops_production.md) | conserver pipeline et surveiller | [Persistance et dérive](07_mlops_production/01_pipeline_persistance_derive.ipynb) | parité avant/après chargement et diagnostic de changement |
| [8 — Responsabilité](08_ethique_securite_regulation/cours_ethique_securite_regulation.md) | mesurer et discuter les arbitrages | [Équité](08_ethique_securite_regulation/01_equite_metriques.ipynb) | métriques par groupe, incertitude et choix explicite |

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
