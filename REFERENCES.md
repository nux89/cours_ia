# Références et documentations de vérification

## Compléments : médias et architectures de recherche

Les lectures, mécanismes et limites sont développés dans le [support multimodal](01_nature_et_preparation_des_donnees/traitement_images_video_audio_multimodal.md) et l'[ouverture sur les architectures](03_deep_learning/architectures_emergentes.md). Sources primaires consultées pour cette extension le 5 septembre 2026 :

- [Rééchantillonnage audio — documentation PyTorch](https://docs.pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html).
- [Mamba : Selective State Spaces](https://arxiv.org/abs/2312.00752) et [Mamba-2 : Structured State Space Duality](https://arxiv.org/abs/2405.21060).
- [Switch Transformers](https://arxiv.org/abs/2101.03961) : mélange d'experts parcimonieux.
- [I-JEPA](https://arxiv.org/abs/2301.08243) : apprentissage de représentations prédictives.
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
- [KAN : Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756).

Cette bibliographie privilégie les articles fondateurs, les documentations officielles et les référentiels de sécurité. Les bibliothèques évoluent : vérifier la documentation correspondant à la version installée.

## Recherche, raisonnement et méthodes spécialisées

- Russell et Norvig, [code associé à Artificial Intelligence: A Modern Approach](https://github.com/aimacode/aima-python) — recherche, logique, CSP et probabilités ; [manuel ouvert CS188 de Berkeley](https://inst.eecs.berkeley.edu/~cs188/textbook/) pour les explications.
- PyWhy, [DoWhy — estimation des effets causaux](https://www.pywhy.org/dowhy/v0.13/user_guide/causal_tasks/estimating_causal_effects/index.html) — modéliser, identifier, estimer, éprouver les hypothèses.
- Hyndman et Athanasopoulos, [Forecasting: Principles and Practice — validation temporelle](https://otexts.com/fpp3/tscv.html) — origines glissantes et horizons de prévision.
- Hu, Koren et Volinsky (2008), [Collaborative Filtering for Implicit Feedback Datasets](https://doi.org/10.1109/ICDM.2008.22) — interactions implicites et confiance.
- Kipf et Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — propagation normalisée et apprentissage transductif.

## Données et machine learning

- Pandas, [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html) et [introduction aux structures](https://pandas.pydata.org/docs/user_guide/dsintro.html) — `Series`, `DataFrame`, sélection, manquants, groupements et types.
- Pandas, [`DataFrame.corr`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html) — corrélations Pearson, Spearman et Kendall avec observations disponibles par paire.
- Polars, [Data types and structures](https://docs.pola.rs/user-guide/concepts/data-types-and-structures/), [expressions](https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/) et [Lazy API](https://docs.pola.rs/user-guide/concepts/lazy-api/) — types, expressions, plans et optimisations de requêtes.
- Seaborn, [`heatmap`](https://seaborn.pydata.org/generated/seaborn.heatmap.html) — représentation d'une matrice par intensité de couleur.
- Scikit-Learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) — prétraitement cohérent, fuite de données et contrôle de l'aléatoire.
- Scikit-Learn, [`load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) — définition du jeu et ordre des classes (`malignant`, `benign`).
- Scikit-Learn, [Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) — métriques, scores et conventions de classe positive.
- Scikit-Learn, [Supervised learning](https://scikit-learn.org/stable/supervised_learning.html) — panorama officiel des modèles linéaires, SVM, voisins, processus gaussiens, arbres, ensembles et réseaux supervisés.
- Scikit-Learn, [Unsupervised learning](https://scikit-learn.org/stable/unsupervised_learning.html) — clustering, mélanges gaussiens, réduction de dimension et détection de structures.
- Scikit-Learn, [Outlier and novelty detection](https://scikit-learn.org/stable/modules/outlier_detection.html) — distinction et modèles d'anomalies.
- Scikit-Learn, [Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html) — filtres univariés, information mutuelle, RFE et sélection intégrée.
- UCI Machine Learning Repository, [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) — provenance du jeu médical.
- Scikit-Learn, [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html) — chaînage des transformations et modèles.
- Scikit-Learn, [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) — validation, données groupées et temporelles.
- Scikit-Learn, [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html) — courbes de calibration, score de Brier et calibration des probabilités.
- Gebru et al. (2021), [Datasheets for Datasets](https://doi.org/10.1145/3458723), *Communications of the ACM* — documentation de motivation, composition, collecte, usages et maintenance des jeux de données.
- Mitchell et al. (2019), [Model Cards for Model Reporting](https://doi.org/10.1145/3287560.3287596) — documentation des usages, performances, sous-groupes et limites des modèles.

## Deep learning

- LeCun, Bengio et Hinton (2015), [Deep learning](https://doi.org/10.1038/nature14539), *Nature*.
- Rumelhart, Hinton et Williams (1986), [Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0), *Nature*.
- Hochreiter et Schmidhuber (1997), [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735), *Neural Computation*.
- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Sutskever, Vinyals et Le (2014), [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — architecture encodeur–décodeur récurrente.
- Vinyals et al. (2014), [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) — association d'un encodeur visuel profond et d'un décodeur récurrent.
- Xu et al. (2015), [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) — attention visuelle pour la génération de légendes.
- Radford et al. (2021), [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — apprentissage contrastif et transfert texte–image.
- Ho, Jain et Abbeel (2020), [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — formulation moderne des modèles de diffusion par débruitage.
- Rombach et al. (2022), [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — diffusion dans un espace latent et conditionnement par cross-attention.
- PyTorch, [`BCEWithLogitsLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) — stabilité numérique pour la classification binaire.
- PyTorch, [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) — portée et limites des graines aléatoires.
- PyTorch, [Transfer Learning for Computer Vision](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — extracteur figé et fine-tuning d'un CNN préentraîné.
- Scikit-Learn, [`load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) — jeu d'images embarqué utilisé par le TP CNN.

## Systèmes agentiques et sécurité

- Yao et al. (2022), [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629).
- Lewis et al. (2020), [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — association d'une mémoire paramétrique et d'un index documentaire récupéré.
- Karpukhin et al. (2020), [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) — récupération dense bi-encodeur.
- Wu et al. (2024), [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://www.microsoft.com/en-us/research/publication/autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-framework/) — composition et patrons de conversation multi-agents.
- OpenAI, [Function calling](https://developers.openai.com/api/docs/guides/function-calling) — outils structurés et validation par schéma dans une API contemporaine.
- Model Context Protocol, [Tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — contrats et découverte d'outils dans un protocole d'interopérabilité.
- Agent Skills, [Specification](https://agentskills.io/specification) — exemple de format ouvert fondé sur un fichier `SKILL.md` et des ressources optionnelles.
- NIST, [AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) et [profil pour l'IA générative](https://doi.org/10.6028/NIST.AI.600-1) — gouvernance du risque sur le cycle de vie.
- OWASP, [AI Agent Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html) — moindre privilège, validation, approbations et journalisation.
- OWASP, [Prompt Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html) — défense en profondeur contre les injections directes et indirectes.
- OWASP, [MCP Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html) — menaces d'autorisation, délégation et chaîne d'approvisionnement autour des serveurs d'outils.

## Fondements mathématiques et Python

- Deisenroth, Faisal et Ong (2020), [Mathematics for Machine Learning](https://mml-book.github.io/), Cambridge University Press — algèbre linéaire, calcul différentiel, probabilités et optimisation pour l'IA (libre d'accès).
- Goodfellow, Bengio et Courville (2016), [Deep Learning](https://www.deeplearningbook.org/), MIT Press — partie I : prérequis mathématiques.
- NumPy, [documentation officielle](https://numpy.org/doc/stable/) et [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) — tableaux, types et vectorisation.
- NumPy, [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) — règles d'extension des formes et pièges associés.
- Python, [Floating-Point Arithmetic: Issues and Limitations](https://docs.python.org/3/tutorial/floatingpoint.html) — pourquoi `0.1 + 0.2 != 0.3` et comment comparer des flottants.

## Traitement du langage naturel et grands modèles de langage

- Jurafsky et Martin, [Speech and Language Processing (3e édition, brouillon)](https://web.stanford.edu/~jurafsky/slp3/) — référence de NLP, du prétraitement aux Transformers.
- Mikolov et al. (2013), [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — word2vec et sémantique distributionnelle.
- Pennington, Socher et Manning (2014), [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) — embeddings par co-occurrence.
- Sennrich, Haddow et Birch (2016), [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — tokenisation par sous-mots (BPE).
- Devlin et al. (2019), [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — encodeur bidirectionnel.
- Brown et al. (2020), [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — apprentissage en contexte (in-context learning).
- Hugging Face, [Transformers documentation](https://huggingface.co/docs/transformers/index) — tokenizers, modèles préentraînés et stratégies de génération.

## Apprentissage par renforcement

- Sutton et Barto (2018), [Reinforcement Learning: An Introduction (2e édition)](http://incompleteideas.net/book/the-book-2nd.html) — référence fondatrice, libre d'accès.
- Mnih et al. (2015), [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236), *Nature* — Deep Q-Network (DQN).
- Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — PPO, largement utilisé y compris pour le RLHF.
- Silver et al. (2016), [Mastering the game of Go with deep neural networks and tree search](https://doi.org/10.1038/nature16961), *Nature* — AlphaGo.
- Christiano et al. (2017), [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — fondements du RLHF.
- Ouyang et al. (2022), [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT et alignement par préférences.

## MLOps et mise en production

- Sculley et al. (2015), [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html), NeurIPS — dette technique propre aux systèmes ML.
- Breck et al. (2017), [The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) — tests d'un système d'apprentissage.
- Google, [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) — bonnes pratiques d'ingénierie ML.
- Google Cloud, [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — niveaux de maturité MLOps.
- Scikit-Learn, [Model persistence](https://scikit-learn.org/stable/model_persistence.html) — sérialisation des modèles et pièges de version.
- MLflow, [Documentation](https://mlflow.org/docs/latest/index.html) — suivi d'expériences et registre de modèles.

## Éthique, sécurité et régulation

- Barocas, Hardt et Narayanan (2023), [Fairness and Machine Learning: Limitations and Opportunities](https://fairmlbook.org/) — biais, équité et résultats d'impossibilité (libre d'accès).
- Hardt, Price et Srebro (2016), [Equality of Opportunity in Supervised Learning](https://arxiv.org/abs/1610.02413) — définitions d'équité et compromis.
- Union européenne, [Règlement (UE) 2024/1689 (AI Act)](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — approche de l'IA par niveaux de risque.
- Union européenne, [Règlement (UE) 2016/679 (RGPD)](https://eur-lex.europa.eu/eli/reg/2016/679/oj) — protection des données personnelles.
- Dwork et Roth (2014), [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) — confidentialité différentielle.
- Strubell, Ganesh et McCallum (2019), [Energy and Policy Considerations for Deep Learning in NLP](https://arxiv.org/abs/1906.02243) — impact environnemental des modèles.

*Voir aussi, dans les sections ci-dessus, le NIST AI Risk Management Framework, les Model Cards (Mitchell et al.) et les Datasheets for Datasets (Gebru et al.), également mobilisés par les modules 7 et 8.*
