# Références et documentations de vérification

Cette bibliographie privilégie les articles fondateurs, les documentations officielles et les référentiels de sécurité. Les bibliothèques évoluent : vérifier la documentation correspondant à la version installée.

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
