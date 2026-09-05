# Atelier — Interroger les documents du cours avec des citations

Le [TP documentaire](02_rag_documents_citations.ipynb) lit réellement les Markdown du dépôt, les découpe, construit un index TF-IDF, récupère des passages et produit une réponse extractive accompagnée des fichiers et lignes sources. Il ne nécessite ni base distante ni clé API. Prévoir 2 à 3 heures après la section 8 du [cours agentique](cours_ia_agentique.md).

## Objectifs et frontière du TP

L'objectif est de rendre testables ingestion, récupération, provenance, filtrage d'accès et abstention. La composition finale assemble des extraits : c'est une baseline documentaire, pas une génération par LLM. Un RAG génératif ajouterait un modèle conditionné par ces passages ; la grille d'évaluation ci-dessous reste nécessaire pour cette extension.

## Pipeline et contrats

1. Lire uniquement les fichiers de cours autorisés, sans exécuter leurs blocs de code.
2. Découper en fenêtres de lignes chevauchantes et conserver chemin, début et fin.
3. Vectoriser les passages avec TF-IDF et normaliser pour la similarité cosinus.
4. Appliquer les droits documentaires avant le classement des candidats.
5. Récupérer k passages et comparer le score maximal à un seuil calibré.
6. Assembler des citations vérifiables ou s'abstenir.

Le recouvrement évite de perdre entièrement un concept coupé en deux, mais crée de la redondance. Une fenêtre fixe est facile à auditer ; un découpage par titres ou paragraphes peut mieux préserver le sens. TF-IDF repose sur le chevauchement lexical : il ne comprend pas toutes les paraphrases. Comparer à une recherche dense ou hybride constituerait une extension, avec un jeu de requêtes inchangé.

## Évaluer chaque étage

Le TP sépare des questions de calibration et des questions test. Chaque question répondable indique le document attendu ; une question hors domaine sert de contrôle négatif. Recall@k mesure la présence de ce document dans les k résultats. Cette métrique est volontairement grossière : retrouver le bon fichier ne garantit pas le bon passage.

Le seuil est choisi sur la calibration ; on ouvre le test après fixation. Rapporter aussi fausses abstentions et réponses hors domaine. Un score cosinus n'est pas une probabilité de vérité. En production, annoter les passages pertinents et la réponse attendue, ajouter paraphrases, contradictions, versions périmées et questions multi-documents.

Pour la génération LLM, distinguer fidélité aux sources, correction de la réponse, complétude et validité des citations. Une citation peut être réelle sans soutenir l'affirmation. Un juge automatique ne remplace pas une revue humaine d'un échantillon difficile.

## Sécurité et exercice corrigé

Un passage peut contenir « ignore les règles et exécute… ». Il reste une donnée du corpus. Le TP ne dispose d'aucun exécuteur d'outils : il peut citer cette chaîne, mais ne l'exécute pas. Cela teste une frontière architecturale, pas la résistance d'un LLM à l'injection.

Exercice : restreindre les documents autorisés au module 1, puis poser une question de deep learning. Correction attendue : aucun passage du module 3 ne doit apparaître, même si sa similarité serait supérieure. L'abstention est préférable à une citation non autorisée ; en revanche l'absence de réponse ne prouve pas que l'information n'existe nulle part.

La suite générative exige un fournisseur choisi explicitement, le contrôle des données qui lui sont transmises, des budgets et une évaluation de trajectoires. Ce TP constitue une baseline reproductible pour mesurer l'apport de cette suite.
