# Module 7 : MLOps et Ingénierie de la Mise en Production

> Un modèle qui obtient un excellent score dans un notebook n'a créé **aucune valeur** tant qu'il ne sert pas une décision de façon fiable, surveillée et maintenable. Le MLOps est l'ensemble des pratiques qui transforment une expérience reproductible en **système vivant** : versionné, testé, déployé progressivement, observé et réparable. Ce module consolide et approfondit les sections « cycle de vie » (module 2), « déploiement » (module 3) et « observabilité » (module 4).

**Objectifs du module.** À l'issue de ce chapitre, vous saurez rendre une expérience reproductible et traçable, organiser des pipelines de données et de features sans fuite, gérer un registre de modèles et un déploiement progressif, choisir un mode de service adapté à la latence, mettre en place une surveillance à plusieurs niveaux (service, données, modèle, impact), réagir à une dérive et concevoir des tests spécifiques aux systèmes d'apprentissage.

**Prérequis.** Modules 1 (contrats de données, dérive), 2 (validation, cycle de vie), 3 (artefacts à versionner) et notions Git/CLI élémentaires. Le module 4 traite la mise en production spécifique aux agents ; le module 8 traite gouvernance et conformité.

---

## 📖 Le Dico du Débutant (Jargon Buster)
- **MLOps** : « DevOps pour le machine learning » — automatiser et fiabiliser le cycle de vie d'un modèle, données comprises.
- **Pipeline** : une suite d'étapes automatisées (préparer → entraîner → évaluer → déployer) reproductible d'un bout à l'autre.
- **Artefact** : tout objet produit et versionné (modèle sérialisé, jeu de données, rapport, image de conteneur).
- **Registre de modèles (*model registry*)** : le catalogue versionné des modèles, avec leur stade (staging, production, archivé).
- **Serving** : exposer un modèle pour qu'il réponde à des requêtes (par lot ou en direct).
- **Dérive (*drift*)** : l'évolution dans le temps des données ou de la relation apprise, qui peut dégrader les performances.
- **CI/CD** : intégration et livraison continues — tester et déployer automatiquement à chaque changement.
- **Rollback** : revenir à une version antérieure stable en cas de problème.

---

## Table des Matières
1. [Pourquoi le MLOps : le Fossé entre Notebook et Production](#1-pourquoi-le-mlops--le-fossé-entre-notebook-et-production)
2. [Reproductibilité et Versioning](#2-reproductibilité-et-versioning)
3. [Suivi d'Expériences](#3-suivi-dexpériences)
4. [Pipelines de Données et de Features](#4-pipelines-de-données-et-de-features)
5. [Empaquetage, Registre de Modèles et CI/CD](#5-empaquetage-registre-de-modèles-et-cicd)
6. [Servir un Modèle : Batch, En Ligne, Streaming](#6-servir-un-modèle--batch-en-ligne-streaming)
7. [Surveillance et Observabilité](#7-surveillance-et-observabilité)
8. [Réentraînement, Incidents et Retour Arrière](#8-réentraînement-incidents-et-retour-arrière)
9. [Tester un Système d'Apprentissage](#9-tester-un-système-dapprentissage)
10. [Coût, Passage à l'Échelle et Efficacité](#10-coût-passage-à-léchelle-et-efficacité)
11. [Checklist et Questions de Compréhension](#11-checklist-et-questions-de-compréhension)

---

## 1. Pourquoi le MLOps : le Fossé entre Notebook et Production

Un projet d'IA échoue rarement sur l'algorithme ; il échoue souvent sur tout **autour** : données qui changent, dépendances qui cassent, modèle impossible à redéployer, absence de surveillance.

### 🏗️ L'Analogie du Prototype et de l'Usine

Un plat réussi une fois dans votre cuisine n'est pas un restaurant. Servir 500 couverts par jour, avec une qualité constante, des ingrédients qui varient, du personnel qui change et des normes d'hygiène, exige des **processus** : recettes écrites, contrôle des stocks, chaîne du froid, retours clients. Le MLOps joue ce rôle pour un modèle.

Un système de ML est particulier car il dépend de **trois** choses mouvantes, pas une :

```text
Logiciel classique :   CODE
Système ML :           CODE  +  DONNÉES  +  MODÈLE
```

Chacune peut changer indépendamment et casser le résultat. La **dette technique cachée** du ML vient de ce couplage : un changement de distribution des données (module 1) peut dégrader un modèle sans qu'une seule ligne de code n'ait bougé.

---

## 2. Reproductibilité et Versioning

Règle fondatrice : **on doit pouvoir reconstruire à l'identique** un résultat passé. Cela exige de versionner **quatre** choses, pas seulement le code.

| À versionner | Comment | Pourquoi |
| :--- | :--- | :--- |
| **Code** | Git (commits, branches, tags) | tracer chaque changement |
| **Données** | empreinte/hachage, snapshot, versioning de jeu | un modèle dépend de *ses* données exactes |
| **Environnement** | dépendances épinglées, conteneur | éviter « ça marchait chez moi » |
| **Modèle + config** | artefact sérialisé + hyperparamètres + seuil | rejouer et comparer |

- **Épingler les dépendances** : des bornes majeures évitent qu'une rupture d'API rende un pipeline non reproductible (c'est exactement la logique du `requirements.txt` du cursus). Un **conteneur** (image reproductible du système) va plus loin en figeant l'OS et les bibliothèques.
- **Graines aléatoires** : facilitent le débogage sans garantir une reproductibilité parfaite entre matériels/versions (rappel modules 2, 3).
- **Le paquet déployé** n'est pas qu'un fichier de poids : il inclut prétraitement, ordre des classes, seuil, schéma d'entrée et métadonnées (rappel module 3, §18.3). Un modèle chargé avec le mauvais prétraitement **s'exécute sans erreur mais produit des résultats faux**.

> 💡 **Test de reproductibilité.** Un·e collègue, sur une autre machine, à partir du dépôt seul, doit pouvoir régénérer le modèle et retrouver des métriques cohérentes. Si ce n'est pas le cas, il manque un artefact versionné.

---

## 3. Suivi d'Expériences

Entraîner des dizaines de variantes sans traçabilité mène au chaos (« quel réglage donnait 0,82 déjà ? »). Le **suivi d'expériences** enregistre, pour chaque run :

- les **hyperparamètres** et la configuration ;
- la **version** du code, des données et de l'environnement ;
- les **métriques** (train/validation, par sous-groupe) et les courbes ;
- les **artefacts** produits (modèle, figures, rapport).

Cela permet de **comparer loyalement** (mêmes plis, mêmes données), de reproduire le meilleur run, et de documenter *pourquoi* un modèle a été retenu. C'est la mémoire d'ingénierie qui complète la **fiche modèle** (*model card*, module 2, §14).

> ⚠️ **Comparaison loyale.** Ne comparez pas un run évalué sur un split et un autre sur un split différent. Fixez le protocole avant, et rapportez l'incertitude : un écart de 0,2 point peut n'être que du bruit (rappel module 2, §11.2).

---

## 4. Pipelines de Données et de Features

Le modèle n'est qu'un maillon ; la **donnée** est le maillon fragile.

### 4.1 Automatiser sans fuite

Toute transformation apprise (imputation, scaling, encodage, sélection) doit être **ajustée sur l'entraînement uniquement** et rejouée à l'identique en production — la `Pipeline` Scikit-Learn matérialise cette frontière (rappel modules 1 et 2). Un pipeline de production doit :

- valider le **schéma** et la **qualité** en entrée (contrats de données, module 1, §10) ;
- appliquer **exactement** les mêmes transformations qu'à l'entraînement ;
- journaliser volumes, taux de manquants, catégories inconnues et fraîcheur.

### 4.2 Le problème train/serving skew

> 🔍 **Le piège n°1 en production.** Le **décalage entraînement/service** (*train/serving skew*) survient quand la donnée calculée en production diffère de celle de l'entraînement : unités différentes, valeur manquante traitée autrement, feature calculée avec une logique légèrement différente, ou **feature indisponible au moment réel de la prédiction** (fuite temporelle, module 1, §7.2). Le modèle se dégrade silencieusement. Parade : **partager le même code** de transformation entre entraînement et service.

### 4.3 Feature store (magasin de features)

Un **feature store** centralise le calcul, le stockage et la distribution des features, avec un objectif clé : garantir la **cohérence** entre l'entraînement (features historiques) et le service (features fraîches), et permettre leur réutilisation entre projets. Il n'est pas obligatoire pour un petit projet, mais devient utile quand plusieurs modèles partagent des features en temps réel.

---

## 5. Empaquetage, Registre de Modèles et CI/CD

### 5.1 Empaqueter le modèle

Sérialiser le modèle **et** son contexte (prétraitement, seuil, schéma, version), puis l'encapsuler derrière une **interface stable** (une fonction `predict` claire, un service). Documenter les formes d'entrée/sortie et les erreurs attendues (même esprit que les contrats d'outils du module 4, §9.2).

### 5.2 Registre de modèles

Le **registre** catalogue les versions et leur **stade** :

```text
entraîné → validé (staging) → production → archivé
                 ▲                  │
                 └──── rollback ◄────┘
```

Chaque version porte : métriques, jeu d'évaluation, données/lignage, auteur, date, limites et usages interdits. La **promotion** vers la production est une décision explicite, pas un écrasement silencieux.

### 5.3 CI/CD pour le ML

L'**intégration continue** exécute automatiquement, à chaque changement : linting, tests unitaires, exécution du pipeline sur un échantillon, et contrôles de qualité (c'est l'esprit des scripts `validate_markdown.py` / `validate_notebooks.py` du cursus, qui font échouer la chaîne dès qu'une cellule échoue). La **livraison continue** déploie l'artefact validé.

Spécificités par rapport au logiciel classique :
- ajouter des **tests de données** (schéma, distributions) et des **seuils de métriques** (bloquer si la performance chute sous une garde) ;
- versionner et rejouer sur des **données de référence** ;
- prévoir un déploiement **progressif** (§8) plutôt qu'un basculement brutal.

---

## 6. Servir un Modèle : Batch, En Ligne, Streaming

| Mode | Principe | Bon usage | Contrainte clé |
| :--- | :--- | :--- | :--- |
| **Batch** | prédire périodiquement sur des lots | scoring nocturne, recommandations pré-calculées | fraîcheur limitée |
| **En ligne (temps réel)** | répondre à la requête à la demande | détection de fraude, réponse interactive | latence, disponibilité |
| **Streaming** | traiter un flux continu d'événements | télémétrie, IoT, alertes | ordre, fenêtres, reprise |
| **Embarqué (*edge*)** | modèle sur l'appareil | hors ligne, vie privée, latence | mémoire, énergie, mises à jour |

Points d'ingénierie transverses :
- **latence** : mesurer la **médiane et les percentiles élevés** (p95/p99), pas seulement la moyenne — et sur le **pipeline entier**, prétraitement compris (rappel modules 2 et 3) ;
- **débit et coût** : le *batching* et le cache améliorent le débit mais modifient latence et mémoire ;
- **disponibilité** : redondance, dégradation gracieuse, système de secours (une baseline simple) si le modèle est indisponible ;
- **compatibilité** : versionner l'API ; un changement de schéma d'entrée est un changement contractuel.

---

## 7. Surveillance et Observabilité

Un modèle déployé **se dégrade** avec le temps : le monde change, mais le modèle est figé. La surveillance suit **quatre niveaux** (rappel module 2, §14) :

```text
1. SERVICE   → disponibilité, latence (p50/p95/p99), taux d'erreur, coût
2. DONNÉES   → qualité, fraîcheur, schéma, catégories inconnues, dérive de P(X)
3. MODÈLE    → performance quand les labels arrivent, calibration, dérive de P(y|X)
4. IMPACT    → bénéfice métier, effets indésirables, équité par sous-groupe
```

### 7.1 Détecter la dérive

Rappel des trois dérives (module 1, §12) : entrées $P(X)$, cible $P(y)$, concept $P(y\mid X)$. On surveille des distances statistiques et des indicateurs opérationnels.

> ⚠️ **Une distance statistique ne prouve pas une baisse de performance.** Une dérive de $P(X)$ peut être inoffensive si la relation apprise tient toujours. Utilisez la dérive comme **déclencheur d'enquête**, pas comme preuve, et confirmez avec la performance réelle dès que les labels arrivent.

### 7.2 Le problème des labels tardifs

Souvent, la vérité terrain n'arrive que bien plus tard (un défaut de paiement se constate des mois après). En attendant, on surveille des **proxys** (dérive des entrées, distribution des scores, taux d'alerte) avec prudence, puis on **réconcilie** avec la vérité quand elle arrive (rappel module 1, §12).

### 7.3 Traces et alertes

Journaliser entrées (expurgées), sorties, version du modèle, latence et décisions permet le diagnostic. Définir des **alertes** avec seuil, propriétaire et procédure — et éviter la fatigue d'alerte (trop d'alertes = plus aucune lue), même logique que la fatigue d'approbation du module 4, §13.4.

---

## 8. Réentraînement, Incidents et Retour Arrière

### 8.1 Réentraîner avec discernement

> ⚠️ **Ne réentraînez pas automatiquement sur toute donnée récente** (rappel module 2, §14). Les données récentes peuvent être de mauvaise qualité, biaisées par une panne, ou empoisonnées par une boucle de rétroaction (le modèle influence les données futures, module 1). Un réentraînement doit être **validé** comme un nouveau modèle : mêmes contrôles, mêmes gardes de métriques, même comparaison à la version en place.

Déclencheurs raisonnables : dégradation confirmée de performance, dérive persistante corrélée à l'impact, disponibilité de nouvelles données de qualité, changement métier.

### 8.2 Déploiement progressif

Ne basculez jamais 100 % du trafic d'un coup (rappel modules 3 et 4) :

```text
hors ligne → shadow (le modèle prédit sans agir, on compare)
          → canary (petit % du trafic)
          → montée progressive conditionnée aux métriques
          → 100 %
```

- **shadow mode** : le nouveau modèle tourne en parallèle sans influencer les décisions ; on compare ses sorties à la production.
- **canary / A-B** : un petit groupe reçoit le nouveau modèle ; on mesure avant d'élargir.

### 8.3 Plan d'incident

Défini **à l'avance** : seuils d'alerte, propriétaire, diagnostic, **interrupteur d'arrêt**, retour à la version stable (**rollback**), communication et post-mortem. Un système sans rollback testé est un système fragile.

---

## 9. Tester un Système d'Apprentissage

Les tests logiciels classiques ne suffisent pas ; on ajoute des tests propres au ML (esprit proche du module 4, §14.4) :

- **unitaires** : fonctions de préparation, features, post-traitement ;
- **de données** : schéma, bornes, taux de manquants, catégories attendues (contrats, module 1) ;
- **d'invariance / comportementaux** : une variation censée ne rien changer ne doit pas changer la prédiction (ex. un identifiant), une variation censée compter doit compter ;
- **de non-régression** : un jeu de référence dont la métrique ne doit pas chuter sous un seuil ;
- **de robustesse** : entrées aux limites, catégories inconnues, valeurs manquantes, décalage (rappel module 2, §13.2) ;
- **du modèle compressé** : revalider après quantification/pruning/distillation, pas seulement l'original (rappel module 3, §18.2).

---

## 10. Coût, Passage à l'Échelle et Efficacité

La performance n'est pas la seule dimension ; un modèle « meilleur » mais trop cher ou trop lent peut ne pas valoir la peine (rappel module 2, §7 : « améliore-t-il la décision, à coût et risque acceptables ? »).

- **Coût par prédiction** : calcul, mémoire, stockage, appels externes ; à comparer au **gain métier**.
- **Optimisation d'inférence** : quantification, pruning, distillation, batching, cache (module 3, §18.2) — chacune peut dégrader certaines entrées, donc revalidation obligatoire.
- **Scalabilité** : dimensionner selon la charge réelle (percentiles), prévoir montée et descente en charge.
- **Sobriété** : mesurer et réduire l'empreinte énergétique fait partie d'une ingénierie responsable (approfondi au module 8).
- **Simplicité** : à performance équivalente, le système le plus simple, disponible et maintenable est souvent préférable (rappel module 1, §16.4).

---

## 11. Checklist et Questions de Compréhension

### Checklist de mise en production

- [ ] Code, données, environnement et modèle+config sont versionnés et rejouables.
- [ ] Les expériences sont tracées et comparées sur un protocole fixe, avec incertitude.
- [ ] Le même code de transformation est utilisé à l'entraînement et au service (pas de skew).
- [ ] Chaque feature est disponible au moment réel de la prédiction (pas de fuite temporelle).
- [ ] Le modèle est empaqueté avec prétraitement, seuil, schéma et métadonnées.
- [ ] Un registre gère les stades et la promotion explicite en production.
- [ ] La CI exécute tests de code, de données et gardes de métriques.
- [ ] Le mode de service correspond à la contrainte de latence/fraîcheur.
- [ ] Les quatre niveaux (service, données, modèle, impact) sont surveillés.
- [ ] Déploiement progressif, interrupteur d'arrêt et rollback sont testés.
- [ ] Le réentraînement est validé, jamais automatique et aveugle.

### Questions de compréhension

1. Pourquoi versionne-t-on les données et l'environnement, et pas seulement le code ?
2. Qu'est-ce que le *train/serving skew* et comment l'éviter ?
3. Pourquoi un modèle chargé avec le mauvais prétraitement est-il particulièrement dangereux ?
4. Une dérive de $P(X)$ implique-t-elle toujours une baisse de performance ? Que faire ?
5. Que surveille-t-on quand les labels arrivent avec des mois de retard ?
6. Différence entre *shadow mode* et *canary* ; pourquoi ne pas basculer 100 % d'un coup ?
7. Pourquoi ne faut-il pas réentraîner automatiquement sur toute donnée récente ?
8. Citez deux tests spécifiques au ML absents du logiciel classique.

**Mini-étude de cas.** Vous devez mettre en production un modèle de scoring de crédit utilisé en temps réel au moment de la demande. (a) Listez les artefacts à versionner. (b) Choisissez un mode de service et justifiez la contrainte de latence. (c) Décrivez la surveillance aux quatre niveaux et un plan de dérive. (d) Détaillez un déploiement progressif avec critères de promotion et procédure de rollback. (e) Identifiez une feature à risque de fuite temporelle et comment la contrôler.

---

Pour approfondir et vérifier ces pratiques, consultez les documentations et référentiels regroupés dans [REFERENCES.md](../REFERENCES.md).
