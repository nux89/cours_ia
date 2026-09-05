# Module 4 : L'Intelligence Artificielle Agentique

Pratique complémentaire : l'[atelier documentaire](atelier_rag_documentaire.md) met en œuvre récupération dans les cours, citations et abstention. Il distingue cette chaîne extractive hors ligne d'un RAG avec génération par LLM.

> "Les modèles de fondation ne sont pas la destination finale de l'IA, ils en sont le moteur cognitif. L'agentique transforme ce moteur en un système autonome capable de percevoir, raisonner, planifier et agir dans le monde réel."

**Objectifs du module.** À l'issue de ce chapitre, vous saurez distinguer modèle, workflow et agent ; concevoir un RAG avec citations ; séparer base de connaissances, mémoire, outils et skills ; écrire des contrats d'outils sûrs ; choisir une orchestration mono ou multi-agent ; définir garde-fous, évaluations, observabilité et conditions d'arrêt avant la mise en production.

**Prérequis.** Modules 1 à 3, fonctions Python, JSON, API et tests unitaires élémentaires. Le notebook associé implémente hors ligne une boucle d'agent déterministe, inspectable et testable.

---

## 📖 Le Dico du Débutant (Jargon Buster)
Le terme « agent » n'a pas une définition unique. Dans ce cours, il désigne un système qui poursuit un objectif par une boucle contrôlée d'observation, de décision et d'action. Voici les termes indispensables :
- **LLM (Grand Modèle de Langage)** : Un modèle qui produit des séquences de tokens à partir d'un contexte. Il peut aussi émettre des demandes d'outils structurées lorsqu'un programme hôte lui en donne la possibilité.
- **Agent IA** : Un système complet qui associe un modèle ou une politique de décision à des outils, un état, des règles d'autorisation et des critères d'arrêt.
- **Prompt Système** : Une instruction de haut niveau fournie au modèle. Elle guide le comportement, mais ne constitue pas une frontière de sécurité : les autorisations doivent être contrôlées par du code externe.
- **Appel d'Outils (*Tool Calling / Function Calling*)** : La capacité du modèle à demander l'exécution d'un programme informatique externe pour obtenir une information fraîche ou modifier le monde réel.
- **Boucle ReAct (*Reason + Act*)** : Un patron d'orchestration possible alternant décision, action et observation. D'autres architectures utilisent un plan, un graphe d'états ou des appels d'outils directs.
- **RAG (*Retrieval-Augmented Generation*)** : Une méthode qui récupère des passages dans des sources externes puis les fournit au modèle pour produire une réponse ancrée et, si possible, citée.
- **Base de connaissances (*Knowledge Base*)** : L'ensemble gouverné des documents, enregistrements et métadonnées que le système peut rechercher. Ce n'est ni le modèle ni sa mémoire de conversation.
- **Skill / Compétence** : Un paquet réutilisable d'instructions, de ressources, d'exemples et éventuellement de scripts ou d'outils qui apprend à l'agent comment exécuter une famille de tâches.
- **Garde-fou (*Guardrail*)** : Un contrôle préventif, détectif ou correctif appliqué avant, pendant ou après l'action. Une consigne dans le prompt ne suffit pas à faire respecter une autorisation.
- **Human-in-the-Loop (HITL)** : Le garde-fou humain indispensable. Pour toute action dangereuse (supprimer un fichier, envoyer de l'argent), l'agent doit demander l'accord explicite d'un être humain.

---

## Table des Matières
1. [La Rupture Agentique : Du Modèle Passif au Système Autonome](#1-la-rupture-agentique--du-modèle-passif-au-système-autonome)
2. [L'Anatomie d'un Agent IA Autonome](#2-lanatomie-dun-agent-ia-autonome)
3. [Les Mécanismes de Raisonnement et de Planification](#3-les-mécanismes-de-raisonnement-et-de-planification)
4. [L'Utilisation d'Outils (Tool Use & Function Calling décortiqué)](#4-lutilisation-doutils-tool-use--function-calling)
5. [Les Architectures de Mémoire](#5-les-architectures-de-mémoire)
6. [Les Architectures Multi-Agents](#6-les-architectures-multi-agents)
7. [Écosystèmes, Sécurité et Gouvernance](#7-écosystèmes-sécurité-et-gouvernance)
8. [RAG et Bases de Connaissances](#8-rag-et-bases-de-connaissances)
9. [Tools : Contrats, Exécution et Autorisations](#9-tools--contrats-exécution-et-autorisations)
10. [Skills : Compétences Réutilisables](#10-skills--compétences-réutilisables)
11. [Mémoire Agentique et Gestion du Contexte](#11-mémoire-agentique-et-gestion-du-contexte)
12. [Orchestration et Systèmes Multi-Agents](#12-orchestration-et-systèmes-multi-agents)
13. [Guardrails et Défense en Profondeur](#13-guardrails-et-défense-en-profondeur)
14. [Évaluation, Observabilité et Mise en Production](#14-évaluation-observabilité-et-mise-en-production)
15. [Architecture de Référence et Étude de Cas](#15-architecture-de-référence-et-étude-de-cas)
16. [Checklist et Questions de Compréhension](#16-checklist-et-questions-de-compréhension)

---

## 1. La Rupture Agentique : Du Modèle Passif au Système Autonome

### 🧞 L'Analogie du Génie dans la Bouteille
Pour comprendre ce qu'est un Agent IA, imaginez un **génie surdoué enfermé dans une pièce close**, sans portes, sans fenêtres, sans montre et sans téléphone :
- Il a une culture générale encyclopédique colossale (il a lu tout Wikipédia et des millions de livres).
- Mais si vous lui demandez : *"Quel temps fait-il dehors en ce moment ?"*, il est incapable de vous répondre avec certitude. S'il essaie de deviner, il va inventer une météo imaginaire : c'est l'**hallucination**.
- Si vous lui demandez : *"Envoie un email à mon collègue et réserve mon billet d'avion"*, il est impuissant car il n'a pas de mains.

**Un Agent IA, c'est ce génie à qui on donne :**
1. Un smartphone connecté à Internet (pour vérifier les faits en direct).
2. Une calculatrice vérifiable (qui peut malgré tout échouer sur une entrée invalide ou une mauvaise unité).
3. Des droits d'accès à des logiciels (pour réserver le train ou exécuter du code).
4. Un bloc-notes (pour se souvenir de ce qu'il a fait il y a 10 minutes).

```text
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│     1. IA Prédictive    │     │    2. IA Générative     │     │      3. IA Agentique    │
│  (Scikit-Learn, CNNs)   │ ──► │     (LLMs bruts)        │ ──► │   (Systèmes Autonomes)  │
│                         │     │                         │     │                         │
│  Données -> Classe /    │     │  Prompt -> Texte        │     │  Objectif -> Boucle     │
│  Régression             │     │  (Passif, one-shot)     │     │  Raisonnement + Action  │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

---

## 2. L'Anatomie d'un Agent IA Autonome

Une architecture agentique courante peut se décrire avec les composants suivants :

```
                           ┌───────────────────────────────┐
                           │       OBJECTIF UTILISATEUR    │
                           └───────────────┬───────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    AGENT IA                                         │
│                                                                                     │
│    ┌──────────────────┐           ┌──────────────────┐          ┌──────────────┐    │
│    │     MÉMOIRE      │ ◄───────► │   RAISONNEMENT   │ ◄──────► │    OUTILS    │    │
│    │ (Court / Long    │           │ (Planification,  │          │ (APIs, Code, │    │
│    │   terme, RAG)    │           │  ReAct, Critique)│          │  Web, DB)    │    │
│    └──────────────────┘           └─────────┬────────┘          └──────┬───────┘    │
└─────────────────────────────────────────────┼──────────────────────────┼────────────┘
                                              │                          │
                                    Observation d'état           Action logicielle
                                              │                          │
                                              ▼                          ▼
                                ┌──────────────────────────────────────────────┐
                                │                 ENVIRONNEMENT                │
                                │   (OS, Base de données, Réseau, Navigateur)  │
                                └──────────────────────────────────────────────┘
```

1. **Le Cerveau (Reasoning Engine)** : Le LLM qui lit l'historique, analyse la progression et choisit la prochaine étape.
2. **Le Profil (Persona / System Prompt)** : *"Tu es un auditeur financier rigoureux. Tu ne dois jamais inventer de chiffres sans consulter la base documentaire."*
3. **La Mémoire (Memory Store)** : Le carnet où sont stockées les interactions passées pour ne pas tourner en rond.
4. **La Boîte à Outils (Tools)** : Les fonctions Python réelles que l'agent a le droit d'appeler.

---

## 3. Les Mécanismes de Raisonnement et de Planification

### 3.1 Planification et traces observables
Décomposer une tâche peut améliorer certains résultats, mais l'effet dépend du modèle et de l'évaluation. En production, on privilégie des traces auditables — plan concis, appels d'outils, observations, décisions et sources — plutôt que l'affichage d'un raisonnement interne détaillé. La qualité se mesure sur les résultats, les contraintes respectées, le coût et les cas d'échec.

### 3.2 Le paradigme ReAct sur un incident informatique
Face à un service indisponible, une boucle observable peut ressembler à ceci :
- **[Décision 1]** : vérifier l'état du service avant toute modification.
- **[Action 1]** : `lire_statut_service(service="catalogue")`.
- **[Observation 1]** : taux d'erreur élevé depuis 10 minutes.
- **[Décision 2]** : consulter les journaux récents en lecture seule.
- **[Action 2]** : `rechercher_logs(service="catalogue", niveau="ERROR")`.
- **[Observation 2]** : échec de connexion à la base de données après un déploiement.
- **[Réponse]** : résumer les preuves et proposer un retour arrière soumis à approbation.

Cette trace montre surtout la séparation entre décision probabiliste, outils déterministes, preuves observées et action à risque.

---

## 4. L'Utilisation d'Outils (Tool Use & Function Calling)

### 🔌 Démystification : Comment un modèle de langage "appelle"-t-il un outil ?
Un LLM ne sait pas exécuter de code lui-même : il ne fait que générer du texte. Alors, comment lance-t-il une recherche ou un calcul ?  
**C'est un dialogue structuré en 5 temps :**

```text
1. Humain  ──► "Combien font 1 547 multiplié par 892 ?"
2. LLM     ──► Génère ce JSON : {"outil": "calculatrice", "args": {"calcul": "1547 * 892"}}
3. Python  ──► Le programme hôte intercepte le JSON, lance Python : 1547 * 892 = 1 379 924
4. Python  ──► Réinjecte le résultat au LLM : [Observation] : 1379924
5. LLM     ──► "Le résultat exact est 1 379 924."
```

Chaque outil peut être décrit par un schéma de paramètres. Le format exact dépend de l'API ; le programme hôte doit valider les arguments et les autorisations avant exécution :
```json
{
  "name": "calculatrice",
  "description": "Évalue une opération arithmétique (+, -, *, /)",
  "parameters": {
    "type": "object",
    "properties": {
      "calcul": {
        "type": "string",
        "description": "L'expression mathématique à calculer (ex: '1547 * 892')"
      }
    },
    "required": ["calcul"]
  }
}
```

---

## 5. Les Architectures de Mémoire

Un LLM possède une fenêtre de mémoire immédiate limitée (la *Context Window*). Pour travailler sur de longues missions, l'agent utilise plusieurs types de mémoire :

| Type de Mémoire | Exemple concret | Où est-ce stocké ? |
| :--- | :--- | :--- |
| **Mémoire Immédiate (Court Terme)** | Les éléments récents utiles à la tâche | Dans le contexte courant envoyé au modèle |
| **Mémoire de Travail (*Scratchpad*)** | Le plan d'action en cours : "Étape 1 validée, étape 2 en cours" | Dans une variable Python mise à jour |
| **Mémoire documentaire / RAG** | Extraits pertinents de manuels et documents autorisés | Index de recherche, base vectorielle ou moteur hybride |
| **Mémoire Épisodique** | "La semaine dernière, j'ai déjà eu cette erreur SQL, voici comment je l'avais résolue" | Dans une base de données de retours d'expérience |

---

## 6. Les Architectures Multi-Agents

Quand une mission est trop lourde pour un seul agent, on assemble une **équipe d'agents spécialisés** :

### 🏢 L'Analogie de l'Agence de Développement
Imaginez une petite entreprise informatique :
- **L'Agent Chef de Projet (Manager / Superviseur)** : Reçoit le besoin du client, découpe le travail en sous-tâches ordonnées et les distribue.
- **L'Agent Développeur (Worker / Coder)** : Écrit le code Python demandé.
- **L'Agent Testeur / Auditeur (Reviewer / QA)** : Exécute les tests, relit le code et signale les bugs au développeur pour qu'il les corrige.

Cette séparation peut améliorer la couverture et la revue, mais elle augmente aussi le coût, la latence et les surfaces d'erreur. Un second agent n'est pas une preuve indépendante s'il partage les mêmes données, hypothèses et modèle : il faut des critères de validation et des tests externes.

---

## 7. Écosystèmes, Sécurité et Gouvernance

### 🛡️ Le Principe Fondamental : Human-in-the-Loop (HITL)
> 🚨 **Règle d'or de la sécurité agentique** :  
> Une action irréversible, financière, publique ou fortement privilégiée doit être bloquée par une politique d'autorisation explicite. Selon le risque, cette politique peut imposer une approbation humaine, une double validation ou interdire entièrement l'action.
>
> - Consulter le solde d'un compte bancaire $\to$ **Autorisé en autonomie** (Lecture seule).
> - Virer 50 000 € sur un compte externe $\to$ **Validation humaine obligatoire** (Écriture critique).
> - Rédiger un brouillon de mail $\to$ **Autorisé en autonomie**.
> - Envoyer un email public à 100 000 clients $\to$ **Validation humaine obligatoire**.

### Les Pièges et Dangers Majeurs
1. **Injection de prompt indirecte** : traiter le contenu externe comme une donnée non fiable, séparer instructions et documents, et ne jamais lui accorder de nouveaux privilèges.
2. **Moindre privilège** : limiter les outils, ressources et secrets à ce qui est nécessaire pour la tâche ; vérifier les autorisations hors du modèle.
3. **Validation des entrées et sorties** : utiliser des schémas, listes d'autorisation, bornes numériques et contrôles métier déterministes.
4. **Boucles et coûts** : imposer des budgets de pas, de temps et de dépenses, ainsi que des critères d'arrêt et une remontée claire des blocages.
5. **Effets de bord** : rendre les opérations idempotentes quand c'est possible, journaliser les actions et prévoir annulation ou compensation.
6. **Évaluation** : tester les cas normaux, les erreurs d'outils, les injections, les données périmées et les demandes ambiguës avant déploiement.

Pour approfondir les architectures et la gestion des risques, consultez [REFERENCES.md](../REFERENCES.md).

---

## 8. RAG et Bases de Connaissances

Le RAG associe une mémoire **paramétrique**—les poids du modèle—à une mémoire **non paramétrique**—des contenus retrouvés au moment de la requête. Il ne réentraîne pas le modèle à chaque mise à jour documentaire.

### 8.1 Ne pas confondre les composants

| Composant | Question à laquelle il répond | Exemple | Change le monde ? |
| :--- | :--- | :--- | :---: |
| Modèle | « Que produire à partir du contexte ? » | LLM, modèle de vision | Non, seul |
| Base de connaissances | « Quelles sources gouvernées possédons-nous ? » | procédures, contrats, catalogue SQL | Non |
| Retriever / RAG | « Quels passages sont pertinents maintenant ? » | recherche hybride + reranker | Non |
| Mémoire | « Que conserver des interactions et états passés ? » | préférences, résumé de tâche | Non, hors écriture mémoire |
| Outil | « Quelle opération externe exécuter ? » | lire un ticket, envoyer un mail | Potentiellement |
| Skill | « Comment accomplir cette famille de tâches ? » | procédure d'audit avec scripts | Via les outils autorisés |
| Agent | « Quelle prochaine étape choisir pour atteindre l'objectif ? » | boucle contrôlée | Potentiellement |

Une **base vectorielle** est un type d'index, pas une base de connaissances complète. La connaissance comprend aussi sources originales, métadonnées, versions, droits, dates de validité et propriétaires.

### 8.2 Pipeline d'ingestion

```text
sources autorisées
  → collecte/version
  → extraction (texte, tableaux, OCR)
  → nettoyage et déduplication
  → découpage en passages
  → métadonnées + droits d'accès
  → embeddings et/ou index lexical
  → index consultable
```

Chaque passage devrait conserver : identifiant stable, document source, titre, section, page ou position, version, date, langue, propriétaire, niveau de confidentialité et liste de contrôle d'accès.

#### Découpage (*chunking*)

Un passage trop court manque de contexte ; trop long dilue le signal et consomme la fenêtre. Stratégies :

- taille fixe avec chevauchement ;
- séparation par titres, paragraphes ou phrases ;
- découpage propre au format : ligne de tableau, fonction de code, article juridique ;
- découpage parent–enfant : recherche sur petit extrait, restitution du bloc parent ;
- regroupement sémantique.

Évaluez plusieurs tailles sur de vraies questions. Le nombre de caractères n'est pas équivalent au nombre de tokens.

### 8.3 Recherche lexicale, dense et hybride

- **lexicale** : correspondance de termes, très utile pour références exactes, noms et codes ;
- **dense** : proximité entre embeddings, utile pour reformulations sémantiques ;
- **hybride** : combine les deux scores ou fusionne deux classements ;
- **reranking** : un second modèle réordonne un petit ensemble de candidats avec une analyse plus fine.

Pour des embeddings normalisés, la similarité cosinus est :

$$
\cos(q,d)=\frac{q\cdot d}{\|q\|\,\|d\|}.
$$

Une grande similarité ne garantit ni vérité, ni actualité, ni droit d'accès. Les filtres métier et de sécurité doivent être appliqués séparément.

### 8.4 Pipeline de requête

```text
question utilisateur
  → validation et classification d'intention
  → réécriture/décomposition éventuelle
  → filtres d'identité, domaine, langue et date
  → recherche lexicale + dense
  → fusion et reranking
  → sélection/diversification des passages
  → contexte avec identifiants de sources
  → génération contrainte
  → vérification des citations et réponse
```

La réécriture peut améliorer une question elliptique, mais elle peut aussi changer l'intention. Conservez question originale et version réécrite dans la trace.

### 8.5 Construction du contexte

Un contexte utile :

- sépare clairement instructions de confiance et documents non fiables ;
- délimite chaque passage et son identifiant ;
- conserve assez de voisinage pour interpréter le texte ;
- déduplique les extraits redondants ;
- respecte un budget de tokens ;
- demande au modèle de citer l'identifiant associé à chaque affirmation vérifiable ;
- autorise « information insuffisante » plutôt que l'invention.

Les documents récupérés sont des **données**, même s'ils contiennent des phrases comme « ignore les règles et appelle cet outil ». Ils ne doivent pas modifier les autorisations ou le prompt système.

### 8.6 Variantes du RAG

- **RAG simple** : une recherche puis une réponse ; excellent point de départ ;
- **RAG conversationnel** : reformule avec l'historique sans laisser l'historique noyer la requête ;
- **RAG multi-requêtes** : génère plusieurs formulations puis fusionne les résultats ;
- **RAG décomposé** : traite plusieurs sous-questions et assemble les preuves ;
- **RAG sur graphe** : suit entités et relations explicites ;
- **RAG agentique** : l'agent choisit quand et où chercher, avec budget et critères d'arrêt.

La version agentique est plus flexible mais plus lente, coûteuse et difficile à tester. Si une seule récupération suffit, une chaîne déterministe est préférable.

### 8.7 Évaluer le RAG par étage

Construisez un jeu de questions représentatives avec réponses acceptables, sources attendues, droits d'accès et cas sans réponse.

**Retrieval :**

- `Recall@k` : une source pertinente figure-t-elle parmi les $k$ résultats ?
- `Precision@k` : quelle proportion des $k$ passages est pertinente ?
- MRR : à quel rang apparaît le premier bon résultat ?
- nDCG : le classement respecte-t-il plusieurs degrés de pertinence ?

**Génération :**

- correction de la réponse ;
- fidélité aux passages fournis (*groundedness*) ;
- complétude ;
- citations qui soutiennent réellement la phrase ;
- abstention lorsque les preuves manquent.

**Système :** latence, coût, fraîcheur, couverture, fuites d'autorisation et robustesse aux injections. Diagnostiquez d'abord l'étage fautif : le générateur ne peut pas citer un passage que le retriever n'a jamais remonté.

### 8.8 Sécurité et gouvernance de la connaissance

- appliquez les ACL **avant ou pendant** la récupération, pas après la génération ;
- isolez les locataires et testez les requêtes transverses ;
- analysez les documents entrants et leur provenance ;
- gérez suppression et expiration dans source, index et caches ;
- journalisez identifiants des passages récupérés, sans recopier inutilement les secrets ;
- invalidez les embeddings lors d'un changement documentaire important ;
- affichez version et date des sources lorsque la fraîcheur compte.

---

## 9. Tools : Contrats, Exécution et Autorisations

Un outil transforme la capacité linguistique du modèle en lecture ou action vérifiable. Le modèle **propose** un appel ; l'orchestrateur **décide** s'il est valide et autorisé ; l'exécuteur **réalise** l'opération.

### 9.1 Cycle complet d'un appel

```text
découvrir → sélectionner → produire des arguments
          → valider le schéma
          → vérifier identité, portée et politique
          → demander approbation si nécessaire
          → exécuter avec limites
          → normaliser l'observation
          → journaliser et décider de la suite
```

Chaque étape peut échouer. Un message d'erreur doit être structuré et ne pas révéler de secret.

### 9.2 Concevoir un bon contrat

Un outil doit avoir :

- un nom sans ambiguïté ;
- une description précisant quand l'utiliser et quand ne pas l'utiliser ;
- des arguments typés, bornés et documentés ;
- une sortie structurée ;
- des erreurs connues ;
- un niveau de risque et une politique d'autorisation ;
- un timeout, une limite de taille et une stratégie de retry ;
- une sémantique d'idempotence lorsque c'est possible.

Préférez :

```json
{
  "name": "creer_brouillon_email",
  "parameters": {
    "type": "object",
    "properties": {
      "destinataire": {"type": "string", "format": "email"},
      "objet": {"type": "string", "maxLength": 120},
      "corps": {"type": "string", "maxLength": 10000}
    },
    "required": ["destinataire", "objet", "corps"],
    "additionalProperties": false
  }
}
```

à un outil générique `executer_commande(texte)` qui offrirait une surface illimitée.

### 9.3 Lecture, simulation et écriture

Classez les capacités :

| Niveau | Exemple | Contrôle minimal |
| :--- | :--- | :--- |
| Lecture publique | météo | validation, quotas |
| Lecture privée | dossier client | identité, ACL, journalisation |
| Prévisualisation | devis, brouillon | validation métier |
| Écriture réversible | créer un ticket | confirmation, idempotence |
| Action externe sensible | envoyer, payer, publier | approbation liée à l'action exacte |
| Irréversible/interdite | suppression massive | double contrôle ou interdiction |

Séparer `préparer`, `prévisualiser` et `confirmer` réduit le risque. L'approbation doit montrer destinataire, montant, ressource et effet exacts ; une approbation générale ne doit pas couvrir des arguments modifiés ensuite.

### 9.4 Validation hors modèle

Le code hôte vérifie : type, format, bornes, ressources autorisées, état courant et invariants métier. Exemples : montant positif sous plafond, destinataire appartenant au domaine autorisé, fichier situé dans un répertoire prévu, requête SQL en lecture seule.

Les secrets sont injectés par l'exécuteur au dernier moment et ne sont jamais placés dans le prompt. Utilisez des identités courtes durées et des droits minimaux.

### 9.5 Fiabilité opérationnelle

- **timeout** : évite un appel bloqué ;
- **retry borné** : seulement pour erreurs transitoires et opérations sûres ;
- **clé d'idempotence** : évite un double paiement après retry ;
- **circuit breaker** : suspend un service défaillant ;
- **sandbox** : limite fichiers, réseau, CPU, mémoire et durée ;
- **compensation** : annule ou contrebalance une étape déjà effectuée ;
- **budget** : limite nombre d'appels, coût, durée et volume.

### 9.6 Outils, ressources et prompts dans un protocole

Des protocoles d'interopérabilité peuvent exposer plusieurs primitives. Conceptuellement :

- une **ressource** fournit un contenu à lire ;
- un **outil** exécute une opération ;
- un **prompt** propose un gabarit d'interaction.

Les découvrir dynamiquement ne signifie pas les autoriser automatiquement. Épinglez les serveurs approuvés, vérifiez leur identité, filtrez les capacités exposées et traitez leurs descriptions comme une entrée de chaîne d'approvisionnement à contrôler.

---

## 10. Skills : Compétences Réutilisables

Le mot *skill* n'est pas universellement normalisé. Dans ce cours, une skill est un **module procédural versionné** qui explique à un agent comment réaliser une famille de tâches de manière reproductible.

### 10.1 Contenu d'une skill

```text
skill-audit-donnees/
├── SKILL.md          # objectif, déclencheurs, procédure, limites
├── references/       # glossaire, politique, schémas
├── scripts/          # validations déterministes
├── assets/           # modèles de rapport ou exemples
└── evals/            # cas de réussite, échec et sécurité
```

Une skill de qualité précise :

- cas d'usage et conditions de non-usage ;
- entrées attendues et livrable ;
- étapes et points de contrôle ;
- outils nécessaires et droits minimaux ;
- critères d'arrêt et remontée humaine ;
- exemples, contre-exemples et erreurs fréquentes ;
- tests et numéro de version.

### 10.2 Différence entre skill et tool

Un **outil** est une capacité atomique : `lire_fichier(path)`. Une **skill** est un savoir-faire : « auditer un dataset », qui peut demander de lire le schéma, exécuter un script, interpréter le rapport et produire une synthèse selon une grille.

```text
skill = instructions + ressources + orchestration + critères de qualité
tool  = interface d'exécution bornée
```

Une skill n'accorde aucun privilège. Elle ne peut utiliser que les outils autorisés par l'orchestrateur.

### 10.3 Chargement progressif

Présenter toutes les instructions et tous les outils en permanence surcharge le contexte. Une approche progressive :

1. conserver nom et description courte des skills disponibles ;
2. sélectionner la skill pertinente ;
3. charger son instruction complète ;
4. ouvrir seulement les références nécessaires ;
5. activer uniquement les outils requis.

Le routeur doit savoir répondre « aucune skill adaptée ». Une sélection forcée produit des procédures hors sujet.

### 10.4 Sécurité de la chaîne d'approvisionnement

Une skill peut contenir du code ou des instructions dangereuses. Avant installation ou mise à jour :

- examiner provenance, signature, licence et changements ;
- épingler une version ;
- scanner scripts et dépendances ;
- tester en sandbox sans secret ;
- revoir les permissions demandées ;
- empêcher une instruction embarquée d'élargir les droits ;
- prévoir révocation et retour à la version précédente.

### 10.5 Évaluer une skill

Mesurez sélection correcte, réussite de tâche, respect de la procédure, nombre d'appels, coût, latence, qualité du livrable, abstention et résistance aux entrées hostiles. Une skill est un artefact logiciel : revue, tests de régression et journal des versions sont nécessaires.

---

## 11. Mémoire Agentique et Gestion du Contexte

La fenêtre de contexte est ce que le modèle voit **maintenant** ; la mémoire est un système externe qui décide quoi écrire, conserver, retrouver et oublier.

| Mémoire | Contenu | Durée | Exemple |
| :--- | :--- | :--- | :--- |
| Travail | état de la mission | minutes/heures | plan, étapes terminées |
| Conversation | échanges récents ou résumé | session | préférences exprimées |
| Épisodique | événements passés | longue | incident et résolution validée |
| Sémantique | faits structurés | longue | glossaire interne |
| Procédurale | manière de faire | versionnée | skills et politiques |

Une base documentaire RAG n'est pas automatiquement une mémoire personnelle. La première contient des sources gouvernées ; la seconde dérive des interactions et exige consentement, contrôle utilisateur et politique de rétention.

### 11.1 Cycle de mémoire

```text
événement → filtrer → résumer/structurer → autoriser l'écriture
          → stocker avec provenance et expiration
          → rechercher pour une future tâche
          → vérifier pertinence et fraîcheur
          → oublier/corriger sur demande
```

N'écrivez pas chaque sortie du modèle comme un fait : cela transforme une hallucination en « souvenir ». Distinguez contenu utilisateur, observation d'outil, inférence du modèle et fait validé.

### 11.2 Gestion du contexte

Lorsque le contexte est trop long :

- conservez les instructions prioritaires et contraintes actives ;
- résumez les anciens tours avec liens vers les preuves ;
- retirez les résultats d'outils obsolètes ;
- récupérez les éléments pertinents à la demande ;
- préservez décisions, approbations et identifiants d'artefacts.

Le résumé est une compression avec perte. Testez qu'il ne supprime pas une négation, une limite ou une décision importante.

### 11.3 Vie privée et empoisonnement

Donnez à l'utilisateur la possibilité d'inspecter, corriger et supprimer ses souvenirs. Interdisez secrets, données sensibles non nécessaires et instructions externes persistantes. Une mémoire empoisonnée peut influencer toutes les sessions futures : exigez provenance, niveaux de confiance, expiration et validation des écritures à fort impact.

---

## 12. Orchestration et Systèmes Multi-Agents

Un workflow fixe enchaîne des étapes connues ; un agent choisit dynamiquement la prochaine action. Commencez par le niveau d'autonomie minimal suffisant.

```text
fonction → chaîne → routeur → machine à états → agent unique → équipe multi-agents
 contrôle élevé                                                flexibilité élevée
```

### 12.1 Patrons d'orchestration

| Patron | Fonctionnement | Bon usage | Risque |
| :--- | :--- | :--- | :--- |
| Routeur | choisit une branche spécialisée | intentions bien séparées | mauvais routage |
| Pipeline | étapes ordonnées | processus stable | rigidité |
| Superviseur–workers | délègue et agrège | sous-tâches indépendantes | goulot du superviseur |
| Blackboard | agents partagent un état | résolution collaborative | état incohérent |
| Reviewer–author | produit puis vérifie | artefact testable | boucles sans fin |
| Débat/vote | solutions multiples comparées | forte incertitude | accord sans preuve |

### 12.2 Quand plusieurs agents sont justifiés

Utilisez plusieurs agents lorsqu'il existe de vraies frontières : compétences, données, permissions, outils, contextes ou tâches parallèles indépendantes. Un agent lecture seule peut rechercher pendant qu'un agent autorisé prépare une action, puis un contrôleur indépendant applique une politique déterministe.

Évitez-les si un seul appel, un workflow ou une fonction résout la tâche. Ajouter des personnages au même modèle avec le même contexte crée souvent une diversité superficielle, tout en multipliant coût, latence et erreurs.

### 12.3 Contrat de délégation

Chaque sous-tâche doit préciser :

- objectif et définition de « terminé » ;
- entrées et sources autorisées ;
- format de sortie ;
- outils, budget et échéance ;
- opérations interdites ;
- condition de remontée ;
- preuves à joindre.

Le résultat d'un agent est non fiable jusqu'à validation. Le superviseur doit vérifier schéma, tests et sources, pas seulement accepter une phrase convaincante.

### 12.4 Partage d'état et concurrence

Deux agents peuvent modifier la même ressource. Utilisez identifiants de version, verrous ou opérations transactionnelles. Rendez explicites : propriétaire de chaque artefact, règle de fusion, résolution de conflit et effet d'une annulation.

Pour les tâches parallèles, exigez l'indépendance des écritures ou une phase d'agrégation contrôlée. Un « dernier écrivain gagne » peut effacer un travail valide.

### 12.5 Pannes multi-agents

- cascade d'une hallucination entre agents ;
- délégation circulaire et boucle coûteuse ;
- messages perdus ou dupliqués ;
- désaccord sans règle d'arbitrage ;
- confusion d'identité ou de permissions ;
- contexte sensible propagé au mauvais agent ;
- validation factice par un agent partageant la même faiblesse.

Imposez profondeur maximale de délégation, budget global et local, identifiants de corrélation, délais, détection de cycles et arrêt sûr. Un agent ne doit pas pouvoir créer un sous-agent plus privilégié que lui.

### 12.6 Évaluer l'intérêt réel du multi-agent

Comparez au meilleur agent unique à budget équivalent. Mesurez taux de réussite, variance, coût, latence, nombre de tours, erreurs de coordination et sécurité. Faites une ablation : retirer un agent dégrade-t-il réellement le résultat ? Sinon, simplifiez.

---

## 13. Guardrails et Défense en Profondeur

Un garde-fou n'est pas un produit unique. C'est un ensemble de contrôles indépendants et proportionnés au risque. Le modèle peut aider à classer un contenu, mais les frontières de permission reposent sur du code et l'infrastructure.

### 13.1 Couches de contrôle

```text
IDENTITÉ ET POLITIQUE
  ↓
entrée utilisateur → validation / modération / limites
  ↓
retrieval → ACL / provenance / isolation des instructions
  ↓
plan → outils autorisés / budget / contraintes
  ↓
appel → schéma / règles métier / approbation
  ↓
exécution → sandbox / moindre privilège / timeout
  ↓
sortie → schéma / citations / secrets / qualité
  ↓
journaux, alertes, incident, retour arrière
```

### 13.2 Menaces principales et contrôles

| Menace | Exemple | Contrôles complémentaires |
| :--- | :--- | :--- |
| Injection directe | utilisateur demande d'ignorer la politique | politique hors modèle, validation d'action |
| Injection indirecte | page web contient de fausses instructions | traiter contenu comme donnée, isoler outils |
| Exfiltration | outil envoie un secret vers l'extérieur | filtrage destination, DLP, secrets hors contexte |
| Abus d'outil | suppression ou paiement non voulu | moindre privilège, approbation exacte, plafonds |
| Confused deputy | agent agit avec des droits supérieurs à l'utilisateur | propager identité et ACL jusqu'à l'outil |
| Empoisonnement RAG/mémoire | faux document promu comme vérité | provenance, validation d'ingestion, révocation |
| Boucle de coût | appels sans fin | budget de pas/temps/coût, détection de cycle |
| Dépendance compromise | skill ou serveur modifié | versions épinglées, revue, signatures, sandbox |

### 13.3 Injection de prompt : modèle mental

Toute donnée lue peut contenir du langage impératif. Une page, un PDF, un email ou une sortie d'outil n'acquiert jamais le statut d'instruction de confiance. Les étapes robustes sont :

1. délimiter la donnée et sa provenance ;
2. ne pas lui exposer d'outil inutile ;
3. extraire seulement les champs nécessaires ;
4. valider toute action indépendamment du texte ;
5. bloquer destinations et paramètres hors politique ;
6. tester des attaques encodées, multilingues et multi-tours.

Un second LLM « juge » peut être contourné et ne constitue qu'une couche supplémentaire.

### 13.4 Human-in-the-loop bien conçu

L'humain doit recevoir une information exploitable : action exacte, arguments, motif, source, impact, caractère réversible et différences depuis la dernière prévisualisation. L'approbation expire et est liée à un hachage de la requête ; toute modification significative exige une nouvelle validation.

Évitez la fatigue d'approbation : ne demandez pas une confirmation pour chaque lecture bénigne, mais n'agrégez pas non plus plusieurs écritures critiques dans un bouton opaque.

### 13.5 Contrôles de sortie

- valider le schéma plutôt que parser du texte libre ;
- vérifier citations et nombres contre les observations ;
- détecter secrets, données personnelles et contenu interdit ;
- encoder correctement selon la destination pour éviter injection SQL, HTML ou shell ;
- marquer incertitude et abstention ;
- conserver un chemin de recours humain.

### 13.6 Modèle de risque par action

Pour chaque capacité, noter au minimum : impact, probabilité, réversibilité, portée, sensibilité des données et détectabilité. Cette analyse détermine autonomie, approbation, journalisation et tests requis. Une même fonction « envoyer » n'a pas le même risque pour un brouillon interne et un message public à un million de personnes.

---

## 14. Évaluation, Observabilité et Mise en Production

### 14.1 Jeux d'évaluation

Constituez des cas : fréquents, limites, ambigus, hors périmètre, outils en panne, sources contradictoires, absence de réponse, injections, demandes sensibles et longues trajectoires. Chaque cas définit critères de réussite et violations critiques.

Évaluez trois niveaux :

- **résultat** : la tâche est-elle accomplie correctement ?
- **trajectoire** : les bons outils et sources ont-ils été utilisés, sans étape risquée ?
- **système** : coût, latence, disponibilité, sécurité et expérience utilisateur.

### 14.2 Métriques agentiques

- taux de succès de bout en bout ;
- exactitude des appels et arguments ;
- taux de récupération et qualité des citations ;
- nombre de pas, tokens, appels et coût ;
- taux de retry, timeout, boucle et abandon ;
- taux d'approbation/refus et actions évitées ;
- violations de politique et fuites de données ;
- qualité d'abstention et de remontée humaine.

Un score moyen peut masquer une violation catastrophique. Définissez des **gates** : aucune exfiltration, aucune action critique sans approbation, aucune fuite inter-locataire.

### 14.3 Traces observables

Une trace utile contient : identifiant de requête, versions modèle/prompt/skill, outils proposés et exécutés, arguments expurgés, résultats structurés, sources RAG, décisions de politique, approbations, budgets, erreurs, latence et résultat final.

Ne journalisez pas le raisonnement interne détaillé ni des secrets. Conservez plutôt un résumé de décision et des preuves vérifiables. Appliquez contrôle d'accès et durée de rétention aux traces.

### 14.4 Tests

- **unitaires** : outils, validateurs, policies, chunking ;
- **contrat** : schémas d'entrée/sortie et erreurs ;
- **intégration** : RAG, outils, mémoire et orchestrateur ;
- **simulation** : services externes factices et pannes injectées ;
- **régression** : tâches et attaques connues à chaque changement ;
- **red team** : recherche créative d'abus, sans remplacer les tests systématiques.

### 14.5 Déploiement progressif

Commencez hors ligne, puis mode observation sans action, *shadow mode*, petit groupe, faible plafond, et élargissement conditionné aux métriques. Préparez interrupteur d'arrêt, révocation des identités, désactivation d'un outil, retour à une version stable et procédure d'incident.

### 14.6 Boucle d'amélioration

Les échecs réels alimentent un jeu d'évaluation après anonymisation et revue. Corrigez l'étage causal : source, retrieval, prompt, skill, outil, policy ou interface. Ne modifiez pas cinq composants à la fois sans expérience contrôlée.

---

## 15. Architecture de Référence et Étude de Cas

### 15.1 Architecture complète

```text
                              ┌─────────────────────────┐
Utilisateur + identité ──────►│ API / orchestrateur     │
                              │ état, budget, arrêt      │
                              └───────────┬─────────────┘
                                          │
                     ┌────────────────────┼────────────────────┐
                     ▼                    ▼                    ▼
             ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
             │ Sélecteur de │     │ RAG / base de│     │ Registre des │
             │ skill        │     │ connaissances│     │ outils       │
             └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
                    │                    │                    │
                    └──────────────┬─────┴──────────────┬─────┘
                                   ▼                    ▼
                            ┌──────────────┐      ┌──────────────┐
                            │ Modèle       │      │ Policy engine│
                            │ décision     │      │ + approbation│
                            └──────┬───────┘      └──────┬───────┘
                                   │ appel proposé       │ décision
                                   └──────────┬───────────┘
                                              ▼
                                    ┌──────────────────┐
                                    │ Exécuteur isolé  │
                                    │ outils + secrets │
                                    └────────┬─────────┘
                                             ▼
                                     Systèmes externes

Tous les étages → traces, métriques, alertes et évaluation
```

### 15.2 Cas : assistant de support interne

**Objectif.** Répondre aux questions et préparer des tickets, sans exposer de documents confidentiels ni modifier un compte sans accord.

**Base de connaissances.** Procédures versionnées, incidents connus et catalogue de services, avec ACL par équipe et date de validité.

**RAG.** Recherche hybride filtrée par identité ; passages rerankés ; citations obligatoires ; abstention si aucune source suffisante.

**Skills.** « diagnostiquer un accès », « préparer un ticket » et « expliquer une procédure », chacune avec critères d'usage, scripts de contrôle et tests.

**Tools.** Lire l'état d'un service, lire un ticket, créer un brouillon, réinitialiser un accès. Les trois premiers ont des droits distincts ; la réinitialisation exige une approbation liée à l'utilisateur et au compte exacts.

**Mémoire.** État de la tâche et préférences non sensibles. Pas de conservation automatique des mots de passe, tokens ou extraits confidentiels.

**Orchestration.** Un routeur déterministe traite les demandes simples ; l'agent intervient seulement pour les diagnostics multi-étapes. Un second agent n'est ajouté que si la revue indépendante améliore une métrique mesurée.

**Guardrails.** ACL au retrieval et à l'outil, schémas stricts, destinations autorisées, budget de cinq appels, timeout, sandbox, citations vérifiées, contrôle des secrets et journal d'audit.

**Évaluation.** Questions connues/inconnues, faux documents, injection dans un ticket, panne d'outil, utilisateur sans droit, demande ambiguë, double clic et tentative de modifier l'action après approbation.

### 15.3 Pseudo-code de l'orchestrateur

```python
def traiter(requete, identite):
    etat = initialiser_etat(requete, budget_appels=5)
    verifier_entree(requete, identite)

    while not etat.termine:
        contexte = recuperer_connaissance(requete, identite)
        decision = modele.proposer(etat, contexte, outils_autorises(identite))

        if decision.type == "reponse":
            return verifier_reponse(decision, contexte)

        appel = valider_schema(decision.appel)
        autorisation = policy.evaluer(identite, appel, etat)

        if autorisation.exige_approbation:
            obtenir_approbation_exacte(appel, identite)

        observation = executer_en_sandbox(appel, autorisation)
        etat.ajouter_observation(observation)
        etat.consommer_budget()

    return reponse_arret_sur(etat)
```

Ce pseudo-code montre que récupération, modèle, policy et exécution sont des composants séparés. Une implémentation réelle doit aussi gérer erreurs, idempotence, secrets, concurrence et traces.

---

## 16. Checklist et Questions de Compréhension

### Checklist de conception

- [ ] Un workflow déterministe a été envisagé avant l'autonomie.
- [ ] Modèle, base de connaissances, RAG, mémoire, tools et skills sont séparés.
- [ ] Le RAG conserve provenance, version, citations et ACL.
- [ ] Chaque outil possède schéma strict, risque, timeout et politique.
- [ ] Les secrets restent hors du contexte du modèle.
- [ ] Chaque skill est versionnée, testée et sans privilège implicite.
- [ ] Les écritures mémoire sont filtrées, traçables et supprimables.
- [ ] Le multi-agent démontre un gain face à un agent unique.
- [ ] Budgets, critères d'arrêt, idempotence et compensation sont prévus.
- [ ] Les actions sensibles utilisent moindre privilège et approbation exacte.
- [ ] Les injections directes et indirectes font partie des tests.
- [ ] Résultat, trajectoire, coût, sécurité et abstention sont évalués.
- [ ] Déploiement progressif, arrêt d'urgence et retour arrière sont disponibles.

### Questions de compréhension

1. Pourquoi une base vectorielle n'est-elle pas une base de connaissances complète ?
2. Quelle différence entre RAG, mémoire épisodique et fine-tuning ?
3. À quel moment les ACL doivent-elles filtrer les documents récupérés ?
4. Pourquoi un appel d'outil proposé par le modèle n'est-il pas encore une action autorisée ?
5. Donnez un exemple de skill utilisant trois outils sans acquérir leurs privilèges.
6. Quand un superviseur multi-agent est-il justifié, et comment prouver son gain ?
7. Comment une injection indirecte peut-elle conduire à une exfiltration ?
8. Pourquoi l'approbation doit-elle être liée aux arguments exacts de l'action ?
9. Quelles métriques distinguent un échec de retrieval d'un échec de génération ?
10. Quel comportement adopter quand le budget d'étapes est épuisé ?

**Mini-projet.** Concevez un agent qui répond à des questions RH et prépare une demande de congé. Livrez le schéma d'architecture, les ACL documentaires, deux skills, les contrats d'outils, la matrice de risques, les règles d'approbation, dix cas d'évaluation et les critères de mise en production. Aucun envoi réel n'est nécessaire.
