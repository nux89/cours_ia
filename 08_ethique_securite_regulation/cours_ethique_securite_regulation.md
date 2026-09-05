# Module 8 : Éthique, Sécurité, Régulation et Impact de l'IA

> **Étape 10/11 du parcours** · [← Précédent](../04_ia_agentique/cours_ia_agentique.md) · [Progression et ordre des sections](../PARCOURS_ET_EVALUATION.md) · [Suivant →](../07_mlops_production/cours_mlops_production.md)

> L'éthique n'est pas une couche de vernis ajoutée à la fin : c'est une **contrainte d'ingénierie** présente à chaque étape, du cadrage des données au retrait d'un modèle. Un système techniquement performant peut être inéquitable, opaque, illégal ou nuisible. Ce module consolide et approfondit ce qui était distribué dans le cursus (biais et vie privée du module 1, interprétabilité du module 2, garde-fous du module 4) et y ajoute le cadre réglementaire et l'impact sociétal et environnemental.

**Objectifs du module.** À l'issue de ce chapitre, vous saurez identifier et mesurer plusieurs formes de biais, comprendre pourquoi certaines définitions d'équité sont incompatibles, appliquer les principes de protection des données, situer les grandes obligations réglementaires (dont l'approche par niveaux de risque), articuler sécurité et sûreté d'un système d'IA, estimer et réduire son impact environnemental, et mettre en place une gouvernance responsable avec documentation et supervision humaine.

**Prérequis.** Pour la synthèse complète : modules 1 (données), 2 (métriques) et 4 (risques agentiques). Les principes des sections 1 et 4 se lisent dès la collecte ; les sections 2–3 accompagnent le ML. Le module 7 n'est pas un prérequis : il concrétisera ensuite la gouvernance en production. Ce module est **transversal**, puis consolidé avant le déploiement.

---

## 📖 Le Dico du Débutant (Jargon Buster)
- **Biais (au sens équité)** : un traitement systématiquement défavorable à un groupe, distinct du « biais » statistique du compromis biais-variance (module 2).
- **Équité (*fairness*)** : une propriété de non-discrimination ; il en existe **plusieurs définitions**, parfois incompatibles entre elles.
- **Explicabilité / Interprétabilité** : la capacité à comprendre ou justifier une décision du modèle.
- **Vie privée (*privacy*)** : la protection des données personnelles et le contrôle des individus sur leurs informations.
- **Données personnelles** : toute information se rapportant à une personne identifiée ou identifiable.
- **Anonymisation / Pseudonymisation** : rendre des données non attribuables à une personne (irréversible) / remplacer les identifiants directs par des codes (réversible avec une clé).
- **Gouvernance** : l'ensemble des rôles, processus et documents qui rendent un système redevable (*accountable*).
- **Passage à l'échelle du risque** : plus une décision automatisée est fréquente et impactante, plus les garde-fous doivent être stricts.

---

## Table des Matières
1. [L'Éthique comme Contrainte d'Ingénierie](#1-léthique-comme-contrainte-dingénierie)
2. [Biais et Équité : Mesurer, Arbitrer, Documenter](#2-biais-et-équité--mesurer-arbitrer-documenter)
3. [Transparence et Explicabilité](#3-transparence-et-explicabilité)
4. [Vie Privée et Protection des Données](#4-vie-privée-et-protection-des-données)
5. [Le Paysage Réglementaire](#5-le-paysage-réglementaire)
6. [Sécurité, Sûreté et Usages Malveillants](#6-sécurité-sûreté-et-usages-malveillants)
7. [Impact Environnemental et Sociétal](#7-impact-environnemental-et-sociétal)
8. [Gouvernance et Redevabilité](#8-gouvernance-et-redevabilité)
9. [Checklist et Questions de Compréhension](#9-checklist-et-questions-de-compréhension)

---

## 1. L'Éthique comme Contrainte d'Ingénierie

Un modèle prend ou informe des **décisions qui touchent des personnes** : accorder un crédit, trier des candidatures, prioriser des soins, modérer des contenus. Les erreurs ne sont pas symétriques et ne tombent pas au hasard : elles frappent souvent les populations déjà fragiles.

> 🔍 **Cas d'école.** Un outil de tri de CV entraîné sur les recrutements passés d'une entreprise majoritairement masculine apprend à pénaliser les signaux associés aux femmes — non par malveillance, mais parce qu'il **reproduit fidèlement un historique biaisé**. « Garbage in, garbage out » (module 1) devient ici « injustice in, injustice out ». Le modèle est statistiquement correct et éthiquement inacceptable.

Trois idées structurantes :

1. **Le biais entre par les données** (historique, sélection, étiquettes) autant que par le modèle (rappel module 1, §7 et §11).
2. **Une variable sensible retirée peut survivre par procuration** (*proxy*) : un code postal peut encoder l'origine sociale ou ethnique (module 1, §9.2).
3. **Aucune métrique unique ne capture l'équité** : il faut évaluer par sous-groupe et expliciter les compromis (rappel module 2, §13).

---

## 2. Biais et Équité : Mesurer, Arbitrer, Documenter

### 2.1 D'où vient le biais

| Source | Exemple | Lien cursus |
| :--- | :--- | :--- |
| Biais historique | l'historique reflète des inégalités réelles | module 1, §7.3 |
| Biais de sélection | la population observée ≠ population d'usage | module 1, §7.3 |
| Biais d'étiquetage | annotateurs incohérents ou partiaux | module 1, §7.4 |
| Biais d'agrégation | un modèle unique pour des sous-groupes hétérogènes | module 2, §13 |
| Biais de déploiement | usage réel différent de l'usage prévu | module 7, §7 |
| Boucle de rétroaction | les décisions modifient les données futures | module 1, §7.3 |

### 2.2 Plusieurs définitions… incompatibles

Il existe des dizaines de notions d'équité. Trois familles fréquentes :

- **Parité démographique** : le taux de décisions positives est le même entre groupes.
- **Égalité des chances** : le taux de vrais positifs (rappel, module 2) est le même entre groupes.
- **Calibration par groupe** : à score égal, la probabilité réelle est la même entre groupes.

> ⚠️ **Résultat d'impossibilité.** Lorsque les **taux de base diffèrent** entre groupes, on ne peut généralement **pas** satisfaire simultanément ces critères. Choisir une notion d'équité est donc un **arbitrage de valeurs**, pas un réglage technique neutre. Il faut documenter le choix, les personnes affectées et la procédure de recours (rappel module 1, §11.1).

### 2.3 Mesurer et atténuer

- **Mesurer** : décomposer les métriques (rappel, précision, faux positifs, calibration, taux de sélection) **par sous-groupe pertinent**, avec effectifs et incertitude. Un score global élevé peut masquer une défaillance sur un groupe minoritaire (rappel module 4, §14.2 sur les moyennes trompeuses).
- **Atténuer** : à trois moments possibles — **pré-traitement** (rééquilibrer, corriger les étiquettes), **in-traitement** (contraintes d'équité dans l'apprentissage), **post-traitement** (seuils par groupe). Chaque levier a des effets de bord et un coût ; aucun ne « résout » l'équité une fois pour toutes.

> 💡 **Un test statistique ne clôt pas le débat.** L'équité se juge aussi sur le contexte, le recours offert aux personnes et la transparence, pas seulement sur un tableau de métriques.

---

## 3. Transparence et Explicabilité

Comprendre **pourquoi** un modèle décide est nécessaire pour le déboguer, le faire confiance à bon escient, respecter des obligations légales et offrir un recours.

- **Global vs local vs contrefactuel** (rappel module 2, §13.1) : comportement moyen, explication d'une décision précise, ou « qu'aurait-il fallu changer pour un autre résultat ».
- **Limites** : les explications sont des **approximations**. Une importance par permutation peut être trompeuse avec des variables corrélées ; une attribution locale n'est pas une preuve causale. Testez leur stabilité et adaptez le niveau à l'utilisateur.
- **Transparence système** : au-delà du modèle, documenter les données, les usages prévus et interdits, les limites — via **fiche de données** (*datasheet*, module 1, §11.3) et **fiche modèle** (*model card*, module 2, §14).
- **Droit à l'explication** : plusieurs cadres exigent une information intelligible sur une décision automatisée significative, et parfois une intervention humaine.

> 🔍 **Distinguer explicable et interprétable.** Un modèle **interprétable par conception** (régression, petit arbre) est lisible directement ; un modèle complexe demande des méthodes d'**explication** post-hoc, plus fragiles. Pour les décisions à fort enjeu, une famille auditable peut être préférable, même à performance légèrement moindre (rappel module 2, §10.12).

---

## 4. Vie Privée et Protection des Données

### 4.1 Principes fondateurs (rappel et approfondissement, module 1, §11.2)

- **Minimisation** : ne collecter que le nécessaire à une finalité déclarée.
- **Limitation de finalité** : ne pas réutiliser des données pour un autre usage sans base légitime.
- **Base légale / consentement** : traiter des données personnelles requiert un fondement (consentement, contrat, obligation légale…).
- **Droits des personnes** : accès, rectification, effacement, opposition, portabilité selon les cadres applicables.
- **Sécurité** : chiffrement en transit et au repos, droits par rôle, journalisation des accès, durée de conservation et suppression.

### 4.2 Anonymisation n'est pas magie

La **pseudonymisation** (remplacer les identifiants directs) réduit le risque mais reste réversible avec la clé. L'**anonymisation** vise l'irréversibilité, mais des **croisements** peuvent réidentifier des individus (rappel module 1, §11.2). Publier un jeu « anonymisé » sans analyse de réidentification est risqué.

### 4.3 Techniques de protection avancées (intuition)

- **Confidentialité différentielle** : ajouter un bruit calibré aux résultats/statistiques pour qu'on ne puisse pas déduire la présence d'un individu, au prix d'un peu de précision.
- **Apprentissage fédéré** : entraîner sur des appareils sans centraliser les données brutes (des mises à jour agrégées circulent, pas les données) — utile mais pas une garantie absolue de vie privée.
- **Mémorisation des modèles** : un modèle (surtout un LLM) peut restituer des exemples d'entraînement sensibles. Dédupliquer, tester la mémorisation, séparer les évaluations (rappel modules 3, §15.5 et 5, §9.3).

---

## 5. Le Paysage Réglementaire

Le droit de l'IA évolue vite et varie selon les juridictions. L'objectif ici n'est pas l'exhaustivité juridique mais une **carte mentale** ; en pratique, consultez un service juridique et les textes applicables à votre secteur et votre pays.

### 5.1 Protection des données

Des cadres comme le **RGPD** (Union européenne) encadrent le traitement des données personnelles : bases légales, droits des personnes, analyses d'impact (*DPIA*) pour les traitements à risque, et encadrement des **décisions entièrement automatisées** ayant un effet significatif. D'autres régions ont leurs propres régimes.

### 5.2 L'approche par niveaux de risque

Plusieurs régulations (dont l'**AI Act** européen) classent les systèmes par **niveau de risque**, avec des obligations croissantes :

```text
Risque inacceptable  → interdit (ex. certaines formes de notation sociale)
Risque élevé         → obligations fortes (gestion des risques, qualité des
                        données, documentation, supervision humaine, robustesse)
Risque limité        → obligations de transparence (ex. signaler une interaction
                        avec une IA, marquer certains contenus générés)
Risque minimal       → peu ou pas d'obligations spécifiques
```

L'idée directrice rejoint le module 4 (§13.6) et le module 7 : **plus l'impact et l'autonomie sont élevés, plus les garanties doivent être strictes**. Les applications sensibles (santé, crédit, emploi, justice, biométrie) tombent souvent dans les catégories à obligations fortes.

### 5.3 Exigences transverses fréquentes

Quel que soit le cadre précis, on retrouve : **documentation** (données, modèle, évaluation), **traçabilité**, **supervision humaine** pour les décisions significatives, **robustesse et sécurité**, **information** des personnes concernées, et parfois **marquage** des contenus générés par IA.

---

## 6. Sécurité, Sûreté et Usages Malveillants

Deux notions distinctes et complémentaires :

- **Sûreté (*safety*)** : éviter que le système ne cause du tort **par accident** (erreur, dérive, effet de bord non voulu).
- **Sécurité (*security*)** : le protéger contre des **attaques intentionnelles**.

### 6.1 Menaces spécifiques à l'IA

| Menace | Idée | Lien cursus |
| :--- | :--- | :--- |
| Exemples adverses | perturbation imperceptible qui trompe le modèle | robustesse, module 2, §13.2 |
| Empoisonnement des données | corrompre l'entraînement pour biaiser le modèle | qualité/provenance, module 1 |
| Injection de prompt (directe/indirecte) | faire dévier un système à base de LLM | module 4, §13 ; module 5, §9.3 |
| Exfiltration / inférence d'appartenance | extraire des données ou déduire qui était dans le jeu | vie privée, §4 |
| Vol / rétro-ingénierie de modèle | copier un modèle via ses réponses | gouvernance des accès |
| Usage malveillant du contenu généré | désinformation, hypertrucages, fraude | ci-dessous |

### 6.2 Contenus générés et désinformation

Les modèles génératifs (module 3) facilitent la création de texte, d'images, de voix et de vidéos trompeurs (*deepfakes*). Réponses partielles : **marquage/filigrane** des contenus, provenance vérifiable, détection (imparfaite), et surtout **transparence** sur le caractère synthétique. Aucune mesure n'est infaillible ; la combinaison de garde-fous techniques, organisationnels et légaux reste nécessaire (défense en profondeur, module 4, §13).

### 6.3 Défense en profondeur (rappel opérationnel)

Les frontières de sécurité reposent sur du **code et de l'infrastructure**, pas sur une consigne dans un prompt (module 4). Moindre privilège, validation hors modèle, approbation liée à l'action exacte, journalisation, budgets et supervision humaine s'appliquent à tout système à fort impact.

---

## 7. Impact Environnemental et Sociétal

### 7.1 Empreinte environnementale

L'entraînement et surtout l'**inférence à grande échelle** consomment énergie, eau (refroidissement) et matériel. Bonnes pratiques :

- **mesurer** avant d'optimiser (énergie/coût par requête, module 7, §10) ;
- privilégier le **modèle le plus simple suffisant** (rappel modules 1 et 4) ;
- réutiliser (**transfert**, module 3) plutôt que réentraîner de zéro ;
- **compresser** (quantification, distillation) et regrouper les requêtes ;
- choisir des infrastructures et des horaires moins carbonés quand c'est possible ;
- **documenter** l'empreinte dans la fiche modèle.

### 7.2 Impacts sociétaux

- **Travail** : automatisation, transformation des métiers, conditions des annotateurs de données.
- **Concentration et accès** : coût des grands modèles, dépendances, fracture d'accès.
- **Effets cognitifs et informationnels** : dépendance excessive, homogénéisation, bulles, érosion de la confiance.
- **Accessibilité** : un système doit rester utilisable par des personnes en situation de handicap (rappel de l'audit du cursus : accessibilité à tester).

Ces impacts ne se mesurent pas dans une seule métrique ; ils exigent une réflexion sur la **finalité** et les **parties prenantes**, en amont du projet (rappel du cadrage, module 1, §7).

---

## 8. Gouvernance et Redevabilité

La gouvernance transforme les bonnes intentions en **processus vérifiables**. Des cadres comme le **NIST AI Risk Management Framework** structurent cette démarche autour de fonctions : *gouverner, cartographier, mesurer, gérer* le risque sur tout le cycle de vie.

### 8.1 Rôles et responsabilités

Qui est **redevable** en cas de préjudice ? La responsabilité doit être **attribuée explicitement** (propriétaire du système, du modèle, des données), pas diluée dans « l'algorithme ». Une décision automatisée reste la responsabilité de l'organisation qui la déploie.

### 8.2 Documentation et cycle de vie

- **Cadrage** : finalité, parties affectées, risques, alternatives (module 1, §7 ; PROJET_FINAL, critère « cadrage et éthique »).
- **Données** : *datasheet* (provenance, consentement, limites).
- **Modèle** : *model card* (usages, performances par sous-groupe, limites, usages interdits).
- **Décisions** : registre des choix, des versions et des évaluations (module 7).
- **Suivi** : surveillance, incidents, recours, retrait planifié.

### 8.3 Supervision humaine et recours

Pour les décisions significatives : **human-in-the-loop** proportionné au risque (module 4, §13.4), information claire des personnes, et **voie de recours** réelle pour contester une décision. Éviter deux extrêmes : l'automatisation totale sans filet, et l'« humain alibi » qui valide sans pouvoir réel.

> 💡 **Le fil rouge du cursus.** De « garbage in, garbage out » (module 1) à l'« approbation liée à l'action exacte » (module 4), une même exigence traverse tout : **rendre chaque décision explicite, mesurable, contestable et réversible autant que possible**. L'éthique n'est pas un module de plus, c'est la lecture responsable de tous les autres.

---

## 9. Checklist et Questions de Compréhension

### Checklist de déploiement responsable

- [ ] La finalité, les parties affectées, les risques et les alternatives sont documentés.
- [ ] Les métriques sont décomposées par sous-groupe, avec effectifs et incertitude.
- [ ] La notion d'équité retenue est explicitée et justifiée (arbitrage assumé).
- [ ] Les variables sensibles et leurs proxys sont audités.
- [ ] Une base légale, la minimisation et les droits des personnes sont respectés.
- [ ] Le niveau de risque réglementaire du système est identifié et ses obligations traitées.
- [ ] Sûreté (accidents) et sécurité (attaques) sont couvertes par des contrôles hors modèle.
- [ ] L'impact environnemental est mesuré et réduit ; le modèle le plus simple suffisant est préféré.
- [ ] Datasheet, model card et registre de décisions existent et sont à jour.
- [ ] La supervision humaine est proportionnée et une voie de recours est offerte.
- [ ] La responsabilité est attribuée explicitement, avec un plan de retrait.

### Questions de compréhension

1. Pourquoi retirer une variable sensible ne suffit-il pas à rendre un modèle équitable ?
2. Citez trois définitions d'équité et expliquez pourquoi elles peuvent être incompatibles.
3. Quelle différence entre pseudonymisation et anonymisation, et pourquoi l'anonymat n'est-il jamais garanti ?
4. Qu'apporte la confidentialité différentielle, et à quel prix ?
5. Décrivez l'approche par niveaux de risque et donnez un exemple par niveau.
6. Différence entre sûreté (*safety*) et sécurité (*security*) ; donnez une menace de chaque type.
7. Citez trois leviers concrets pour réduire l'empreinte environnementale d'un système d'IA.
8. Pourquoi une moyenne globale de performance peut-elle masquer une injustice, et que faire ?
9. Que signifie « human-in-the-loop proportionné » et quel est le piège de l'« humain alibi » ?

**Mini-étude de cas.** Une administration veut automatiser la priorisation des demandes d'aide sociale. (a) Identifiez les parties affectées, les risques et le niveau de risque probable. (b) Choisissez une notion d'équité et justifiez l'arbitrage. (c) Décrivez les mesures de vie privée et la documentation à produire. (d) Définissez la supervision humaine, la voie de recours et le plan de suivi/retrait. (e) Expliquez pourquoi un modèle plus simple mais auditable pourrait être préférable ici.

---

Pour approfondir et vérifier ces notions, consultez les référentiels et documentations regroupés dans [REFERENCES.md](../REFERENCES.md).
