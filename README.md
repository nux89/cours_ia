# 🎓 Cursus Universitaire d'Intelligence Artificielle : De la Donnée aux Systèmes Agentiques

Bienvenue dans ce cursus complet d'Intelligence Artificielle conçu pour des étudiants en sciences de l'ingénieur, en informatique et pour les professionnels souhaitant acquérir une maîtrise théorique et pratique approfondie du domaine.

---

## 👥 Public, prérequis et charge de travail

- **Public visé** : étudiants de niveau licence, élèves ingénieurs et professionnels disposant des bases de Python.
- **Prérequis** : variables, fonctions, boucles et classes en Python ; algèbre linéaire élémentaire ; moyenne, variance et probabilités de base.
- **Environnement recommandé** : Python 3.10 à 3.12, dans un environnement virtuel dédié.
- **Charge indicative** : 38 à 48 heures pour le parcours complet, dont environ la moitié en pratique. Une lecture « fondamentaux » peut se limiter aux sections 1 à 9 de chaque cours ; les sections suivantes constituent l'approfondissement et la mise en production.

À la fin du parcours, l'apprenant doit pouvoir :

1. diagnostiquer un jeu de données et construire un prétraitement sans fuite d'information ;
2. choisir une métrique cohérente avec le coût des erreurs et évaluer un modèle sur des données non vues ;
3. expliquer puis entraîner des architectures neuronales, choisir une stratégie de transfert et raisonner sur un système encodeur–décodeur ou multimodal ;
4. distinguer modèle génératif, RAG, mémoire, outil et skill, puis concevoir une orchestration mono ou multi-agent avec des garde-fous vérifiables.

Chaque notebook contient des vérifications exécutables. Le [projet final](PROJET_FINAL.md) permet d'évaluer l'ensemble de ces compétences avec une grille explicite.

---

## 🎯 Philosophie & Objectifs Pédagogiques

L'intelligence artificielle moderne ne se résume pas à l'appel de bibliothèques "boîtes noires". Ce cours a été structuré autour de quatre piliers progressifs :

1. **Comprendre la matière première** : La qualité d'un système d'IA est bornée par la qualité de ses données. Maîtriser la nature des données, leur mise en forme, leur nettoyage et éviter le piège fatal du *Data Leakage*.
2. **Les fondements statistiques et algorithmiques (Machine Learning)** : Assimiler le compromis biais-variance, les métriques d'évaluation selon les contextes métiers, et savoir bâtir des chaînes d'apprentissage robustes avec Scikit-Learn.
3. **La modélisation neuronale et expressive (Deep Learning)** : Comprendre descente de gradient et rétropropagation, puis relier MLP, CNN, RNN/LSTM, attention et Transformers aux architectures encodeur–décodeur, au transfer learning, au fine-tuning, au multimodal et à la diffusion.
4. **Le paradigme de l'autonomie (IA Agentique)** : Passer du modèle au système contrôlé qui récupère des connaissances, gère un contexte, charge des skills, utilise des outils externes et orchestre éventuellement plusieurs agents sous politiques de sécurité.

---

## 📂 Organisation du Cursus

Le cursus est découpé en 4 dossiers thématiques autonomes et progressifs :

```text
cours_ia/
├── 01_nature_et_preparation_des_donnees/
│   ├── cours_nature_et_preparation_donnees.md   # Typologie étendue, EDA, associations, qualité, gouvernance & dérive
│   ├── 01_preparation_donnees_pratique.ipynb    # TP : prétraitement étanche avec Scikit-Learn
│   └── 02_eda_correlations_pandas_polars.ipynb  # TP : Pandas, Polars, visualisations, corrélations & sélection
│
├── 02_machine_learning/
│   ├── cours_machine_learning.md                # Catalogue par problème, métriques, validation, interprétation & cycle de vie
│   └── 02_machine_learning_scikit_learn.ipynb   # TP interactif : Régression, Classification, PCA, Pipelines & GridSearch
│
├── 03_deep_learning/
│   ├── cours_deep_learning.md                   # Réseaux, transfert, fine-tuning, multimodal, captioning & diffusion
│   ├── 01_perceptron_et_mlp.ipynb               # TP 1 : Perceptron unitaire & Multi-Layer Perceptron (PyTorch)
│   ├── 02_cnn_vision.ipynb                      # TP 2 : Réseaux convolutifs pour la vision par ordinateur
│   ├── 03_rnn_series_temporelles.ipynb          # TP 3 : Réseaux récurrents & séries temporelles
│   ├── 04_lstm_sequences.ipynb                  # TP 4 : Cellules LSTM & résolution du vanishing gradient
│   └── 05_decouverte_autres_architectures.ipynb # TP 5 : Attention, Transformers, Auto-encodeurs & panorama génératif
│
└── 04_ia_agentique/
    ├── cours_ia_agentique.md                    # RAG, bases de connaissances, outils, skills, multi-agents & garde-fous
    └── 01_tp_agent_autonome.ipynb               # TP : boucle d'outils structurée, sûre et testable (sans API)
```

---

## 🛠️ Installation & Démarrage Rapide

### 1. Prérequis
- **Python 3.10 à 3.12** installé sur votre machine.
- Un gestionnaire d'environnement virtuel (`venv` ou `conda`).

### 2. Création de l'environnement virtuel
Ouvrez un terminal dans le répertoire `cours_ia` :

```bash
# Création de l'environnement
python3 -m venv venv_cours_ia

# Activation de l'environnement
# Sur macOS/Linux :
source venv_cours_ia/bin/activate
# Sur Windows :
# .\venv_cours_ia\Scripts\activate

# Installation des dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Lancement des notebooks Jupyter
```bash
jupyter lab
```

Naviguez dans les dossiers et exécutez les cellules pas à pas en lisant attentivement les explications et les exercices d'application.

Pour vérifier les liens et blocs des supports, puis exécuter tous les notebooks depuis la racine du projet :

```bash
python scripts/validate_markdown.py
python scripts/validate_notebooks.py
```

Les références primaires et documentations officielles utilisées pour vérifier le contenu sont regroupées dans [REFERENCES.md](REFERENCES.md). Le diagnostic éditorial ayant conduit à cette révision est consigné dans [AUDIT_PEDAGOGIQUE.md](AUDIT_PEDAGOGIQUE.md).

---

## 🗺️ La Grande Carte de l'IA : Comprendre l'Écosystème

Si vous débutez en Intelligence Artificielle, le vocabulaire peut sembler vertigineux. Voici comment s'emboîtent naturellement toutes les notions abordées dans ce cursus :

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│ 1. INTELLIGENCE ARTIFICIELLE (IA)                                                │
│    Le champ global : tout système capable de simuler un comportement intelligent  │
│    (systèmes experts à base de règles, algorithmes de recherche A*, logique).    │
│                                                                                  │
│    ┌────────────────────────────────────────────────────────────────────────┐    │
│    │ 2. MACHINE LEARNING (ML) [Module 1 & 2]                                 │    │
│    │    L'ordinateur apprend les règles à partir des données statistiques    │    │
│    │    (Régression, Arbres de décision, Forêts Aléatoires, SVM, K-Means).  │    │
│    │                                                                        │    │
│    │    ┌──────────────────────────────────────────────────────────────┐    │    │
│    │    │ 3. DEEP LEARNING (Apprentissage Profond) [Module 3]           │    │    │
│    │    │    Sous-ensemble du ML basé sur des réseaux de neurones      │    │    │
│    │    │    artificiels profonds à plusieurs couches (MLP, CNN, LSTM).│    │    │
│    │    │                                                              │    │    │
│    │    │    ┌────────────────────────────────────────────────────┐    │    │    │
│    │    │    │ 4. MODÈLES DE FONDATION & IA GÉNÉRATIVE            │    │    │    │
│    │    │    │    Transformers, modèles de langage et diffusion,  │    │    │    │
│    │    │    │    Diffusion (génération d'images et de code).     │    │    │    │
│    │    │    └────────────────────────────────────────────────────┘    │    │    │
│    │    └──────────────────────────────────────────────────────────────┘    │    │
│    └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│    ┌────────────────────────────────────────────────────────────────────────┐    │
│    │ 5. IA AGENTIQUE [Module 4]                                             │    │
│    │    Donne des bras, des yeux et une mémoire au modèle : autonomie,      │    │    │
│    │    boucle ReAct, utilisation d'outils externes et systèmes multi-agents│    │
│    └────────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧭 Guide de Survie pour le Débutant en IA

> 💡 **"Faut-il être un génie des mathématiques pour apprendre l'IA ?"**  
> **Non !** La quasi-totalité des concepts fondamentaux repose sur des intuitions visuelles et physiques très simples (une pente qu'on descend, une moyenne qu'on calcule, un filtre qu'on applique sur une photo). Les formules mathématiques rigoureuses sont fournies pour ancrer la précision, mais chaque formule du cours est précédée d'une analogie du quotidien et d'un exemple avec des petits chiffres simples calculés pas à pas.

### Les 4 Conseils d'Or pour Réussir ce Cursus :
1. **Lisez le cours théorique (`.md`) avant d'ouvrir le notebook (`.ipynb`)** : Chaque chapitre markdown introduit les notions avec des métaphores, un dictionnaire des termes techniques (*Le Dico du Débutant*) et des exemples chiffrés.
2. **Ne subissez pas le code, apprivoisez-le** : Dans les notebooks, chaque ligne de code clé est commentée. Prenez le temps de comprendre ce que produit chaque commande en observant les affichages et les graphiques.
3. **Cassez le code et expérimentez !** : L'IA s'apprend par l'expérimentation. Changez une valeur (ex: le taux d'apprentissage `lr`, le nombre d'arbres d'une forêt ou la taille d'un filtre d'image) et regardez ce qui se passe : le modèle apprend-il mieux ou fait-il n'importe quoi ?
4. **Faites les exercices guidés** : Chaque notebook propose un exercice d'application avec des indices et une solution commentée détaillée pour vous mettre en confiance.

---

## 📚 Recommandations de travail pour les étudiants
- **Suivez l'ordre des modules** : Ne commencez pas par le Deep Learning sans avoir compris ce qu'est un jeu de données d'entraînement (`Train`) et de test (`Test`) au Module 1.
- **Gardez un carnet de notes** : Notez avec vos propres mots les définitions des termes récurrents (*overfitting*, *gradient*, *loss*, *batch*, *epoch*).
- **Vérifiez votre compréhension** en expliquant à voix haute les concepts à quelqu'un d'autre (technique de Feynman).
