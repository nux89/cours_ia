# Ateliers — Transfert, captioning et diffusion sur CPU

Ces trois ateliers rendent observables les mécanismes du cours. Ils utilisent des données locales ou synthétiques, sans poids à télécharger ni API. Ils ne mesurent pas les performances d'un modèle de fondation. Prévoir 6 à 10 heures, exercices compris, après les sections 15 à 17 du [cours de deep learning](cours_deep_learning.md).

## 1. Transfert et fine-tuning : comparer trois stratégies

Le [TP transfert](06_transfert_fine_tuning.ipynb) préentraîne un extracteur sur les chiffres `digits` 0 à 4, puis adapte une nouvelle tête aux chiffres 5 à 9. Les populations source et cible sont donc disjointes par classes ; le test cible est également exclu du train cible. Ce choix rend le protocole explicite, mais ne représente qu'un petit transfert entre tâches proches.

L'extracteur transforme 64 pixels en un vecteur de 32 caractéristiques. La tête source prédit cinq classes. On retire cette tête, conserve les poids de l'extracteur, initialise une nouvelle tête et compare :

| Stratégie | Extracteur | Tête cible | Question |
|---|---|---|---|
| Depuis zéro | aléatoire, entraîné | entraînée | quelle baseline sans préentraînement ? |
| Extracteur figé | préentraîné, inchangé | entraînée | les features sont-elles transférables ? |
| Fine-tuning | préentraîné, entraîné à petit pas | entraînée à pas plus grand | faut-il adapter la représentation ? |

La loss est l'entropie croisée ; les logits ont la forme [B,5]. Les comparaisons utilisent les mêmes observations, la même graine d'initialisation de la tête et un budget fixé à l'avance. Ce budget n'est pas égal en coût total : le préentraînement a un coût supplémentaire. Les hyperparamètres sont pédagogiques et ne sont pas choisis sur le test.

Un contrôle compare les poids avant/après et exige que l'extracteur figé soit strictement inchangé. Le test mesure l'exactitude des trois stratégies sans imposer que le transfert gagne. Une petite source peut être moins utile qu'un entraînement direct ; ce résultat serait instructif.

Exercice : réduire le nombre de labels cibles et répéter sur plusieurs graines, sans sélectionner la meilleure sur le test. Correction attendue : rapporter moyenne et dispersion ; discuter séparation des effets de préentraînement, budget et variance. Pour une comparaison publiable, ajouter validation et recherche d'hyperparamètres avec budget comparable.

## 2. Image-to-caption : un encodeur CNN et un décodeur GRU

Le [TP captioning](07_captioning_cnn_gru.ipynb) crée de petites images RGB de carrés et barres, en rouge ou vert. Leur position et le bruit varient. Les légendes décrivent deux attributs : forme et couleur. Une GRU est utilisée pour montrer que le principe CNN→décodeur séquentiel ne dépend pas du choix LSTM/GRU.

Le pipeline est : image [B,3,12,12] → CNN → état [B,32] → état initial du décodeur [1,B,32]. Le décodeur reçoit un token à chaque pas et produit des logits sur le vocabulaire. Il apprend P(mot_t | image, mots précédents).

Le vocabulaire contient PAD, BOS, EOS et les mots des légendes. Exemple : entrée `[BOS, carre, rouge]`, cible `[carre, rouge, EOS]`. La fonction de perte masque PAD ; les exemples ont ici une longueur identique mais le code explicite le traitement des séquences rembourrées.

Pendant le **teacher forcing**, les mots précédents sont les vrais mots. Pendant la génération, on réinjecte la prédiction. Cette différence crée un risque d'accumulation d'erreurs. Le décodage greedy choisit l'argmax ; beam search explore plusieurs préfixes mais ne garantit pas une légende visuellement correcte.

Le test est constitué d'images nouvellement générées, issues de la même distribution. Il mesure la correspondance exacte de la légende complète ; un second essai met les images à zéro pour vérifier la dépendance au signal visuel. Il ne démontre pas une généralisation compositionnelle : toutes les combinaisons forme/couleur peuvent être vues à l'entraînement.

Exercice : retenir une combinaison forme/couleur uniquement pour le test. Correction attendue : séparer généralisation à de nouveaux pixels et recombinaison de concepts. Ajouter une légende erronée mais grammaticalement correcte pour montrer pourquoi la fluidité n'est pas une mesure de fidélité.

## 3. Diffusion : apprendre à débruiter une distribution 2D

Le [TP diffusion](08_diffusion_2d.ipynb) entraîne un petit DDPM sur huit amas de points. Deux dimensions permettent d'observer le mécanisme sans entraîner un générateur d'images.

Pour t dans {0,…,T−1}, β_t définit la variance ajoutée, α_t=1−β_t et ᾱ_t=∏_(s≤t)α_s. Le bruitage direct s'écrit : x_t=√ᾱ_t x_0 + √(1−ᾱ_t) ε, avec ε normale standard.

Le réseau reçoit x_t et un encodage de t, et prédit ε. Sa loss moyenne est ||ε−ε_θ(x_t,t)||². Le temps est nécessaire : un point légèrement bruité et un point presque gaussien ne demandent pas la même correction.

À la génération, on part d'un bruit indépendant, puis on applique les transitions inverses apprises. Le TP utilise la moyenne DDPM et la variance postérieure β̃_t=β_t(1−ᾱ_(t−1))/(1−ᾱ_t), avec ᾱ_(-1)=1 et sans bruit ajouté au dernier pas. La grille de bruit est choisie pour que ᾱ_(T−1) soit proche de zéro ; sinon démarrer avec une normale serait une approximation plus grossière.

On compare la loss de débruitage sur un test indépendant à la prédiction ε=0, puis on affiche données réelles et échantillons générés. Distance aux centres et couverture des huit modes sont des diagnostics adaptés à ce jouet, pas des garanties de qualité générale. Un faible MSE de bruit n'assure pas à lui seul diversité et absence de mémorisation.

Exercice : retirer le temps des entrées ou réduire le nombre de pas. Correction attendue : évaluer à nouveau débruitage et modes ; ne pas conclure sur une seule figure. Passer à des images nécessiterait un débruiteur convolutionnel, un jeu plus grand et davantage de calcul. Le conditionnement textuel et la guidance restent étudiés dans le cours, sans implémentation dans ce TP.

## 4. Critères communs et sources

Pour chaque atelier, rendre un schéma des formes, une comparaison à baseline, les résultats obtenus, une ablation et une limite de généralisation. Ne pas convertir un exemple fonctionnel en preuve de performance industrielle.

Sources : [PyTorch — transfert](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), [Vinyals et al. — Show and Tell](https://arxiv.org/abs/1411.4555), [Ho et al. — DDPM](https://arxiv.org/abs/2006.11239).
