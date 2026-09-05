# Images, vidéo, son et données multimodales : de la mesure au tenseur

Complément du [cours sur les données](cours_nature_et_preparation_donnees.md). Objectifs : décrire ce que mesure un fichier, choisir une représentation, préserver annotations et synchronisation, éviter les fuites et tester la qualité d'un lot. Prérequis : tableaux NumPy, split train/validation/test, unités et fréquences. Prévoir 3–4 heures avec les exercices. Les architectures qui consomment ces données sont expliquées au module 3 ; les données sont préparées **avant** ce choix.

## 1. Une extension de fichier n'est pas un type sémantique

Un même tableau peut représenter des pixels, des fréquences ou des coordonnées. Conserver le sens, les unités, la calibration et la provenance, pas seulement les nombres.

| Donnée industrielle | Structure et métadonnées indispensables | Traitements et pièges |
|---|---|---|
| Photo RGB, inspection de surface | hauteur, largeur, canaux, exposition, orientation | décodage, couleur, resize ; défaut minuscule perdu par réduction |
| Image thermique / hyperspectrale | bandes, longueurs d'onde, calibration, température | corriger capteur ; ne pas assimiler 100 bandes à RGB |
| Volume médical ou tomographique | voxels, espacement physique, orientation, série/patient | 3D ou coupes ; split par patient, pas par coupe |
| Vidéo / caméras industrielles | images datées, fréquence parfois variable, caméra, session | clips, suivi, mouvement ; frames voisines corrélées |
| Parole, machine, musique | échantillons, fréquence, canaux, gain, session | filtrage, spectrogrammes ; bruit utile ou nuisance selon cible |
| Texte, PDF scanné, formulaire | encodage, langue, ordre de lecture, page et boîtes | OCR, layout, tables ; erreurs de lecture à mesurer |
| LiDAR, nuage de points, maillage | coordonnées, repère, intensité, normales, topologie | recalage, sous-échantillonnage ; préserver distances physiques |
| Géospatial et trajectoires | projection, coordonnées, temps, précision | reprojection, voisinages ; fuite spatiale ou temporelle |
| Graphe et événements | nœuds/arêtes typés, identifiants, temps d'arrivée | agrégations et voisinages disponibles à la date de décision |
| Multimodal | liens entre modalités, disponibilité, droits, horodatages | alignement et fusion ; données manquantes ou contradictoires |

Pandas et Polars servent à gérer le **manifeste** (chemins, identifiants, labels, statistiques), filtrer et joindre les sources. Ils ne remplacent pas les décodeurs d'images, audio et vidéo. Stocker de gros médias en colonnes d'objets peut saturer la mémoire ; préférer des références et un chargement par lots, avec formats et schémas versionnés.

## 2. Images : couleur, géométrie et annotations

### Du fichier aux valeurs

Un JPEG est généralement compressé avec perte ; PNG permet une compression sans perte mais ne garantit pas que le contenu initial l'était. DICOM est un format avec des métadonnées et des modalités multiples, pas simplement « une image de plus ». Un volume NIfTI possède une géométrie physique ; deux tableaux de même forme ne sont pas nécessairement alignés.

Vérifier après décodage : forme, `dtype`, ordre des canaux, valeurs finies, plage et orientation. Une image RGB 8 bits est souvent `[H,W,3]` en `uint8` dans `[0,255]`, mais une image peut être monochrome, 16 bits, flottante ou à canal alpha. Certains outils utilisent BGR. Corriger l'orientation EXIF de façon cohérente avec les annotations ; tenir compte du profil colorimétrique. Des valeurs sRGB ne représentent pas linéairement l'intensité lumineuse : les calculs photométriques peuvent demander une linéarisation.

Convertir en flottants **avant** les opérations qui pourraient déborder en entier. Diviser par 255 ne convient qu'à l'encodage concerné, pas à toute image. En transfert, réutiliser le prétraitement attendu par les poids (couleur, résolution, moyenne/écart-type) ; ne pas le mélanger arbitrairement avec une autre normalisation.

### Redimensionner sans changer la tâche

Étirement : donne la forme requise mais déforme les proportions. Recadrage : conserve l'échelle locale mais peut supprimer l'objet. *Letterbox* : conserve les proportions avec bandes de remplissage ; enregistrer le facteur d'échelle et les décalages pour revenir aux coordonnées initiales.

Exemple : une image de largeur 640 et hauteur 480 est réduite à 320 × 240. Une boîte `(x_min,y_min,x_max,y_max)=(100,50,300,200)` devient `(50,25,150,100)`. Dans un carré 320 × 320 centré, ajouter 40 aux ordonnées : `(50,65,150,140)`. Après crop, borner les boîtes au nouveau cadre et supprimer ou signaler celles devenues invisibles.

Pour un masque de classes, utiliser un rééchantillonnage discret comme le plus proche voisin : une interpolation linéaire entre classe 1 et classe 3 inventerait la classe 2. Les transformations géométriques s'appliquent ensemble à l'image, aux boîtes, aux points et aux masques. Une augmentation n'est valide que si le label reste vrai : retourner un chiffre, un texte ou un côté anatomique peut changer le sens.

### Lots et qualité

Convention PyTorch fréquente : `[B,C,H,W]`, avec lot B et canaux C. Un volume peut devenir `[B,C,D,H,W]`. Un `reshape` ne remplace pas une permutation d'axes. Contrôler visuellement des exemples **après** augmentation, avec annotations superposées. Détecter doublons exacts et quasi-doublons, capteurs absents, surexposition et différences de résolution par classe. Séparer patients, pièces, sites ou sessions avant de créer les variantes augmentées.

## 3. Audio : temps, fréquence et niveau

### Échantillonnage et rééchantillonnage

Un signal échantillonné $x[n]$ associe l'indice $n$ au temps $n/f_s$, où $f_s$ est en hertz (échantillons par seconde). N échantillons couvrent une durée nominale $N/f_s$ ; le dernier est au temps $(N-1)/f_s$. Deux secondes à 16 kHz contiennent 32 000 échantillons **par canal**. Un signal stéréo n'est pas une séquence deux fois plus longue.

Pour représenter sans repliement une composante de fréquence $f$, il faut un échantillonnage au-delà de $2f$ et un filtrage adapté. Passer de 48 à 16 kHz demande un filtre anti-repliement avant décimation : garder un point sur trois conserve sinon des hautes fréquences sous de fausses fréquences basses. Changer seulement l'en-tête ou `sample_rate` modifie l'interprétation temporelle, pas le signal échantillonné. Les filtres pratiques ont une bande de transition ; Nyquist n'est pas une coupure réalisable parfaite.

Fusionner les canaux par moyenne peut annuler deux signaux en opposition de phase. Garder la spatialisation pour localisation, ou justifier le passage mono. Distinguer normalisation de crête, niveau RMS et normalisation de caractéristiques : supprimer les différences d'amplitude peut effacer l'indice d'une panne.

### Spectrogramme : formule et paramètres

La transformée de Fourier à court terme (STFT) analyse des fenêtres successives :

$$X[m,k]=\sum_{n=0}^{L-1}x[mH+n]w[n]e^{-2\pi i kn/N_{FFT}}.$$

$L$ est la longueur de fenêtre, $H$ le pas (*hop*), $w$ la fenêtre (par exemple Hann), $m$ l'indice temporel et $k$ l'indice fréquentiel ; $N_{FFT}\ge L$ peut ajouter du zero-padding. Sans padding aux bords, le nombre de fenêtres est $1+\lfloor(N-L)/H\rfloor$. Pour un signal réel, on conserve souvent $N_{FFT}/2+1$ bins si $N_{FFT}$ est pair. Le bin $k$ correspond à $kf_s/N_{FFT}$ Hz.

À 16 kHz, choisir $L=400$ (25 ms), $H=160$ (10 ms) et $N_{FFT}=512$ donne 257 bins et un espacement de 31,25 Hz. Le zero-padding densifie la grille, mais n'augmente pas la résolution physique offerte par la fenêtre. Une longue fenêtre distingue mieux les fréquences proches mais localise moins bien un événement bref.

Le module $|X|$ décrit l'amplitude ; $|X|^2$ une puissance à un facteur de normalisation près. En décibels, utiliser $20\log_{10}(A/A_{ref})$ pour une amplitude et $10\log_{10}(P/P_{ref})$ pour une puissance, avec un plancher strictement positif. Préciser la référence : « −20 dB » sans référence est ambigu. Un spectrogramme mel agrège les puissances par filtres sur une échelle perceptuelle ; un log-mel ne conserve ni toutes les fréquences ni la phase et n'est pas directement inversible.

### Tâche et évaluation

Transcription : texte aligné ou séquence, WER avec substitutions/insertions/suppressions et conventions de normalisation explicites. Classification de son : labels parfois simultanés, donc multilabel. Détection d'événements : dates début/fin et tolérance temporelle. Diarisation : qui parle quand, distinct de l'identification de la personne. Débruitage : préserver le signal utile et éviter d'inventer de la parole.

Découper par locuteur/session/appareil selon la généralisation visée. Ne pas répartir les fenêtres d'un même enregistrement au hasard entre train et test. Le padding n'est pas une observation de silence : fournir longueurs et masques. Documenter resampling, gains, clipping, détection d'activité vocale et leurs erreurs.

## 4. Vidéo : une séquence datée, pas un sac d'images

Un clip RGB peut être `[T,H,W,C]`, puis `[B,T,C,H,W]` pour un CNN image + modèle temporel, ou `[B,C,T,H,W]` pour certaines convolutions 3D. Les conventions dépendent de l'API. Images clés et prédiction inter-images concernent le codec, pas les labels de la tâche.

À cadence constante $f$, l'image $i$ est approximativement à $i/f$. À cadence variable, utiliser les horodatages de présentation, pas une division par un FPS moyen. Pour une action brève, un sous-échantillonnage trop fort peut supprimer l'événement. Un modèle entraîné sur des frames indépendantes peut reconnaître le décor sans comprendre le mouvement.

Choisir durée du clip, pas de prélèvement et causalité selon l'objectif : classification d'un clip complet ou alerte en ligne sans accès aux frames futures. Appliquer les augmentations spatiales de manière cohérente entre frames lorsque le mouvement doit être conservé. Le flux optique est une estimation de déplacement apparent, sensible aux occlusions et changements d'éclairage, pas une vérité physique automatique.

Synchronisation : si le temps global est $t=a\,t_{capteur}+b$, $b$ corrige le décalage et $a$ la dérive d'horloge. Conserver les unités et le point d'origine. Pour l'intervalle vidéo `[1,00;1,04[` seconde, l'audio 16 kHz correspond aux indices `[16000:16640]` lorsque les origines coïncident. Avec un décalage d'horloge, cette tranche change. Un alignement approximatif doit avoir une tolérance justifiée par la vitesse du phénomène.

## 5. Multimodal : aligner, fusionner, gérer l'absence

Une observation doit avoir un identifiant stable, une cible, une unité de split (patient, session…), des références de médias, des intervalles temporels, les modalités présentes et leur qualité. Exemple de manifeste :

```text
sample_id | session_id | t_start_s | t_end_s | image_uri | audio_uri | text | audio_present | target
clip_001  | session_A  | 1.00      | 1.04    | ...       | ...       | ...  | true          | défaut
```

Ajouter version d'annotation, capteur, licence/consentement, checksum et date de disponibilité. Distinguer « absent », « capteur en panne » et « aucun événement ». Une jointure sur l'identifiant seul peut multiplier les lignes ; vérifier cardinalité, taux de non-appariement et décalage temporel. Une jointure temporelle au plus proche peut utiliser une observation future : en prédiction en ligne, sélectionner une donnée déjà disponible, souvent via une jointure arrière bornée.

Les stratégies de fusion ne sont pas interchangeables :

| Stratégie | Construction | Question à vérifier |
|---|---|---|
| Précoce | concaténer des caractéristiques alignées puis apprendre | unités, résolution, modalités manquantes |
| Tardive | combiner scores de modèles spécialisés | calibration et dépendance des erreurs |
| Cross-attention | requêtes d'une modalité vers clés/valeurs de l'autre | masque, coût mémoire, alignement appris |
| Contrastive | rapprocher embeddings de paires associées | faux négatifs, paires erronées, raccourcis |
| Encodeur–décodeur | encoder un média et générer une autre représentation | perte séquentielle, hallucination, fidélité |

Une image→légende n'est pas une transcription exacte : plusieurs descriptions sont plausibles. Évaluer les objets et relations réellement présents, pas seulement le chevauchement des mots. Une association image–texte peut apprendre les filigranes ou les descriptions répétées plutôt que la sémantique.

Tester séparément chaque modalité, leur combinaison, puis leur retrait ou corruption. Un gain multimodal n'est convaincant qu'avec mêmes splits et budget comparable. Le *modality dropout* masque certaines modalités à l'entraînement pour préparer leur absence ; il ne résout pas automatiquement une absence liée à la classe ou à un changement de population. Signaler l'absence au modèle et prévoir abstention ou mode dégradé.

## 6. Protocole de préparation et contrôle

1. Définir cible, instant de décision, unité statistique et généralisation attendue.
2. Inventorier formats, droits, capteurs, unités et mécanismes de collecte ; conserver les originaux.
3. Décoder et valider sans apprentissage : intégrité, formes, fréquences, valeurs, annotations et alignement.
4. Dédupliquer et créer les splits par groupe/temps avant extraction de clips et augmentation.
5. Apprendre les statistiques et transformations ajustables sur le train uniquement.
6. Charger par lots ; appliquer augmentation au train, transformation déterministe à l'évaluation, puis fournir tenseurs et masques.
7. Auditer quelques lots visuellement/à l'écoute, mesurer erreurs par groupe et versionner les opérations.

Pour des données lourdes : lecture paresseuse, cache de décodage versionné, limite de mémoire et stratégie de reprise. Ne pas cacher silencieusement les fichiers corrompus : comptabiliser exclusions et conséquences sur les classes. Une moyenne de pixels ou un spectrogramme « joli » ne remplace pas l'inspection des cas difficiles.

## 7. Exercices avec réponses attendues

- Une vidéo fournit 100 clips d'un même patient : peut-on les répartir au hasard ? Non pour une évaluation sur de nouveaux patients ; grouper au niveau patient, puis extraire les clips.
- Une image a été retournée mais pas son masque : quel symptôme ? Labels spatialement faux malgré dimensions correctes ; afficher une superposition avant entraînement.
- Deux secondes à 16 kHz, fenêtre 400, hop 160 sans padding : combien de fenêtres ? $1+\lfloor31600/160\rfloor=198$, chacune à 257 bins si FFT 512.
- Une composante à 6 kHz est échantillonnée directement à 8 kHz : quelle fréquence apparente ? 2 kHz par repliement ; un filtre avant réduction aurait dû atténuer la composante.
- Audio absent uniquement dans les exemples de panne : un masque peut-il créer un raccourci ? Oui. Mesurer performances par disponibilité, investiguer collecte et tester un changement du mécanisme d'absence.

Pour pratiquer le couplage image–texte, poursuivre avec le [TP CNN–GRU de captioning](../03_deep_learning/07_captioning_cnn_gru.ipynb) et son [support détaillé](../03_deep_learning/ateliers_avances.md). Il utilise des images synthétiques : il ne valide ni codecs industriels ni parole naturelle.

## Références techniques

- [Tutoriel officiel de rééchantillonnage audio](https://docs.pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html) : filtrage et compromis du resampling.
- [Guide NumPy : broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) : règles de formes.
- [Tutoriel PyTorch sur le transfert](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) : prétraitement et adaptation d'un modèle visuel.
