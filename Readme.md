
# 🫁 ChestScan AI Pro

> Classification de radiographies thoraciques (**NORMAL** vs **PNEUMONIA**) par Deep Learning, avec une interface web Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Status](<https://img.shields.io/badge/Usage-Éducatif%20uniquement-yellow>)

> ⚠️ **Avertissement** : ce projet est un **prototype pédagogique et expérimental**. Il n'est **pas** un dispositif médical et ne doit **jamais** être utilisé pour un diagnostic. Toute décision clinique doit être prise par un professionnel de santé qualifié.

---

## 📑 Sommaire

1. [Présentation](#-présentation)
2. [Aperçu de l&#39;application](#-aperçu-de-lapplication)
3. [Dataset](#-dataset)
4. [Prétraitement](#-prétraitement)
5. [Modèles](#-modèles)
6. [Résultats et comparaison](#-résultats-et-comparaison)
7. [Analyse et limites](#-analyse-et-limites)
8. [Installation et exécution](#-installation-et-exécution)
9. [Structure du projet](#-structure-du-projet)
10. [Précautions d&#39;utilisation](#-précautions-dutilisation)
11. [Pistes d&#39;amélioration](#-pistes-damélioration)

---

## 🎯 Présentation

**ChestScan AI** est un prototype développé dans le cadre d'un projet Deep Learning : une startup d'imagerie médicale souhaite distinguer automatiquement les radiographies thoraciques **normales** de celles associées à une **pneumonie**.

Le projet couvre toute la chaîne, des données jusqu'au déploiement :

- Chargement, contrôle qualité et exploration du dataset
- **CNN construit à partir de zéro**
- **Transfer Learning** avec MobileNetV2 (base gelée)
- **Fine-tuning** progressif de MobileNetV2
- Évaluation comparative des trois approches sur le même jeu de test
- Sauvegarde des modèles (`.keras`) et fonction de prédiction
- Application **Streamlit** pour tester de nouvelles radiographies

---

## 🖼️ Aperçu de l'application

|            Page d'accueil            |           Résultat : Normal           |            Résultat : Pneumonie            |
| :-----------------------------------: | :------------------------------------: | :------------------------------------------: |
| ![Accueil](docs/screenshots/home.png) | ![Normal](docs/screenshots/normal.png) | ![Pneumonie](docs/screenshots/pneumonia.png) |

**Fonctionnalités de l'interface**

- Import d'une radiographie (JPG, JPEG, PNG)
- Affichage de l'image d'origine
- Classe prédite (NORMAL / PNEUMONIA) avec niveau de confiance
- Probabilités détaillées pour chaque classe + barre de progression
- Carte d'attention illustrative (voir [limites](#-analyse-et-limites))
- Design moderne (glassmorphism), entièrement local : aucune image n'est envoyée à un serveur

---

## 🗂️ Dataset

**Source :** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kaggle (Paul Mooney). Téléchargement automatique via `kagglehub` dans le notebook.

| Ensemble      | Images | NORMAL | PNEUMONIA |
| ------------- | -----: | -----: | --------: |
| Entraînement |  5 216 |  1 341 |     3 875 |
| Validation    |     16 |      8 |         8 |
| Test          |    624 |    234 |       390 |

- **Contrôle qualité** : toutes les images ont été vérifiées (décodage) → **0 image invalide**.
- **Déséquilibre** : environ 74 % de pneumonies dans l'entraînement. Il est compensé par des **poids de classes** (`NORMAL ≈ 1.94`, `PNEUMONIA ≈ 0.67`).
- **Indépendance du test** : le jeu de test fourni par le dataset n'est jamais utilisé pendant l'entraînement.
- La validation officielle ne contient que 16 images : les métriques de validation sont donc **peu fiables** (voir limites).

---

## ⚙️ Prétraitement

| Étape           | CNN                                                  | MobileNetV2 (TL & FT)                                        |
| ---------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| Taille d'entrée | 180 × 180                                           | 180 × 180 (redimensionnement intégré au modèle)          |
| Normalisation    | pixels / 255 → `[0, 1]`                           | `Rescaling(2.0, -1.0)` intégré au modèle → `[-1, 1]` |
| Canaux           | RGB (3)                                              | RGB (3)                                                      |
| Augmentation     | Retournement horizontal, rotation ±5 %, zoom ±10 % | idem                                                         |

> 💡 Pour MobileNetV2, le redimensionnement et la normalisation font **partie du modèle** : l'image doit donc lui être fournie en pixels bruts `[0, 255]`.

---

## 🧠 Modèles

### 1. CNN from scratch

```
Input (180×180×3)
→ Conv2D(32, 3×3, ReLU) → MaxPooling2D
→ Conv2D(64, 3×3, ReLU) → MaxPooling2D
→ … blocs convolutifs …
→ Flatten (118 336)
→ Dense(128, ReLU) → Dropout
→ Dense(1, Sigmoid)
```

- **15,17 M paramètres**, tous entraînables
- Perte : `binary_crossentropy` · Optimiseur : Adam
- 5 époques, batch size 8, poids de classes

### 2. Transfer Learning — MobileNetV2 (base gelée)

```
Resizing(180) → Rescaling([-1,1]) → MobileNetV2 (ImageNet, gelé)
→ GlobalAveragePooling2D → Dropout(0.2) → Dense(1, Sigmoid)
```

- **2,26 M paramètres**, dont seulement **1 281 entraînables** (tête de classification)
- Adam, `lr = 1e-3`, 5 époques

### 3. Fine-tuning — MobileNetV2

- Mêmes architecture et prétraitement que le Transfer Learning
- Dégel des **30 dernières couches** de MobileNetV2 (blocs 14 à 16 + `Conv_1`)
- Taux d'apprentissage très faible (`1e-7`) pour préserver les poids pré-entraînés

### Callbacks communs

| Callback              | Paramètres                                                |
| --------------------- | ---------------------------------------------------------- |
| `EarlyStopping`     | `val_loss`, patience 3, restauration des meilleurs poids |
| `ReduceLROnPlateau` | facteur 0.1, patience 2, `min_lr = 1e-7`                 |
| `ModelCheckpoint`   | sauvegarde du meilleur modèle selon `val_loss`          |

---

## 📊 Résultats et comparaison

Évaluation sur le **même jeu de test (624 images)**, seuil de décision 0.5, classe positive = PNEUMONIA.

| Modèle                     |         Accuracy |        Precision |           Recall |         F1-score |          ROC-AUC |             Loss |
| --------------------------- | ---------------: | ---------------: | ---------------: | ---------------: | ---------------: | ---------------: |
| CNN from scratch            |           0.7436 |           0.7106 | **0.9949** |           0.8291 |           0.9159 |           1.2803 |
| **Transfer Learning** | **0.8926** | **0.8646** |           0.9821 | **0.9196** | **0.9670** | **0.2951** |
| Fine-tuning                 |           0.6474 |           0.7136 |           0.7282 |           0.7208 |           0.6667 |           0.6317 |

**Détail du CNN par classe**

| Classe    | Precision | Recall |   F1 | Support |
| --------- | --------: | -----: | ---: | ------: |
| NORMAL    |      0.97 |   0.32 | 0.49 |     234 |
| PNEUMONIA |      0.71 |   0.99 | 0.83 |     390 |

**Analyse du seuil de décision (Fine-tuning)**

| Seuil | Accuracy | Precision | Recall |    F1 |
| ----: | -------: | --------: | -----: | ----: |
|   0.3 |    0.643 |     0.642 |  0.967 | 0.772 |
|   0.4 |    0.641 |     0.661 |  0.872 | 0.752 |
|   0.5 |    0.647 |     0.714 |  0.728 | 0.721 |
|   0.6 |    0.583 |     0.764 |  0.482 | 0.591 |
|   0.7 |    0.490 |     0.810 |  0.241 | 0.372 |

### 🏆 Modèle recommandé : Transfer Learning (MobileNetV2 gelé)

C'est le meilleur sur presque toutes les métriques (accuracy 89,3 %, F1 91,96 %, AUC 0,967) tout en restant léger et rapide à entraîner.

---

## 🔍 Analyse et limites

**Lecture des résultats**

- Le **CNN from scratch** détecte presque toutes les pneumonies (recall 99,5 %) mais **rate 68 % des images normales** (recall NORMAL 0,32) : il a tendance à « tout classer en pneumonie ».
- Le **Transfer Learning** apporte un net gain : les caractéristiques apprises sur ImageNet se transfèrent bien aux radiographies, avec un bon équilibre entre sensibilité et précision.
- Le **Fine-tuning** se **dégrade** (220 erreurs sur 624). Hypothèses : sur-adaptation de la tête et des dernières couches sur un dataset très déséquilibré, interaction avec les couches de *BatchNorm* dégelées, entraînement court et validation de seulement 16 images pour piloter les callbacks.

**Limites de l'expérience**

- **Validation minuscule (16 images)** : `EarlyStopping` et `ReduceLROnPlateau` s'appuient sur un signal très bruité.
- **Entraînement court** : 5 époques seulement.
- **Un seul dataset** (enfants, un seul établissement) : la généralisation à d'autres populations, appareils ou protocoles n'est pas démontrée.
- **Déséquilibre des classes** et décalage de distribution entre entraînement et test.
- **Pas de validation croisée** ni d'intervalles de confiance sur les métriques.
- **Carte d'attention simulée** : la « heatmap » de l'application est **générée de façon illustrative** (formes floues positionnées selon la prédiction). Ce n'est **pas** un vrai Grad-CAM et elle **ne reflète pas** les zones réellement utilisées par le réseau.
- Les scores de confiance affichés sont des **probabilités de sortie du modèle**, pas une garantie clinique ; un modèle peut être très confiant et se tromper.

---

## 🚀 Installation et exécution

### Prérequis

- Python 3.10+
- `pip`
- (Optionnel) GPU pour ré-entraîner les modèles

### 1. Cloner et installer

```bash
git clone <url-du-depot>
cd <nom-du-depot>

python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

pip install -r requirements.txt
```

**Dépendances de l'application** (`requirements.txt`) : `streamlit`, `tensorflow`, `numpy`, `Pillow`.
**Dépendances supplémentaires du notebook** : `kagglehub`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.

> 📌 Pour la reproductibilité, figez vos versions avec `pip freeze > requirements.txt`.

### 2. Obtenir le modèle

Les fichiers `*.keras` sont ignorés par Git (`.gitignore`). Deux options :

- **Entraîner** : ouvrir et exécuter `ChestScan.ipynb` (le dataset est téléchargé via `kagglehub`, un compte Kaggle peut être requis).
- **Récupérer** un modèle déjà entraîné et le placer à la racine sous le nom `best_pneumonia_model.keras`.

### 3. Lancer l'application

```bash
streamlit run app.py
```

Puis ouvrir [http://localhost:8501](http://localhost:8501), importer une radiographie et lire le résultat.

### 4. Prédire en Python (sans l'interface)

```python
import keras, numpy as np

model = keras.models.load_model("best_pneumonia_model.keras", compile=False)

img = keras.utils.load_img("exemple.jpg", target_size=(180, 180))
x = np.expand_dims(keras.utils.img_to_array(img), 0)   # pixels bruts [0, 255] pour MobileNetV2

p = float(model.predict(x, verbose=0)[0][0])
print("PNEUMONIA" if p >= 0.5 else "NORMAL", f"({p:.2%})")
```

---

## 📁 Structure du projet

```
.
├── app.py                       # Application Streamlit
├── ChestScan.ipynb              # Notebook : prétraitement, CNN, TL, fine-tuning, évaluation
├── requirements.txt             # Dépendances
├── best_pneumonia_model.keras   # Modèle chargé par l'application (ignoré par Git)
├── images/                      # (optionnel) images d'illustration pour la page d'accueil
├── docs/screenshots/            # Captures d'écran du README
├── saved_models/                # Modèles entraînés + model_metadata.json
├── data/                        # Dataset local (ignoré par Git)
├── .gitignore
└── README.md
```

---

## 🩺 Précautions d'utilisation

- Outil **éducatif et expérimental**, sans validation clinique ni certification.
- Ne remplace en aucun cas l'avis d'un radiologue ou d'un médecin.
- Un résultat « NORMAL » **n'exclut pas** une pathologie ; un résultat « PNEUMONIA » **ne constitue pas** un diagnostic.
- N'importez pas de données de patients réels identifiables.
- L'inférence est locale ; l'application ne journalise ni ne transmet les images.

---

## 🔭 Pistes d'amélioration

- Constituer un **vrai jeu de validation** (re-découpage stratifié du train) et utiliser une validation croisée.
- Entraîner plus longtemps, ajuster le learning rate du fine-tuning (ex. `1e-5`) et ne dégeler qu'après convergence de la tête ; garder les couches *BatchNorm* gelées.
- Implémenter un **vrai Grad-CAM** pour une explicabilité fidèle.
- Calibrer les probabilités et choisir un **seuil orienté sensibilité** selon le cas d'usage.
- Tester d'autres architectures (EfficientNet, DenseNet121) et des jeux de données externes.
- Ajouter des tests automatisés et un `model_metadata.json` (taille d'entrée, classes, prétraitement).

---

## 📜 Licence et crédits

- Dataset : Paul Mooney, *Chest X-Ray Images (Pneumonia)*, Kaggle.
- Modèle pré-entraîné : MobileNetV2 (ImageNet), Keras Applications.
- Projet académique — usage éducatif uniquement.
