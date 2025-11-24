# CropClassification

Classification supervisée de cultures agricoles avec pipeline ML complet (prétraitement, entraînement avec validation croisée 10-fold, tuning d'hyperparamètres et évaluation avancée).

---

##  Objectif du projet

L'objectif est de construire un modèle de **classification de cultures** à partir de variables numériques (caractéristiques agro‑climatiques, spectrales, ou autres), en mettant l'accent sur :

- une **pipeline claire et réutilisable** en notebooks ;
- une **évaluation robuste** via **Stratified K-Fold (k = 10)** ;
- un **comparatif de plusieurs modèles** (KNN, Random Forest, SVM, MLP, etc.) avec recherche d'hyperparamètres ;
- des **métriques complètes** et des **visualisations** permettant d'interpréter les performances.


---

##  Structure du projet

```text
CropClassification/
├── README.md                 # Ce fichier
└── deepnote/
		└── CropClassification/
				├── 1-EDA.ipynb          # Exploration des données (EDA)
				├── 2-preprocessing.ipynb# Prétraitement & split train/test + scaling
				├── 3-model_training.ipynb# Entraînement, CV 10-fold, tuning & sauvegarde du meilleur modèle
				└── 4-model_evaluation.ipynb# Évaluation détaillée du meilleur modèle
```

Les dossiers générés lors de l'exécution des notebooks (non tous présents dans le repo brut) :

```text
data/                         # Données prétraitées (X_train_scaled.npy, ...)
models/
		├── label_encoder.pkl
		└── tuned/
				└── <best_model>_best.pkl
results/
		├── tuning/               # Résultats comparatifs des modèles
		├── metrics/              # CSV des métriques détaillées
		└── visualizations/       # Graphiques (matrice de confusion, ROC, etc.)
```

---

##  Méthodologie ML

### 1. EDA (`1-EDA.ipynb`)

- Chargement des données brutes
- Inspection des types, distributions et éventuelles valeurs manquantes
- Visualisations de base (distributions, corrélations, répartition des classes, etc.)

### 2. Prétraitement (`2-preprocessing.ipynb`)

- Encodage de la variable cible avec `LabelEncoder` (sauvegardé dans `models/label_encoder.pkl`)
- Split **train / test**
- Standardisation / normalisation des features (ex. `StandardScaler`)
- Sauvegarde des jeux prétraités :
	- `data/X_train_scaled.npy`
	- `data/X_test_scaled.npy`
	- `data/y_train.npy`
	- `data/y_test.npy`

### 3. Entraînement & Tuning (`3-model_training.ipynb`)

- Chargement de `X_train`, `X_test`, `y_train`, `y_test`
- Définition d'un **Stratified K-Fold** avec **k = 10** :
	- `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`
- Définition des modèles de base :
	- `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`,
	- `GaussianNB`, `SVC`, `LogisticRegression`, `MLPClassifier`,
	- `GradientBoostingClassifier`, `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis`.
- Stratégie d'optimisation :
	- **GridSearchCV** pour : KNN, Logistic Regression
	- **RandomizedSearchCV** pour : Decision Tree, Random Forest, SVM, MLP, Gradient Boosting
	- Entraînement par défaut + `cross_val_score` (10-fold) pour : LDA, QDA, Naive Bayes
- Pour chaque modèle :
	- meilleur score de **CV 10-fold**, accuracy train / test, temps d'entraînement
	- stockage des hyperparamètres optimaux et du modèle entraîné
- Création d'un tableau récapitulatif `results_df` et classement des modèles par **Test Accuracy**.
- Sauvegardes :
	- Meilleur modèle : `models/tuned/<model_name>_best.pkl`
	- Résumé : `results/tuning/best_model_info.json`
	- Résultats complets : `results/tuning/tuning_results.csv`, `results/tuning/all_models_performance.csv`
	- Visualisation comparative : `results/visualizations/training_results.png`

### 4. Évaluation détaillée (`4-model_evaluation.ipynb`)

- Rechargement du meilleur modèle et des données prétraitées
- Prédictions sur train et test
- Calcul de métriques :
	- Accuracy (train / test)
	- Precision, Recall, F1 **macro** et **weighted**
	- `classification_report` complet (par classe)
- Visualisations :
	- Matrice de confusion (sauvegardée en PNG)
	- Barplots des principales métriques (precision/recall/F1/accuracy)
	- Courbes ROC **One-vs-Rest** et **AUC micro/macro/weighted** si le modèle fournit des probabilités
	- Barplot des AUC par classe
- Sauvegardes :
	- `results/metrics/best_model_metrics.csv`
	- `results/metrics/classification_report_<model>.csv`
	- `results/metrics/best_model_auc.csv`, `best_model_auc_per_class.csv`
	- Graphiques dans `results/visualizations/`

---

##  Environnement & dépendances

Créer un environnement virtuel (optionnel mais recommandé) :

```bash
python -m venv .venv
source .venv/bin/activate  # sous Linux/macOS
# ou
.venv\Scripts\activate    # sous Windows (PowerShell/CMD)
```

Installer les dépendances minimales (adapter les versions si besoin) :

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib jupyter
```

Tu peux aussi créer un `requirements.txt` avec ces paquets si tu veux geler les versions.

---

##  Comment exécuter le projet

1. **Ouvrir les notebooks** dans VS Code, Jupyter ou Deepnote.
2. **Ordre recommandé d'exécution** :
	 1. `1-EDA.ipynb` (optionnel mais recommandé pour comprendre les données)
	 2. `2-preprocessing.ipynb` (génère les fichiers `.npy` et le `label_encoder`)
	 3. `3-model_training.ipynb` (entraîne tous les modèles, tuning + sauvegarde du meilleur)
	 4. `4-model_evaluation.ipynb` (évalue en détail le meilleur modèle et génère toutes les figures)
3. Consulter :
	 - les CSV dans `results/tuning/` et `results/metrics/`
	 - les graphiques dans `results/visualizations/`

---

##  Résultats (à compléter)

Tu peux résumer ici, après exécution des notebooks, par exemple :

- **Meilleur modèle** : NaÏve Bayes
- **Accuracy (test)** : 0.9955
- **CV 10-fold (moyenne)** : 0.9949


---

##  Points forts de la pipeline

- Validation croisée **stratifiée 10-fold** utilisée systématiquement
- Comparaison de nombreux modèles classiques de ML supervisé
- Recherche d'hyperparamètres (GridSearchCV + RandomizedSearchCV) adaptée à la complexité de chaque modèle
- Sauvegarde structurée des modèles, métriques et visualisations
- Évaluation très complète (métriques globales, par classe, ROC/AUC, confusion matrix)

