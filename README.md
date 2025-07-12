# Projet de Prédiction des Prix de l'Immobilier

Ce projet a pour but de prédire les prix de l'immobilier en utilisant le jeu de données "Boston Housing". Nous allons explorer les données, prétraiter les caractéristiques, entraîner plusieurs modèles de régression et évaluer leurs performances.

## Comment exécuter

1.  **Clonez le dépôt** :
    ```bash
    git clone <URL_DU_DEPOT>
    cd predictionimmobilier
    ```

2.  **Créez un environnement virtuel** (recommandé) :
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3.  **Installez les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

4.  **Exécutez les scripts** :

    *   Pour prétraiter les données :
        ```bash
        python3 scripts/preprocess_data.py
        ```

    *   Pour entraîner et évaluer le modèle de Régression Linéaire :
        ```bash
        python3 scripts/train_model.py
        ```

    *   Pour entraîner et évaluer le modèle Random Forest :
        ```bash
        python3 scripts/train_random_forest.py
        ```

## 1. Structure du Projet

Le projet est organisé comme suit :

```
/predictionimmobilier
|-- /data
|   |-- /raw
|   |   `-- train.csv         # Données brutes
|   `-- /processed
|       |-- X_train.csv       # Caractéristiques d'entraînement prétraitées
|       |-- X_test.csv        # Caractéristiques de test prétraitées
|       |-- y_train.csv       # Variable cible d'entraînement
|       `-- y_test.csv        # Variable cible de test
|-- /notebooks
|   `-- data_exploration.ipynb # Notebook pour l'analyse exploratoire
|-- /scripts
|   |-- eda_script.py
|   |-- preprocess_data.py
|   |-- train_model.py
|   `-- train_random_forest.py
|-- requirements.txt
`-- README.md
```

## 2. Analyse Exploratoire des Données

L'analyse initiale a révélé plusieurs informations clés :

*   **Distribution de la variable cible (`MEDV`)** : La distribution des prix des maisons est approximativement normale, avec quelques valeurs aberrantes à l'extrémité supérieure.
*   **Corrélations** : La matrice de corrélation a montré que les caractéristiques `RM` (nombre moyen de pièces par logement) et `LSTAT` (pourcentage de la population de statut inférieur) sont fortement corrélées avec la variable cible `MEDV`.
    *   `RM` a une forte corrélation positive (0.70), ce qui suggère que plus il y a de pièces, plus le prix de la maison est élevé.
    *   `LSTAT` a une forte corrélation négative (-0.74), ce qui suggère que plus le pourcentage de population de statut inférieur est élevé, plus le prix de la maison est bas.

## 3. Prétraitement des Données

Le prétraitement a consisté en trois étapes principales :

1.  **Séparation des données** : Les données ont été divisées en caractéristiques (`X`) et en variable cible (`y`).
2.  **Division en ensembles d'entraînement et de test** : Les données ont été divisées en un ensemble d'entraînement (80%) et un ensemble de test (20%) pour permettre une évaluation impartiale du modèle.
3.  **Mise à l'échelle des caractéristiques** : `StandardScaler` de Scikit-learn a été utilisé pour mettre à l'échelle les caractéristiques. Cette étape est cruciale car elle garantit que toutes les caractéristiques ont une moyenne de 0 et un écart-type de 1, ce qui empêche les caractéristiques avec des échelles plus grandes de dominer le processus d'apprentissage du modèle.

## 4. Modélisation et Évaluation

Deux modèles de régression ont été entraînés et évalués :

1.  **Régression Linéaire** : Un modèle de base simple pour établir une performance de référence.
2.  **Random Forest Regressor** : Un modèle d'ensemble plus complexe et puissant.

Les performances ont été évaluées à l'aide de deux métriques :
*   **Mean Squared Error (MSE)** : Mesure l'erreur quadratique moyenne entre les valeurs prédites et réelles. Un MSE plus faible est meilleur.
*   **R-squared (R2)** : Représente la proportion de la variance de la variable dépendante qui est prévisible à partir des variables indépendantes. Un R2 plus proche de 1 est meilleur.

### Résultats

| Modèle                  | Mean Squared Error (MSE) | R-squared (R2) |
| ----------------------- | ------------------------ | -------------- |
| Régression Linéaire     | 24.29                    | 0.67           |
| Random Forest Regressor | 7.91                     | 0.89           |

Le modèle **Random Forest Regressor** a surpassé de manière significative le modèle de Régression Linéaire, avec un R2 de **0.89**, indiquant qu'il explique 89% de la variance des prix de l'immobilier dans l'ensemble de test.

## 5. Conclusion et Pistes d'Amélioration

Ce projet a démontré avec succès la capacité à prédire les prix de l'immobilier à l'aide de techniques de machine learning. Le modèle Random Forest s'est avéré être un prédicteur très efficace pour ce jeu de données.

Pour améliorer davantage les performances, les étapes suivantes pourraient être envisagées :

*   **Optimisation des hyperparamètres** : Utiliser des techniques comme `GridSearchCV` ou `RandomizedSearchCV` pour trouver les meilleurs hyperparamètres pour le modèle Random Forest.
*   **Ingénierie des caractéristiques** : Créer de nouvelles caractéristiques en combinant ou en transformant les caractéristiques existantes.
*   **Explorer d'autres modèles** : Tester des modèles de boosting de gradient comme XGBoost ou LightGBM, qui sont souvent très performants dans les compétitions de machine learning.