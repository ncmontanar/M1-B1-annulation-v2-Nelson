# 09 — Les pipelines scikit-learn : l'essentiel

> **Utilisé dans** : M1-B1 (tâche 6), M1-B2 (mise en service), puis les modules 5 et 6.


## 1. Le problème : préparer les données… deux fois

Un modèle ne lit que des nombres, sans valeurs manquantes. Avant de l'entraîner, on prépare donc les données : encoder les catégories, remplir les trous, parfois mettre à l'échelle, créer de nouvelles variables.

Le piège : **ces mêmes transformations doivent être refaites, à l'identique, à chaque prédiction**, par exemple dans l'API. Si elles sont dispersées dans des cellules de notebook, on finit toujours par en oublier une, ou par la refaire un peu différemment. Le modèle reçoit alors des données qui ne ressemblent pas à celles de son entraînement, et il se trompe sans le dire.

Un **`Pipeline`** enchaîne la préparation et le modèle dans **un seul objet** :

```
données brutes ──▶ [ préparation ] ──▶ [ modèle ] ──▶ prédiction
                  └──────────── un seul Pipeline ────────────┘
```

On l'entraîne avec `fit`, on prédit avec `predict`, et on le sauvegarde d'un bloc avec `joblib`. C'est exactement ce que contenait le `.joblib` que tu as exposé en M0-B1 : c'est pour cela que l'API pouvait lui envoyer des données brutes.

## 2. Les briques de préparation

| Besoin | Outil | Remarque |
|---|---|---|
| Remplir les valeurs manquantes | `SimpleImputer(strategy="median")` ou `strategy="most_frequent"`, `"constant"` | |
| Encoder une catégorie | `OneHotEncoder(handle_unknown="ignore")` | une colonne 0/1 par valeur ; `handle_unknown` évite de planter sur une valeur jamais vue |
| Regrouper les catégories rares | `OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist")` | utile pour des colonnes à beaucoup de valeurs, comme un pays |
| Mettre à l'échelle | `StandardScaler()` | indispensable pour une régression logistique ; inutile pour les arbres (forêt, gradient boosting) |

Chaque brique suit la même logique :

- `fit` **apprend** quelque chose sur les données (la médiane, la liste des catégories, la moyenne et l'écart-type) ;
- `transform` **applique** ce qui a été appris.

C'est pour cela qu'on ne fait `fit` que sur les données d'entraînement : sinon, des informations du jeu de test se glissent dans la préparation.

## 3. `ColumnTransformer` : une préparation par type de colonne

Les colonnes numériques et catégorielles ne se préparent pas de la même façon. `ColumnTransformer` applique chaque traitement aux bonnes colonnes, puis recolle le tout :

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeriques = ["nb_nuits", "prix"]
categorielles = ["type_chambre", "canal"]

preparation = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeriques),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ]), categorielles),
])
```

⚠️ Par défaut, **les colonnes qui ne sont listées nulle part sont supprimées** (`remainder="drop"`). C'est pratique (les colonnes inutiles disparaissent), mais une colonne oubliée l'est silencieusement.

Puis on ajoute le modèle :

```python
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("preprocessing", preparation),
    ("classifier", LogisticRegression(max_iter=1000)),
])

pipeline.fit(X_train, y_train)        # X_train : un DataFrame de données brutes
pipeline.predict(X_nouveau)         #X_test = le set de test à évaluer  ou 
pipeline.predict(X_test)            # prédictions sur le jeu de test
```

Pour voir ce que la préparation produit : `pipeline[:-1].fit_transform(X_train)` (tout sauf le modèle) et `pipeline[:-1].get_feature_names_out()`.

## 4. Créer de nouvelles variables dans le pipeline

Une nouvelle variable (le *feature engineering*) doit, elle aussi, être calculée **dans** le pipeline, pour être recalculée automatiquement en production.

L'outil le plus simple : une fonction Python qui reçoit le `DataFrame` brut et renvoie un `DataFrame` enrichi, enveloppée dans un `FunctionTransformer`.

```python
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


def ajouter_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()                                   # ne jamais modifier l'entrée
    df["prix_total"] = df["prix"] * df["nb_nuits"]
    df["sejour_long"] = (df["nb_nuits"] >= 7).astype(int)
    return df


pipeline = Pipeline([
    ("variables", FunctionTransformer(ajouter_variables)),
    ("preprocessing", preparation),                  # qui liste aussi prix_total et sejour_long
    ("classifier", LogisticRegression(max_iter=1000)),
])
```

Trois règles :

- **Une fonction nommée, pas une `lambda`** : `joblib` ne sait pas sauvegarder une `lambda`.
- **La fonction doit être importable là où on charge le modèle.** `joblib` enregistre une *référence* à la fonction, pas son code : si le modèle est chargé dans l'API, l'API doit pouvoir importer la fonction. Place-la dans un module Python (par exemple `features.py`), pas dans un notebook.
- **La variable doit être calculable au moment de la prédiction**, uniquement à partir de ce que l'on connaît à cet instant-là.

## 5. Choisir le modèle

Pour une classification sur des données tabulaires, trois familles à comparer :

| Modèle | Points forts | Points d'attention |
|---|---|---|
| `LogisticRegression` | rapide, interprétable (coefficients) | nécessite la mise à l'échelle ; relations seulement linéaires |
| `RandomForestClassifier` | robuste, peu de réglages, gère les interactions | modèle volumineux, plus lent à prédire |
| `HistGradientBoostingClassifier` | souvent le meilleur sur tabulaire, rapide, gère les valeurs manquantes | plus d'hyperparamètres |

L'avantage du pipeline : **changer de modèle ne change que la dernière étape**.

## 6. Exercice guidé — un premier pipeline sur les données de M1-B1

Il ne s'agit pas encore de ton pipeline final, mais d'un pipeline minimal qui tourne.

**Étape 1 — Partir petit.** Prends 1 000 lignes de ton jeu d'entraînement, 3 colonnes numériques et 2 colonnes catégorielles de ton choix.

**Étape 2 — Construire la préparation.** Un `ColumnTransformer` avec imputation et encodage. Vérifie ce qu'il produit :

```python
preparation.fit_transform(X_petit).shape
preparation.get_feature_names_out()[:10]
```

✅ *Autant de lignes qu'en entrée ; des noms de colonnes du type `cat__colonne_valeur`.*

**Étape 3 — Ajouter un modèle et entraîner.** Puis prédis sur 5 autres lignes **brutes**, sans les préparer toi-même.

✅ *`predict` renvoie 5 valeurs 0 ou 1, sans erreur.*

**Étape 4 — Ajouter une variable.** Écris une fonction `ajouter_variables` qui crée une variable simple, glisse-la en tête du pipeline avec `FunctionTransformer`, et ajoute la nouvelle colonne à la liste du `ColumnTransformer`.

**Étape 5 — Sauvegarder et recharger.**

```python
import joblib
joblib.dump(pipeline, "test_pipeline.joblib")
recharge = joblib.load("test_pipeline.joblib")
recharge.predict(X_cinq_lignes)
```

✅ *Mêmes prédictions qu'avant la sauvegarde.* Puis passe à tout le jeu d'entraînement et aux colonnes de ton choix.

## 7. Pièges fréquents

| Symptôme | Cause probable |
|---|---|
| `ValueError: could not convert string to float` | Une colonne texte n'est pas passée par l'encodeur (oubliée dans le `ColumnTransformer`) |
| `ValueError: Input X contains NaN` | Pas d'imputation sur une colonne qui a des valeurs manquantes |
| `Found unknown categories ... during transform` | `OneHotEncoder` sans `handle_unknown="ignore"` : une valeur du test n'existait pas à l'entraînement |
| `A given column is not a column of the dataframe` | Faute de frappe dans une liste de colonnes, ou variable créée **après** le `ColumnTransformer` au lieu d'avant |
| `PicklingError` ou `Can't pickle <lambda>` | Une `lambda` dans le pipeline : la remplacer par une fonction nommée |
| `AttributeError: Can't get attribute 'ajouter_variables'` au chargement | La fonction n'est pas importable là où on charge le modèle : la mettre dans un module |
| `ConvergenceWarning` (régression logistique) | Données non mises à l'échelle, ou `max_iter` trop faible |
| Le score est bon en validation, mais les prédictions sont absurdes en production | Préparation faite à la main hors du pipeline, et refaite différemment ailleurs |

## Pour aller plus loin

- `ColumnTransformer` et données hétérogènes : https://scikit-learn.org/stable/modules/compose.html
- Prétraitement des données : https://scikit-learn.org/stable/modules/preprocessing.html
