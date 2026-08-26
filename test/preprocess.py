"""
Pré-traitement des données Open Food Facts pour la prédiction du Nutriscore.

Approche par fonctions : chaque fonction fait une seule
chose et retourne un résultat, sans rien garder en mémoire entre les appels.

Point important : le split train/test est fait AVANT tout calcul appris
sur les données (seuil de valeurs manquantes, imputation...), pour éviter
la fuite de données (data leakage). Ce qui est "appris" sur X_train est
ensuite appliqué tel quel à X_test et à df_inconnu.

Ici : chargement des données, séparation connu/inconnu, split train/test, calcul du % de valeurs manquantes, suppression des colonnes trop manquantes, suppression des lignes totalement vides, et imputation des valeurs manquantes (moyenne pour le numérique, mode pour le catégoriel).
A faire en + : encodage, outliers , autres transformations si nécessaire 
"""
import csv

import pandas as pd
from sklearn.model_selection import train_test_split

from outliers import identify_impossible, apply_impossible_drop


SCORES_CONNUS = ["a", "b", "c", "d", "e"]


def load_data(file_path=r"..\data\en.openfoodfacts.org.products.csv.gz", sample_size=10000, sep="\t", encoding="utf-8"):
    """Charge le CSV et retourne un DataFrame."""
    return pd.read_csv(file_path, sep=sep, encoding=encoding, nrows=sample_size, quoting=csv.QUOTE_NONE)


def load_and_split_known_unknown(file_path, target_col="nutriscore_grade", chunksize=100_000, sep="\t", encoding="utf-8"):
    """
    Lit le CSV par morceaux de `chunksize` lignes (au lieu de tout charger
    d'un coup avec load_data) pour pouvoir traiter un fichier énorme.
    Sépare connu/inconnu à chaque morceau, puis recolle les morceaux entre eux.
    """
    connus, inconnus = [], []
    lecteur = pd.read_csv(
        file_path, sep=sep, encoding=encoding, chunksize=chunksize,
        low_memory=False, quoting=csv.QUOTE_NONE,
    )
    for morceau in lecteur:
        morceau_connu, morceau_inconnu = split_known_unknown(morceau, target_col=target_col)
        connus.append(morceau_connu)
        inconnus.append(morceau_inconnu)
    df_connu = pd.concat(connus, ignore_index=True)
    df_inconnu = pd.concat(inconnus, ignore_index=True)
    return df_connu, df_inconnu


def split_known_unknown(df, target_col="nutriscore_grade"):
    """
    Sépare le dataset en deux :
    - df_connu : lignes où le nutriscore est connu (a, b, c, d, e)
    - df_inconnu : le reste (unknown, not-applicable, NaN...)
    """
    df_connu = df[df[target_col].isin(SCORES_CONNUS)].copy()
    df_inconnu = df[~df[target_col].isin(SCORES_CONNUS)].copy()
    return df_connu, df_inconnu


def split_train_test(df_connu, features, target_col="nutriscore_grade", test_size=0.2, random_state=42):
    """
    Sépare df_connu en train/test. Fait AVANT tout calcul statistique
    (percent_missing, imputation...) pour éviter la fuite de données.
    """
    X = df_connu[features]
    y = df_connu[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def compute_missing_percentage(X_train):
    """
    Calcule le % de valeurs manquantes par colonne, UNIQUEMENT sur X_train
    (jamais sur X_test ni df_inconnu, pour ne pas apprendre sur eux).
    """
    return (X_train.isnull().sum() * 100 / len(X_train)).sort_values(ascending=False)


def get_columns_to_drop(percent_missing, threshold=50):
    """Retourne la liste des colonnes à retirer, décidée sur X_train."""
    return percent_missing[percent_missing.values > threshold].index


def apply_column_drop(df, columns_to_drop):
    """
    Applique la MÊME liste de colonnes à retirer sur n'importe quel
    DataFrame (X_train, X_test ou df_inconnu). errors='ignore' évite un
    crash si une colonne n'existe pas (utile pour df_inconnu).
    """
    return df.drop(columns=columns_to_drop, errors="ignore")


def drop_rows_all_missing(df):
    """
    Supprime les lignes où TOUTES les colonnes sont manquantes.
    Applicable à X_train (perte de lignes acceptable) ET à df_inconnu
    (une ligne 100% vide est de toute façon impossible à prédire).
    """
    nb_colonnes = df.shape[1]
    return df[df.isna().sum(axis=1) != nb_colonnes]


def fit_imputation_values(X_train):
    """
    Calcule les valeurs d'imputation (moyenne pour le numérique, mode
    pour le catégoriel) UNIQUEMENT sur X_train. Retourne un dict
    {colonne: valeur} à réutiliser tel quel ailleurs (transform).
    """
    numerical_cols = X_train.select_dtypes(include="number").columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns

    valeurs = {}
    for col in numerical_cols:
        valeurs[col] = X_train[col].mean()
    for col in categorical_cols:
        mode = X_train[col].mode()
        valeurs[col] = mode.iloc[0] if not mode.empty else None
    return valeurs


def apply_imputation(df, valeurs_imputation):
    """
    Applique des valeurs d'imputation déjà apprises (via fit_imputation_values)
    à n'importe quel DataFrame (X_train, X_test, df_inconnu). Ne recalcule
    jamais rien à partir de df lui-même.
    """
    df = df.copy()
    for col, valeur in valeurs_imputation.items():
        if col in df.columns:
            df[col] = df[col].fillna(valeur)
    return df



def preprocess_pipeline(
    file_path=r"..\data\en.openfoodfacts.org.products.csv.gz",
    sample_size=10000,
    chunksize=None,
    features=None,
    target_col="nutriscore_grade",
    missing_threshold=60,
    test_size=0.2,
    random_state=42,
):
    """
    Enchaîne toutes les étapes dans le bon ordre et retourne un dict
    contenant tout ce dont tu as besoin pour entraîner et prédire.

    Si `chunksize` est renseigné, le CSV entier est lu morceau par morceau
    (au lieu de s'arrêter à `sample_size` lignes) : utile pour un fichier
    trop gros pour tenir en mémoire d'un coup.

    Si `remove_outliers=True`, les lignes hors bornes IQR (calculées
    UNIQUEMENT sur X_train, après imputation) sont retirées de X_train,
    X_test et X_inconnu. `outlier_cols` permet de restreindre les colonnes
    examinées (toutes les features par défaut) ; `outlier_exclude` permet
    d'en exclure certaines (ex : `energy_100g`, souvent très étalée).
    """
    features = features or [
        "energy_100g", "saturated-fat_100g", "sugars_100g", "salt_100g",
        "fiber_100g", "proteins_100g", "fruits-vegetables-legumes_100g",
        "fat_100g", "carbohydrates_100g",
    ]

    # 1. Chargement + séparation connu/inconnu (aucun calcul appris ici)
    if chunksize is not None:
        df_connu, df_inconnu = load_and_split_known_unknown(
            file_path=file_path, target_col=target_col, chunksize=chunksize
        )
    else:
        df = load_data(file_path=file_path, sample_size=sample_size)
        df_connu, df_inconnu = split_known_unknown(df, target_col=target_col)

    # 2. Split AVANT tout calcul statistique
    X_train, X_test, y_train, y_test = split_train_test(
        df_connu, features, target_col=target_col,
        test_size=test_size, random_state=random_state,
    )
    X_inconnu = df_inconnu[features]

    # 3. Décision des colonnes à retirer, apprise sur X_train seul
    percent_missing = compute_missing_percentage(X_train)
    columns_to_drop = get_columns_to_drop(percent_missing, threshold=missing_threshold)

    X_train = apply_column_drop(X_train, columns_to_drop)
    X_test = apply_column_drop(X_test, columns_to_drop)
    X_inconnu = apply_column_drop(X_inconnu, columns_to_drop)

    # 4. Suppression des lignes totalement vides (train et inconnu, pas test
    #    ici pour garder un test propre, mais tu peux l'ajouter si tu veux)
    nb_train_avant_drop = len(X_train)
    nb_test_avant_drop = len(X_test)
    nb_inconnu_avant_drop = len(X_inconnu)

    
    X_train = drop_rows_all_missing(X_train)
    y_train = y_train.loc[X_train.index]  # on garde y_train aligné avec X_train
    X_test = drop_rows_all_missing(X_test)
    y_test = y_test.loc[X_test.index]  # on garde y_test aligné avec X_test
    X_inconnu = drop_rows_all_missing(X_inconnu)

    lignes_vides_supprimees = {
        "train": nb_train_avant_drop - len(X_train),
        "test": nb_test_avant_drop - len(X_test),
        "inconnu": nb_inconnu_avant_drop - len(X_inconnu),
    }

    # 4bis. Suppression des valeurs physiquement impossibles (ex : plus de
    #    100g d'un nutriment pour 100g de produit), AVANT l'imputation pour
    #    que ces valeurs aberrantes ne polluent pas la moyenne/le mode appris
    #    sur X_train. Bornes fixes (pas apprises sur les données), donc
    #    applicables telles quelles sur train/test/inconnu sans fuite.
    nb_train_avant_impossible = len(X_train)
    nb_test_avant_impossible = len(X_test)
    nb_inconnu_avant_impossible = len(X_inconnu)

    X_train, y_train = apply_impossible_drop(X_train, y_train)
    X_test, y_test = apply_impossible_drop(X_test, y_test)
    X_inconnu = apply_impossible_drop(X_inconnu)

    lignes_impossibles_supprimees = {
        "train": nb_train_avant_impossible - len(X_train),
        "test": nb_test_avant_impossible - len(X_test),
        "inconnu": nb_inconnu_avant_impossible - len(X_inconnu),
    }

    # 5. Imputation apprise sur X_train, appliquée partout
    valeurs_manquantes_avant_imputation = {
        "train": X_train.isna().sum().to_dict(),
        "test": X_test.isna().sum().to_dict(),
        "inconnu": X_inconnu.isna().sum().to_dict(),
    }

    valeurs_imputation = fit_imputation_values(X_train)
    X_train = apply_imputation(X_train, valeurs_imputation)
    X_test = apply_imputation(X_test, valeurs_imputation)
    X_inconnu = apply_imputation(X_inconnu, valeurs_imputation)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_inconnu": X_inconnu,
        "percent_missing": percent_missing,
        "columns_dropped": columns_to_drop,
        "valeurs_imputation": valeurs_imputation,
        "lignes_vides_supprimees": lignes_vides_supprimees,
        "valeurs_manquantes_avant_imputation": valeurs_manquantes_avant_imputation,
        "lignes_impossibles_supprimees": lignes_impossibles_supprimees,
    }