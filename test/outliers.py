"Ici on va avoir des fonctions pour détecter les outliers dans un dataset. "
"Pour cela je vais utiliser le méthode de l'écart interquartile (IQR) pour identifier les valeurs aberrantes. "
"Ainsi je vais identifier les valeurs impossibles dans le dataset et les marquer comme outliers. "
"Enfin, on va  enlever ces outliers du dataset pour obtenir un dataset plus propre et plus fiable pour l'analyse. "
"Cette fonction doit être appelée au dernier moment après preprocess.py et avant de tous es modèles appliqués."

import pandas as pd

# Bornes de ce qui est physiquement possible dans un aliment (pour 100 g)
BORNE_NUTRIMENT = (0, 100)      # 100 g pour un nutriscore
BORNE_ENERGIE = (0, 3800)       # 3800 kJ max
COLONNE_ENERGIE = "energy_100g" # le nom de la colonne d'énergie

# Bornes spécifiques pour des colonnes qui ne sont pas des g/100g
# (ex : nova_group va de 1 à 4, additives_n est un simple compte)
BORNES_SPECIFIQUES = {
    "nova_group": (1, 4),
    "additives_n": (0, 200),  # large marge de sécurité, c'est juste un compte
}

# Identifier et compter les valeurs impossibles
def identify_impossible(X):
    lignes_impossibles = pd.Series(False, index=X.index)
    comptes = {}

    for col in X.columns:
        # on ignore les colonnes non-numériques (ex : pnns_groups_1) :
        # les bornes physiques ne s'appliquent qu'à des valeurs numériques
        if not pd.api.types.is_numeric_dtype(X[col]):
            continue
        # on choisit les bonnes bornes selon la colonne
        if col in BORNES_SPECIFIQUES:
            bas, haut = BORNES_SPECIFIQUES[col]
        elif col == COLONNE_ENERGIE:
            bas, haut = BORNE_ENERGIE
        elif col.endswith("_100g"):
            bas, haut = BORNE_NUTRIMENT
        else:
            # colonne numérique sans borne physique connue -> on ne filtre pas
            continue

        # hors_bornes = True là où la valeur est impossible
        hors_bornes = X[col].notna() & ~X[col].between(bas, haut)

        # compter combien de valeurs impossibles dans cette colonne
        comptes[col] = int(hors_bornes.sum())

        # la liste des lignes à supprimer
        lignes_impossibles = lignes_impossibles | hors_bornes

    return lignes_impossibles, comptes


import numpy as np

# Remplacer les valeurs impossibles par NaN (au lieu de supprimer la ligne)
def apply_impossible_nan(X, y=None):
    X = X.copy()   # on travaille sur une copie pour ne pas modifier l'original

    # on regarde chaque colonne
    for col in X.columns:
        # on choisit les bonnes bornes selon la colonne
        if not pd.api.types.is_numeric_dtype(X[col]):
            continue

        if col in BORNES_SPECIFIQUES:
            bas, haut = BORNES_SPECIFIQUES[col]
        elif col == COLONNE_ENERGIE:
            bas, haut = BORNE_ENERGIE
        elif col.endswith("_100g"):
            bas, haut = BORNE_NUTRIMENT
        else:
            # colonne numérique sans borne physique connue -> on ne touche pas
            continue

        # les valeurs impossibles (hors bornes) deviennent NaN
        # notna() évite de toucher aux NaN déjà présents
        impossible = X[col].notna() & ~X[col].between(bas, haut)
        X.loc[impossible, col] = np.nan

    # si pas de y : on renvoie juste X
    if y is None:
        return X

    # sinon on renvoie X et y (y n'a pas changé, aucune ligne supprimée)
    return X, y


# Outliers à voir et visualiser (Tukey)
# Un outlier n'est PAS forcément à supprimer 
def tukey_bounds(colonne):
    # bornes de Tukey : hors de [Q1 - 1.5*IQR ; Q3 + 1.5*IQR] = outlier
    Q1 = colonne.quantile(0.25)
    Q3 = colonne.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR


def inspect_outliers(X, cols=None, infos=None, verbose=True):
    # si on ne précise pas les colonnes, on prend toutes
    if cols is None:
        cols = list(X.columns)

    resume = {}
    # on regarde chaque colonne
    for col in cols:
        bas, haut = tukey_bounds(X[col])

        # les valeurs en dehors des bornes de Tukey
        outliers = (X[col] < bas) | (X[col] > haut)
        resume[col] = int(outliers.sum())

        # on affiche seulement si verbose=True
        if verbose:
            print(f"\n{col} : {outliers.sum()} outliers selon Tukey")
            print(f"Bornes : [{bas:.2f} ; {haut:.2f}]")

            # si on a fourni des infos produits, on montre les 10 plus extrêmes
            if infos is not None and outliers.sum() > 0:
                apercu = (X.loc[outliers, [col]]
                          .join(infos)
                          .sort_values(col, ascending=False)
                          .head(10))
                print(apercu.to_string())

    return resume