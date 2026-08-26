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


# Identifier et compter les valeurs impossibles
def identify_impossible(X):
    lignes_impossibles = pd.Series(False, index=X.index)
    comptes = {}

    for col in X.columns:
        # on choisit les bonnes bornes selon la colonne
        if col == COLONNE_ENERGIE:
            bas, haut = BORNE_ENERGIE
        else:
            bas, haut = BORNE_NUTRIMENT

        # hors_bornes = True là où la valeur est impossible
        hors_bornes = X[col].notna() & ~X[col].between(bas, haut)

        # compter combien de valeurs impossibles dans cette colonne
        comptes[col] = int(hors_bornes.sum())

        # la liste des lignes à supprimer
        lignes_impossibles = lignes_impossibles | hors_bornes

    return lignes_impossibles, comptes


# Supprimer les lignes avec des valeurs impossibles
def apply_impossible_drop(X, y=None):
    lignes_impossibles, _ = identify_impossible(X)

    # garder seulement les lignes qui ne sont PAS impossibles
    X_propre = X[~lignes_impossibles]

    # si pas de y : on renvoie juste X nettoyé
    if y is None:
        return X_propre

    # sinon on garde y aligné avec X (mêmes lignes)
    return X_propre, y.loc[X_propre.index]


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