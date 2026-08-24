"""
logreg.py

Fonctions pour :
- afficher une matrice de corrélation
- calculer le VIF (multicolinéarité)
- entraîner une régression logistique ordinale sur le nutriscore
- entraîner un LassoCV

Usage dans un notebook :
    from nutriscore_model import (
        plot_correlation_matrix,
        compute_vif,
        train_ordinal_logit,
        evaluate_ordinal_model,
        run_lassocv,
    )
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.miscmodels.ordinal_model import OrderedModel

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score, confusion_matrix


# ==========================
# Matrice de corrélation
# ==========================

def plot_correlation_matrix(df_restr, target_col="nutriscore_grade", method="pearson", figsize=(12, 10)):
    """
    Affiche la matrice de corrélation des variables numériques (target exclue).

    Parameters
    ----------
    df : DataFrame
        Données complètes (avec la colonne cible).
    target_col : str
        Nom de la colonne cible à exclure du calcul.
    method : str
        Méthode de corrélation ('pearson', 'spearman', 'kendall').
    figsize : tuple
        Taille de la figure.

    Returns
    -------
    corr : DataFrame
        La matrice de corrélation.
    """
    df_num = df_restr.drop(columns=[target_col]).copy()
    corr = df_num.corr(method=method)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                annot_kws={"size": 10}, ax=ax)
    plt.tight_layout()
    plt.show()

    return corr


# ==========================
# VIF (multicolinéarité)
# ==========================

def compute_vif(df_restr, target_col="nutriscore_grade", threshold=5.0):
    """
    Calcule le VIF (Variance Inflation Factor) de chaque variable numérique.
    Un VIF > 5-10 signale un problème de colinéarité.

    Parameters
    ----------
    df : DataFrame
        Données complètes (avec la colonne cible).
    target_col : str
        Colonne cible à exclure du calcul.
    threshold : float
        Seuil au-delà duquel une variable est jugée trop colinéaire.

    Returns
    -------
    vif_data : DataFrame
        VIF de toutes les variables.
    vif_ok : list
        Liste des variables dont le VIF est sous le seuil.
    """
    df_num = df_restr.drop(columns=[target_col]).copy()
    X = add_constant(df_num)

    vif_data = pd.DataFrame()
    vif_data['variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data = vif_data[vif_data['variable'] != 'const'].reset_index(drop=True)

    vif_ok = vif_data.loc[vif_data['VIF'] < threshold, 'variable'].tolist()

    print(vif_data)
    print(f"\nVariables gardées (VIF < {threshold}) : {vif_ok}")

    return vif_data, vif_ok


# ==========================
# Régression logistique ordinale
# ==========================

ORDRE_NUTRISCORE = ["a", "b", "c", "d", "e"]


def train_ordinal_logit(X_train, y_train, ordre=ORDRE_NUTRISCORE):
    """
    Standardise X_train et entraîne un modèle de régression logistique ordinale.

    Parameters
    ----------
    X_train : DataFrame
        Variables explicatives d'entraînement.
    y_train : Series
        Nutriscore d'entraînement (valeurs dans `ordre`).
    ordre : list
        Ordre hiérarchique des catégories cibles.

    Returns
    -------
    result : OrderedModelResults
        Résultat du modèle ajusté (result.summary() disponible).
    scaler : StandardScaler
        Le scaler ajusté sur X_train (à réutiliser pour X_test).
    """
    y_train = y_train.astype("category")
    y_train = y_train.cat.set_categories(ordre, ordered=True)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    model = OrderedModel(endog=y_train, exog=X_train_scaled, distr="logit")
    result = model.fit(method="bfgs")

    print(result.summary())

    return result, scaler

def evaluate_ordinal_model(result, scaler, X_test, y_test, ordre=ORDRE_NUTRISCORE):
    """
    Applique le scaler à X_test, prédit et évalue le modèle ordinal.

    Parameters
    ----------
    result : OrderedModelResults
        Modèle entraîné (sortie de train_ordinal_logit).
    scaler : StandardScaler
        Scaler entraîné sur X_train.
    X_test : DataFrame
        Variables explicatives de test.
    y_test : Series
        Nutriscore réel de test.
    ordre : list
        Ordre hiérarchique des catégories cibles.

    Returns
    -------
    pred : Series
        Prédictions (catégorie la plus probable).
    probas : DataFrame
        Probabilités prédites par catégorie.
    """
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    probas = result.predict(X_test_scaled)
    # Les colonnes de probas sont les codes 0..K-1 -> on les remappe vers les labels d'origine
    probas.columns = ordre
    pred = probas.idxmax(axis=1)

    print("Accuracy :", accuracy_score(y_test, pred))
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, pred, labels=ordre))

    return pred, probas

# ==========================
# LASSO (avec CV interne)
# ==========================

from sklearn.linear_model import LogisticRegressionCV

def run_lasso_logit(X, y, cv=5, max_iter=1000, scale=True, dropna=True, random_state=42, ordre=ORDRE_NUTRISCORE):
    """
    Entraîne une régression logistique multinomiale pénalisée L1 (Lasso logistique)
    avec sélection automatique de C par validation croisée. Utile pour la sélection
    de variables sur une cible catégorielle (ex : nutriscore).

    Parameters
    ----------
    X : DataFrame
        Variables explicatives.
    y : Series
        Variable cible catégorielle (ex : nutriscore_grade).
    cv : int
        Nombre de folds pour la validation croisée.
    max_iter : int
        Nombre maximal d'itérations.
    scale : bool
        Si True, standardise les variables explicatives avant l'entraînement.
    dropna : bool
        Si True, supprime les lignes avec des NaN (X et y ensemble).
    random_state : int
        Pour la reproductibilité.
    ordre : list
        Ordre des catégories cibles (pour l'affichage des coefficients par classe).

    Returns
    -------
    model : LogisticRegressionCV
        Le modèle entraîné.
    scaler : StandardScaler ou None
        Le scaler utilisé (None si scale=False).
    """
    if dropna:
        data = pd.concat([X, y], axis=1).dropna()
        X = data[X.columns]
        y = data[y.name] if hasattr(y, "name") and y.name in data.columns else data.iloc[:, -1]

    feature_names = X.columns if hasattr(X, 'columns') else None

    scaler = None
    if scale:
        scaler = StandardScaler()  # centre-réduit les données : moyenne 0, écart-type 1
        X = scaler.fit_transform(X)  # calcule moyenne/écart-type puis transforme

    model = LogisticRegressionCV(
        cv=cv,
        max_iter=max_iter,
        random_state=random_state,
        penalty="l1",
        solver="saga",
        Cs=10,  # nombre de valeurs de régularisation testées le long de la grille
    )
    model.fit(X, y)

    print(f"Lignes utilisées : {len(y)}")
    print(f"Accuracy (train) : {model.score(X, y):.3f}")

    if feature_names is not None:
        classes = list(model.classes_)
        # on réordonne les classes selon `ordre` si elles correspondent, sinon on garde l'ordre du modèle
        classes_affichage = [c for c in ordre if c in classes] or classes

        print("\nCoefficients par classe (Lasso logistique) :")
        for cls in classes_affichage:
            idx = list(classes).index(cls)
            coefs_cls = model.coef_[idx]
            print(f"\n  Classe '{cls}' :")
            for name, coef in zip(feature_names, coefs_cls):
                print(f"    {name}: {coef:.4f}")

        # variables dont le coefficient est nul sur TOUTES les classes -> jugées non informatives par Lasso
        coefs_nuls_partout = [
            name for i, name in enumerate(feature_names)
            if all(model.coef_[c][i] == 0 for c in range(len(classes)))
        ]
        print(f"\nVariables avec coefficient nul sur toutes les classes : {coefs_nuls_partout}")

    return model, scaler