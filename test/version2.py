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

def plot_correlation_matrix(df_restr, target_col="nutriscore_grade",cols_categorielles=None, method="pearson", figsize=(12, 10)):
    """
    Affiche la matrice de corrélation des variables numériques (target exclue).

    Parameters
    ----------
    df : DataFrame
        Données complètes (avec la colonne cible).
    target_col : str
        Nom de la colonne cible à exclure du calcul.
    cols_categorielles : list, optional
        Liste des colonnes catégorielles à exclure du calcul.
    method : str
        Méthode de corrélation ('pearson', 'spearman', 'kendall').
    figsize : tuple
        Taille de la figure.

    Returns
    -------
    corr : DataFrame
        La matrice de corrélation.
    """
    cols_categorielles = cols_categorielles or []
    df_num = df_restr.drop(columns=[target_col] + cols_categorielles, errors="ignore").copy()
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

def compute_vif(df_restr, target_col="nutriscore_grade", cols_categorielles=None, threshold=5.0):
    cols_categorielles = cols_categorielles or []
    df_num = (
        df_restr
        .drop(columns=[target_col] + cols_categorielles, errors="ignore")
        .select_dtypes(include="number")
        .copy()
    )
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
# Encodage des variables catégorielles
# ==========================

from sklearn.preprocessing import OneHotEncoder

def encode_categorical(X_train, X_test, cols_categorielles, cols_numeriques, drop='first'):
    """
    Encode les variables catégorielles en One-Hot (fit sur train, transform sur test),
    puis recolle les colonnes numériques avec les colonnes encodées.

    Parameters
    ----------
    X_train : DataFrame
        Données d'entraînement (avant encodage).
    X_test : DataFrame
        Données de test (avant encodage).
    cols_categorielles : list
        Colonnes catégorielles à encoder.
    cols_numeriques : list
        Colonnes numériques à conserver telles quelles.
    drop : str or None
        Stratégie de suppression de modalité pour éviter le piège de la variable
        dummy (colinéarité avec la constante). 'first' recommandé pour VIF/logit.

    Returns
    -------
    X_train_final : DataFrame
        Données d'entraînement encodées (numériques + catégorielles encodées).
    X_test_final : DataFrame
        Données de test encodées (mêmes colonnes que X_train_final).
    encoder : OneHotEncoder
        L'encodeur ajusté sur X_train (à réutiliser si besoin).
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=drop)

    # fit + transform sur train
    encoded_train = encoder.fit_transform(X_train[cols_categorielles])
    encoded_train_df = pd.DataFrame(
        encoded_train,
        columns=encoder.get_feature_names_out(cols_categorielles),
        index=X_train.index
    )

    # transform uniquement sur test (jamais de fit sur test)
    encoded_test = encoder.transform(X_test[cols_categorielles])
    encoded_test_df = pd.DataFrame(
        encoded_test,
        columns=encoder.get_feature_names_out(cols_categorielles),
        index=X_test.index
    )

    # on recolle les colonnes numériques avec les colonnes encodées
    X_train_final = pd.concat([X_train[cols_numeriques], encoded_train_df], axis=1)
    X_test_final = pd.concat([X_test[cols_numeriques], encoded_test_df], axis=1)

    return X_train_final, X_test_final, encoder
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

import shap
import numpy as np

def explain_shap_brands(result, scaler, X_train, X_test, cols_categorielles_encoded, ordre=ORDRE_NUTRISCORE, nsamples=100):
    """
    Calcule les valeurs SHAP et classe les modalités de brands
    par importance moyenne (toutes classes confondues).

    Parameters
    ----------
    cols_categorielles_encoded : list
        Noms des colonnes encodées (ex: toutes celles commençant par 'brands_grouped_').
    """
    def predict_fn(X_array):
        X_df = pd.DataFrame(X_array, columns=X_train.columns)
        X_scaled = pd.DataFrame(scaler.transform(X_df), columns=X_df.columns)
        probas = result.predict(X_scaled)
        probas.columns = ordre
        return probas.values

    background = shap.sample(X_train, nsamples, random_state=42)
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_test)  # shape (n_obs, n_features, n_classes)

    # importance moyenne absolue, toutes classes confondues
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))  # moyenne sur obs et classes
    importance = pd.Series(mean_abs_shap, index=X_train.columns)

    # on filtre uniquement les colonnes de brands
    importance_brands = importance[cols_categorielles_encoded].sort_values(ascending=False)

    print("Modalités de brands les plus importantes (SHAP moyen absolu) :")
    print(importance_brands)

    return shap_values, importance_brands