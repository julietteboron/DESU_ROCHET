import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import log_loss


def train_logistic_multinomial(X_train, y_train, C=0.1, class_weight="balanced",max_iter=1000):
    """
    Entraîne une régression logistique multinomiale.
    """
    logistic_regression_multinomial = LogisticRegression(
        C=C,
        max_iter=max_iter
    )

    logistic_regression_multinomial.fit(X_train, y_train)

    return logistic_regression_multinomial


def train_logistic_ovr(X_train, y_train, C=0.1, max_iter=1000):
    """
    Entraîne une régression logistique One-vs-Rest.
    """
    logistic_regression_ovr = OneVsRestClassifier(
        LogisticRegression(
            C=C,
            max_iter=max_iter
        )
    )

    logistic_regression_ovr.fit(X_train, y_train)

    return logistic_regression_ovr


def evaluate_model(model, X_test, y_test):
    """
    Calcule l'accuracy du modèle sur les données de test.
    """
    accuracy = model.score(X_test, y_test)

    return accuracy


def grid_search_multinomial(X_train, y_train):
    """
    Recherche les meilleurs paramètres pour
    une régression logistique multinomiale.
    """

    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "max_iter": [500, 1000, 2000]
    }

    grid_multinomial = GridSearchCV(
        LogisticRegression(),
        param_grid,
        cv=5,
        scoring="accuracy"
    )

    grid_multinomial.fit(X_train, y_train)

    return grid_multinomial


def grid_search_ovr(X_train, y_train):
    """
    Recherche les meilleurs paramètres pour
    une régression logistique One-vs-Rest.
    """

    param_grid = {
        "estimator__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "estimator__max_iter": [500, 1000, 2000]
    }

    grid_ovr = GridSearchCV(
        OneVsRestClassifier(
            LogisticRegression()
        ),
        param_grid,
        cv=5,
        scoring="accuracy"
    )

    grid_ovr.fit(X_train, y_train)

    return grid_ovr


def calculate_log_loss(model, X, y):
    """
    Calcule la log-loss du modèle sur les données fournies.
    """
    probabilities = model.predict_proba(X)

    loss = log_loss(y, probabilities)

    return loss


def plot_loss_train_test(X_train, y_train, X_test, y_test):

    C_values = [0.001, 0.01, 0.1, 1, 10, 100]

    multinomial_train_losses = []
    multinomial_test_losses = []

    ovr_train_losses = []
    ovr_test_losses = []

    for C in C_values:

        # Multinomial
        logistic_regression_multinomial = LogisticRegression(
            C=C,
            max_iter=1000
        )

        logistic_regression_multinomial.fit(X_train, y_train)

        multinomial_train_losses.append(
            log_loss(
                y_train,
                logistic_regression_multinomial.predict_proba(X_train)
            )
        )

        multinomial_test_losses.append(
            log_loss(
                y_test,
                logistic_regression_multinomial.predict_proba(X_test)
            )
        )

        # One-vs-Rest
        logistic_regression_ovr = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                max_iter=1000
            )
        )

        logistic_regression_ovr.fit(X_train, y_train)

        ovr_train_losses.append(
            log_loss(
                y_train,
                logistic_regression_ovr.predict_proba(X_train)
            )
        )

        ovr_test_losses.append(
            log_loss(
                y_test,
                logistic_regression_ovr.predict_proba(X_test)
            )
        )

    # Visualisation
    plt.figure(figsize=(9, 6))

    plt.plot(C_values, multinomial_train_losses, marker="o", label="Multinomial - Train")
    plt.plot(C_values, multinomial_test_losses, marker="o", label="Multinomial - Test")
    plt.plot(C_values, ovr_train_losses, marker="o", label="OvR - Train")
    plt.plot(C_values, ovr_test_losses, marker="o", label="OvR - Test")

    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("Log-loss")
    plt.title("Évolution de la Log-loss en fonction de C")
    plt.legend()
    plt.grid()
    plt.show()


def grid_search_multinomial_v2(X_train, y_train):
    """
    Recherche les meilleurs paramètres (C, penalty, solver, class_weight, max_iter)
    pour une régression logistique multinomiale.
    """
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["saga"],  # saga supporte l1 et l2
        "class_weight": [None, "balanced"],
        "max_iter": [1000, 2000],
    }

    grid_multinomial = GridSearchCV(
        LogisticRegression(),
        param_grid,
        cv=5,
        scoring="accuracy"  # ou "f1_macro" si tes classes sont déséquilibrées
    )

    grid_multinomial.fit(X_train, y_train)

    return grid_multinomial


def plot_learning_curve_sklearn(model, X, y, cv=5, scoring="accuracy", n_points=8, random_state=42):
    """
    Trace la learning curve d'un modèle scikit-learn (compatible avec cross-validation).

    Parameters
    ----------
    model : estimator scikit-learn
        Modèle non entraîné (ex: LogisticRegression(C=0.1, ...)).
    X : DataFrame ou array
        Variables explicatives.
    y : Series ou array
        Variable cible.
    cv : int
        Nombre de folds pour la validation croisée.
    scoring : str
        Métrique utilisée ('accuracy', 'f1_macro', etc.).
    n_points : int
        Nombre de tailles d'échantillon testées.
    random_state : int
        Pour la reproductibilité.

    Returns
    -------
    train_sizes, train_scores, test_scores : arrays
        Résultats bruts de sklearn.model_selection.learning_curve.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X, y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, n_points),
        random_state=random_state,
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, marker='o', label='Train accuracy')
    plt.plot(train_sizes, test_mean, marker='o', label='Test accuracy')
    plt.xlabel("Nombre d'échantillons d'entraînement")
    plt.ylabel(scoring)
    plt.title("Learning curve — Régression logistique multinomiale")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return train_sizes, train_scores, test_scores