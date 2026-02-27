"""
Reusable plotting helpers for the ML Masterclass.

All functions return matplotlib figure/axes objects for further customization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix",
                          cmap="Blues", figsize=(6, 5), ax=None):
    """Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        Class label names.
    title : str
        Plot title.
    cmap : str
        Colormap name.
    figsize : tuple
        Figure size (used only if ax is None).
    ax : matplotlib Axes, optional
        Axes to plot on.

    Returns
    -------
    fig, ax
    """
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046)

    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.tight_layout()
    return fig, ax


def plot_roc_curve(y_true, y_score, title="ROC Curve", figsize=(7, 5), ax=None):
    """Plot ROC curve with AUC score.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted probabilities for the positive class.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, ax


def plot_precision_recall_curve(y_true, y_score, title="Precision-Recall Curve",
                                figsize=(7, 5), ax=None):
    """Plot Precision-Recall curve with average precision.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted probabilities for the positive class.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(recall, precision, color="steelblue", lw=2,
            label=f"PR curve (AP = {ap:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig, ax


def plot_residuals(y_true, y_pred, title="Residual Plot", figsize=(8, 5), ax=None):
    """Plot residuals vs predicted values for regression diagnostics.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    residuals = np.array(y_true) - np.array(y_pred)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="k", linewidths=0.5)
    ax.axhline(y=0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_feature_importance(feature_names, importances, title="Feature Importance",
                            top_n=None, figsize=(8, 5), ax=None):
    """Plot horizontal bar chart of feature importances.

    Parameters
    ----------
    feature_names : list
        Names of features.
    importances : array-like
        Importance values (e.g., from model.feature_importances_).
    title : str
        Plot title.
    top_n : int, optional
        Show only top N features. If None, show all.
    figsize : tuple
        Figure size.
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    # Sort by importance
    indices = np.argsort(importances)
    if top_n is not None:
        indices = indices[-top_n:]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.barh(range(len(indices)), importances[indices], align="center", color="steelblue")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_learning_curve(train_sizes, train_scores, val_scores,
                        title="Learning Curve", figsize=(8, 5), ax=None):
    """Plot learning curve showing train and validation scores vs training size.

    Parameters
    ----------
    train_sizes : array-like
        Number of training samples at each step.
    train_scores : array-like, shape (n_steps, n_folds)
        Training scores.
    val_scores : array-like, shape (n_steps, n_folds)
        Validation scores.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="orange")
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training score")
    ax.plot(train_sizes, val_mean, "o-", color="orange", label="Validation score")

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
