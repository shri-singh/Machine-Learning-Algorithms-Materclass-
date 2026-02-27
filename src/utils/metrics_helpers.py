"""
Metrics helper functions for the ML Masterclass.

Provides convenient wrappers for computing and displaying model evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
)


def classification_report_df(y_true, y_pred, y_proba=None, target_names=None):
    """Generate a classification report as a pandas DataFrame.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_proba : array-like, optional
        Predicted probabilities for the positive class (binary only).
    target_names : list, optional
        Class names.

    Returns
    -------
    pd.DataFrame
        Classification report with optional ROC-AUC.
    """
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report).T

    # Add ROC-AUC if probabilities are provided (binary classification)
    if y_proba is not None:
        try:
            auc_val = roc_auc_score(y_true, y_proba)
            df.loc["roc_auc", :] = [auc_val, np.nan, np.nan, np.nan]
        except ValueError:
            pass

    return df.round(4)


def regression_metrics_summary(y_true, y_pred, model_name="Model"):
    """Compute a summary of regression metrics.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    model_name : str
        Name for the model (used as index).

    Returns
    -------
    pd.DataFrame
        Row with MAE, MSE, RMSE, R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return pd.DataFrame(
        {"MAE": [mae], "MSE": [mse], "RMSE": [rmse], "R2": [r2]},
        index=[model_name],
    ).round(4)


def compare_models(results_list):
    """Combine multiple regression_metrics_summary results into one table.

    Parameters
    ----------
    results_list : list of pd.DataFrame
        List of DataFrames from regression_metrics_summary.

    Returns
    -------
    pd.DataFrame
        Combined comparison table.
    """
    return pd.concat(results_list)
