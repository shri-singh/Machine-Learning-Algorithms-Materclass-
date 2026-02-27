"""
Preprocessing helper functions for the ML Masterclass.

Provides factory functions for building sklearn pipelines and column transformers.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessing_pipeline(numeric_features, categorical_features=None,
                                  numeric_strategy="median", scaler=None):
    """Build a ColumnTransformer preprocessing pipeline.

    Parameters
    ----------
    numeric_features : list
        List of numeric column names.
    categorical_features : list, optional
        List of categorical column names.
    numeric_strategy : str
        Imputation strategy for numeric features ('mean', 'median', 'most_frequent').
    scaler : sklearn scaler, optional
        Scaler to use for numeric features. Defaults to StandardScaler.

    Returns
    -------
    ColumnTransformer
        Fitted column transformer ready to use in a Pipeline.
    """
    if scaler is None:
        scaler = StandardScaler()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=numeric_strategy)),
        ("scaler", scaler),
    ])

    transformers = [("num", numeric_pipeline, numeric_features)]

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers)


def get_feature_names_from_pipeline(column_transformer, numeric_features,
                                     categorical_features=None):
    """Extract feature names from a fitted ColumnTransformer.

    Parameters
    ----------
    column_transformer : ColumnTransformer
        A fitted ColumnTransformer.
    numeric_features : list
        Original numeric feature names.
    categorical_features : list, optional
        Original categorical feature names.

    Returns
    -------
    list
        Combined list of feature names after transformation.
    """
    try:
        return list(column_transformer.get_feature_names_out())
    except AttributeError:
        # Fallback for older sklearn
        feature_names = list(numeric_features)
        if categorical_features:
            encoder = column_transformer.named_transformers_["cat"].named_steps["encoder"]
            cat_names = list(encoder.get_feature_names_out(categorical_features))
            feature_names.extend(cat_names)
        return feature_names
