from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(df[column])]
    categorical = [column for column in features if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(transformers)


def top_coefficients(model: Pipeline, limit: int = 10) -> list[dict[str, float | str]]:
    preprocess = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    names = preprocess.get_feature_names_out().tolist()
    coefficients = np.asarray(estimator.coef_)
    if coefficients.ndim > 1:
        coefficients = np.mean(np.abs(coefficients), axis=0)
    else:
        coefficients = np.abs(coefficients)
    ranked = sorted(zip(names, coefficients), key=lambda item: float(item[1]), reverse=True)[:limit]
    return [{"feature": name, "absolute_coefficient": float(value)} for name, value in ranked]
