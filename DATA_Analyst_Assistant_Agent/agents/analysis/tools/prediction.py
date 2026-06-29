from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.frame import frame_from_records
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.modeling import build_preprocessor, top_coefficients

@tool
def fit_regression_model(records: list[dict[str, Any]], target: str, features: list[str]) -> dict[str, Any]:
    """Fit a deterministic linear regression baseline and evaluate it on a holdout set."""

    df = frame_from_records(records)
    selected = [column for column in features if column in df.columns and column != target]
    if target not in df.columns or not selected:
        raise ValueError("A numeric target and at least one valid feature are required.")
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    x = df.loc[valid, selected]
    y = y.loc[valid]
    if len(y) < 20 or y.nunique() < 2:
        raise ValueError("Regression requires at least twenty rows and a non-constant target.")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model = Pipeline([("preprocess", build_preprocessor(x, selected)), ("model", LinearRegression())])
    model.fit(x_train, y_train)
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)
    return {
        "method": "linear_regression",
        "target": target,
        "features": selected,
        "observation_count": len(y),
        "train_count": len(y_train),
        "test_count": len(y_test),
        "train_r2": float(r2_score(y_train, train_prediction)),
        "test_r2": float(r2_score(y_test, test_prediction)),
        "test_mae": float(mean_absolute_error(y_test, test_prediction)),
        "test_rmse": float(mean_squared_error(y_test, test_prediction) ** 0.5),
        "top_coefficients": top_coefficients(model),
        "evaluation_scope": "fixed_holdout",
    }

@tool
def fit_classification_model(records: list[dict[str, Any]], target: str, features: list[str]) -> dict[str, Any]:
    """Fit a deterministic logistic baseline and evaluate it on a stratified holdout set."""

    df = frame_from_records(records)
    selected = [column for column in features if column in df.columns and column != target]
    if target not in df.columns or not selected:
        raise ValueError("A target and at least one valid feature are required.")
    valid = df[target].notna()
    x = df.loc[valid, selected]
    y = df.loc[valid, target].astype(str)
    if len(y) < 20 or y.nunique() < 2 or int(y.value_counts().min()) < 4:
        raise ValueError("Classification requires twenty rows and at least four observations per class.")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("preprocess", build_preprocessor(x, selected)),
        ("model", LogisticRegression(max_iter=1000, random_state=0)),
    ])
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    result: dict[str, Any] = {
        "method": "logistic_regression",
        "target": target,
        "features": selected,
        "classes": sorted(y.unique().tolist()),
        "observation_count": len(y),
        "train_count": len(y_train),
        "test_count": len(y_test),
        "test_accuracy": float(accuracy_score(y_test, prediction)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "test_f1_weighted": float(f1_score(y_test, prediction, average="weighted")),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=model.named_steps["model"].classes_).tolist(),
        "top_coefficients": top_coefficients(model),
        "evaluation_scope": "stratified_fixed_holdout",
    }
    if y.nunique() == 2:
        probability = model.predict_proba(x_test)[:, 1]
        positive = model.named_steps["model"].classes_[1]
        result["test_roc_auc"] = float(roc_auc_score((y_test == positive).astype(int), probability))
    return result
