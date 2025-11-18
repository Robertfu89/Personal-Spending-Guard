"""
cashflow_optimizer.py

Core engine for:
- Training an MLP cashflow model
- Optimizing household spending (negative -> positive cashflow)
- Selecting demo families
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance


# --------------------------
# Configuration
# --------------------------

FLEX_CATEGORIES = [
    "utilities",
    "groceries",
    "transport",
    "entertainment",
    "dining_out",
    "clothing",
    "other_variable",
    "unexpected_spend",
]

FIXED_CATEGORIES = [
    "housing",
    "insurance",
    "childcare",
    "healthcare",
    "debt_min_payments",
]

TARGET_COL = "cashflow"
INCOME_COL = "total_income"


@dataclass
class ModelArtifacts:
    model: MLPRegressor
    feature_cols: List[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


# --------------------------
# Training
# --------------------------

def train_cashflow_model(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelArtifacts:
    """
    Train an MLPRegressor to predict cashflow from spending + income features.
    Assumes df[target_col] is the true cashflow.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=800,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    return ModelArtifacts(
        model=model,
        feature_cols=feature_cols,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


# --------------------------
# Single-Family Optimization
# --------------------------
def optimize_family(
    row: pd.Series,
    artifacts: ModelArtifacts,
    max_cut: float = 0.2,
    n_steps: int = 5,
) -> Dict:
    """
    Quality-of-life–aware optimization.

    - Primary goal: minimal reduction in flex categories such that
      predicted_cashflow_after > 0.
    - 5% of income is reported as a diagnostic flag, not enforced.
    - Cuts are uneven:
        * Heavy flex (dining_out, entertainment, other_variable, unexpected_spend)
          can be cut up to max_cut.
        * Soft flex (utilities, groceries, transport, clothing) only up to max_cut/2.
    """

    model = artifacts.model
    feature_cols = artifacts.feature_cols

    # Define which flex categories are cut heavily vs softly
    heavy_flex = ["dining_out", "entertainment", "other_variable", "unexpected_spend"]
    soft_flex  = ["utilities", "groceries", "transport", "clothing"]

    income = row[INCOME_COL]
    target_min_cf = 0.05 * income  # for reporting only

    base_features = row[feature_cols].copy()
    base_pred = float(model.predict(base_features.to_frame().T)[0])

    best_features = base_features.copy()
    best_pred = base_pred
    best_cut = 0.0

    # Candidate cut levels (0%, 5%, 10%, 15%, 20%)
    cut_levels = np.linspace(0.0, max_cut, n_steps)

    # 1) Try to find the *minimal* cut that makes cashflow positive
    for alpha in cut_levels:
        candidate = base_features.copy()

        # Soft flex: lighter cut
        for col in soft_flex:
            candidate[col] = base_features[col] * (1.0 - 0.5 * alpha)

        # Heavy flex: full cut
        for col in heavy_flex:
            candidate[col] = base_features[col] * (1.0 - alpha)

        pred_cf = float(model.predict(candidate.to_frame().T)[0])

        # If this is the first alpha that makes cashflow positive, we stop
        if pred_cf > 0:
            best_features = candidate
            best_pred = pred_cf
            best_cut = alpha
            break

        # Track best improvement even if still negative
        if pred_cf > best_pred:
            best_features = candidate
            best_pred = pred_cf
            best_cut = alpha

    # 2) Evaluate final status vs 5% income threshold (diagnostic only)
    meets_5pct = best_pred >= target_min_cf

    return {
        "income": income,
        "original_features": base_features,
        "optimized_features": best_features,
        "pred_cashflow_before": base_pred,
        "pred_cashflow_after": best_pred,
        "meets_5pct_constraint": meets_5pct,
        "cut_fraction": best_cut,
    }

# --------------------------
# Selecting Negative -> Positive Families
# --------------------------

def select_negative_to_positive_families(
    df: pd.DataFrame,
    artifacts: ModelArtifacts,
    n_families: int = 10,
) -> pd.DataFrame:
    """
    From the test set, pick families whose historical cashflow < 0
    and whose AI-recommended cashflow > 0.
    Randomly sample up to n_families.
    """
    model = artifacts.model
    X_test = artifacts.X_test
    y_test = artifacts.y_test

    # Build a DataFrame containing features + true cashflow
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test

    negative_df = test_df[test_df[TARGET_COL] < 0].copy()
    results = []

    for idx, row in negative_df.iterrows():
        opt = optimize_family(row, artifacts)
        if opt["pred_cashflow_after"] > 0:
            orig = opt["original_features"]
            new = opt["optimized_features"]

            results.append({
                "index": idx,
                INCOME_COL: opt["income"],
                "actual_cashflow": row[TARGET_COL],
                "pred_cashflow_before": opt["pred_cashflow_before"],
                "pred_cashflow_after": opt["pred_cashflow_after"],
                "cashflow_5pct_threshold": 0.05 * opt["income"],
                "meets_5pct_constraint": opt["meets_5pct_constraint"],
                "cut_fraction": opt["cut_fraction"],
                "entertainment_before": orig["entertainment"],
                "entertainment_after": new["entertainment"],
                "dining_before": orig["dining_out"],
                "dining_after": new["dining_out"],
                "clothing_before": orig["clothing"],
                "clothing_after": new["clothing"],
                "other_variable_before": orig["other_variable"],
                "other_variable_after": new["other_variable"],
                "unexpected_before": orig["unexpected_spend"],
                "unexpected_after": new["unexpected_spend"],
            })

    results_df = pd.DataFrame(results)

    if len(results_df) > n_families:
        results_df = results_df.sample(n_families, random_state=42).reset_index(drop=True)

    return results_df


# --------------------------
# Feature Importance
# --------------------------

def compute_feature_importance(
    artifacts: ModelArtifacts,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation-based feature importance (SHAP-style proxy).
    """
    model = artifacts.model
    X_test = artifacts.X_test
    y_test = artifacts.y_test
    feature_cols = artifacts.feature_cols

    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    df_importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_effect": np.abs(perm.importances_mean),
    }).sort_values("mean_abs_effect", ascending=False)

    return df_importance
