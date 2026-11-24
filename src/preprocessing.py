"""Leakage-controlled winsorization and imputation for FNCE 2370 Team 7.

Call fit_preprocessor only with the eligible, labeled estimation sample.
Use the returned parameter table unchanged on validation or test observations.
No scaling is applied; model-specific scaling belongs in the model pipeline.
"""
import numpy as np
import pandas as pd


def _finite_numeric(series):
    return pd.to_numeric(series, errors="raise").astype("float64").replace(
        [np.inf, -np.inf], np.nan
    )


def fit_preprocessor(train, chs_features, global_features=(),
                     lower=0.05, upper=0.95):
    """Estimate feature-specific bounds and fill values on training rows."""
    if not 0 <= lower < upper <= 1:
        raise ValueError("Invalid winsorization quantiles.")
    if train.empty:
        raise ValueError("Training sample is empty.")
    chs_features, global_features = list(chs_features), list(global_features)
    features = chs_features + global_features
    if not features or len(set(features)) != len(features):
        raise ValueError("Feature lists must be nonempty and unique.")
    records = []
    for feature in features:
        x = _finite_numeric(train[feature])
        if not x.notna().any():
            raise ValueError(f"No finite training values: {feature}")
        bounds = x.quantile([lower, upper])
        lo, hi = float(bounds.iloc[0]), float(bounds.iloc[1])
        clipped = x.clip(lo, hi)
        if feature == "sigma" and feature in chs_features:
            policy, fill = "mean", clipped.mean()
        elif feature in global_features:
            policy, fill = "median", clipped.median()
        else:
            policy, fill = "required", np.nan
            if x.isna().any():
                raise ValueError(f"Missing required training feature: {feature}")
        records.append({
            "feature": feature, "lower": lo, "upper": hi,
            "imputation": policy, "fill_value": fill,
            "training_nonmissing": int(x.notna().sum()),
            "training_rows": len(x),
        })
    return pd.DataFrame(records).set_index("feature")


def transform_features(frame, params):
    """Apply fixed parameters without learning anything from frame."""
    if not params.index.is_unique:
        raise ValueError("Duplicate feature names in parameters.")
    columns = {}
    for feature, rule in params.iterrows():
        x = _finite_numeric(frame[feature]).clip(rule["lower"], rule["upper"])
        if rule["imputation"] in ("mean", "median"):
            x = x.fillna(rule["fill_value"])
        elif rule["imputation"] != "required":
            raise ValueError(f"Unknown imputation rule: {feature}")
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError(f"Nonfinite values after transformation: {feature}")
        columns[feature] = x
    return pd.DataFrame(columns, index=frame.index)
