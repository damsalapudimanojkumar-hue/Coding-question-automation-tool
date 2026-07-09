"""
dataset_ops.py

Deterministic dataset edits — plain pandas, NO LLM. These are the "preset"
edits from the Edit-dataset panel (resize, rebalance, add noise, inject nulls,
drop columns). Doing them in code (instead of asking the model) makes them
100% reliable and lets us enforce the file-touch rules that keep the
assignment auto-gradable:

    - ground_truth.csv is NEVER touched (it defines grading).
    - train and test feature columns stay in sync (a dropped column is
      removed from BOTH; ground truth is targets, so it is left alone).
    - size / noise / null edits apply to TRAIN only, so the test set and
      ground truth remain a stable grading target.

The target column is detected structurally (it is the column present in train
but absent from test), so we never guess a name.
"""

import numpy as np
import pandas as pd

from naming import asset_paths


def _feature_and_target(train_df, test_df):
    feature_cols = list(test_df.columns)
    target_cols = [c for c in train_df.columns if c not in test_df.columns]
    target = target_cols[0] if target_cols else None
    return feature_cols, target


def feature_columns(workspace, code):
    """Feature column names (from the test CSV) — used to populate the UI's
    drop-columns picker."""
    test = pd.read_csv(asset_paths(workspace, code)["dataset_test"])
    return list(test.columns)


def apply_edits(workspace, code, ops, seed=42):
    """
    Apply the requested preset edits in-place to the workspace CSVs.

    ops keys (all optional):
        drop_columns : list[str]   feature columns to remove (train + test)
        rebalance    : bool        down-sample majority classes to the minority count
        resize_factor: float       train-row multiplier (0.5 = halve, 1.5 = bootstrap up)
        add_noise    : float       Gaussian noise intensity as a fraction of each col's std
        inject_nulls : float       fraction of train feature cells to set NaN

    Returns a summary dict describing exactly what changed.
    """
    paths = asset_paths(workspace, code)
    train = pd.read_csv(paths["dataset_train"])
    test = pd.read_csv(paths["dataset_test"])
    feature_cols, target = _feature_and_target(train, test)
    rng = np.random.RandomState(seed)

    orig_train_shape = train.shape
    changes = []

    # 1) drop feature columns from BOTH train and test (never the target)
    drop = [c for c in ops.get("drop_columns", []) if c in feature_cols and c != target]
    if drop:
        train = train.drop(columns=drop)
        test = test.drop(columns=drop)
        feature_cols = [c for c in feature_cols if c not in drop]
        changes.append(f"dropped columns {drop} from train and test")

    # 2) rebalance classes (train only): down-sample every class to the minority count.
    # Iterate groups + concat (keeps the target column; avoids groupby.apply index pitfalls).
    if ops.get("rebalance") and target and target in train.columns:
        per_class = int(train[target].value_counts().min())
        parts = [g.sample(n=per_class, random_state=seed) for _, g in train.groupby(target)]
        train = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
        changes.append(f"rebalanced to {per_class} rows per class")

    # 3) resize train rows: <1 subsample, >1 bootstrap (with replacement -> duplicates)
    factor = float(ops.get("resize_factor", 1.0) or 1.0)
    if factor > 0 and abs(factor - 1.0) > 1e-9:
        replace = factor > 1.0
        train = train.sample(frac=factor, replace=replace, random_state=seed).reset_index(drop=True)
        how = "bootstrapped up" if replace else "subsampled"
        changes.append(f"{how} train to {len(train)} rows (x{factor:g})")

    # 4) add Gaussian noise to numeric TRAIN features only
    noise = float(ops.get("add_noise", 0.0) or 0.0)
    if noise > 0:
        num_cols = [c for c in feature_cols
                    if c in train.columns and pd.api.types.is_numeric_dtype(train[c])]
        for c in num_cols:
            std = train[c].std()
            if std and not np.isnan(std):
                train[c] = train[c] + rng.normal(0.0, std * noise, size=len(train))
        changes.append(f"added Gaussian noise (intensity {noise:g}) to {len(num_cols)} numeric features")

    # 5) inject nulls into TRAIN features only (never the target)
    nulls = float(ops.get("inject_nulls", 0.0) or 0.0)
    if nulls > 0:
        for c in [c for c in feature_cols if c in train.columns]:
            mask = rng.rand(len(train)) < nulls
            train.loc[mask, c] = np.nan
        changes.append(f"injected ~{int(round(nulls * 100))}% nulls into train features")

    # persist (ground_truth.csv intentionally left untouched)
    train.to_csv(paths["dataset_train"], index=False)
    test.to_csv(paths["dataset_test"], index=False)

    return {
        "orig_train_shape": orig_train_shape,
        "new_train_shape": train.shape,
        "test_shape": test.shape,
        "target": target,
        "features": feature_cols,
        "changes": changes,
    }
