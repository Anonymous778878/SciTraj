"""
Tier 24 — LightGBM on the expanded pair features.

Loads the saved feature matrices from T22 and trains a LightGBM
classifier. LightGBM finds non-linear interactions between features
that an MLP can miss; on tabular pair-prediction tasks it often
yields +0.5 to +2 AUC over the MLP for free.

Output: outputs/metrics/tier24_lightgbm.json
        models/tier24_lightgbm/model.txt
        models/tier24_lightgbm/feature_importance.json
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from utils import ensure_dir, get_logger, load_config

log = get_logger("tier24_lightgbm")


def main():
    cfg = load_config()
    metrics_dir = ensure_dir(cfg["paths"]["metrics_dir"])
    models_root = Path(cfg["paths"]["models_dir"])
    models_dir = ensure_dir(models_root / "tier24_lightgbm")

    # Load features from T22
    feat_path = models_root / "tier22_pair_mlp_expanded" / "features.npz"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Expected feature file at {feat_path}. Run T22 first."
        )
    log.info("loading features from %s", feat_path)
    z = np.load(feat_path, allow_pickle=False)

    X_train = np.vstack([z["X_train_pos"], z["X_train_neg"]])
    y_train = np.concatenate([
        np.ones(z["X_train_pos"].shape[0]),
        np.zeros(z["X_train_neg"].shape[0]),
    ])
    X_val = np.vstack([z["X_val_pos"], z["X_val_neg"]])
    y_val = np.concatenate([
        np.ones(z["X_val_pos"].shape[0]),
        np.zeros(z["X_val_neg"].shape[0]),
    ])
    X_test = np.vstack([z["X_test_pos"], z["X_test_neg"]])
    y_test = np.concatenate([
        np.ones(z["X_test_pos"].shape[0]),
        np.zeros(z["X_test_neg"].shape[0]),
    ])

    log.info("train %s val %s test %s", X_train.shape, X_val.shape, X_test.shape)

    try:
        import lightgbm as lgb
    except ImportError:
        log.error("LightGBM not installed. Install with: pip install --user lightgbm")
        raise

    # LightGBM with mild regularisation
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        reg_alpha=0.0,
        reg_lambda=0.1,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        objective="binary",
        verbose=-1,
        n_jobs=-1,
    )

    log.info("training LightGBM with early stopping")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)],
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    val_auc = float(roc_auc_score(y_val, val_proba))

    test_proba = model.predict_proba(X_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, test_proba))
    test_ap = float(average_precision_score(y_test, test_proba))

    log.info("=" * 50)
    log.info("TIER 24 TEST: AUC=%.4f AP=%.4f  val=%.4f", test_auc, test_ap, val_auc)
    log.info("=" * 50)

    # Save
    model.booster_.save_model(str(models_dir / "model.txt"))
    importances = dict(zip(
        [f"f{i}" for i in range(X_train.shape[1])],
        model.feature_importances_.tolist(),
    ))
    with open(models_dir / "feature_importance.json", "w") as f:
        json.dump(importances, f, indent=2)

    # Save logits for ensembling (use raw scores)
    test_pos_logits = test_proba[: z["X_test_pos"].shape[0]]
    test_neg_logits = test_proba[z["X_test_pos"].shape[0]:]
    np.savez(
        models_dir / "test_logits.npz",
        pos_logits=test_pos_logits, neg_logits=test_neg_logits,
    )

    metrics = {
        "model": "tier24_lightgbm",
        "architecture": "LightGBM on T22 expanded pair features",
        "n_features": int(X_train.shape[1]),
        "best_val_auc": round(val_auc, 4),
        "link_prediction_auc_hard": round(test_auc, 4),
        "link_prediction_ap_hard": round(test_ap, 4),
        "n_estimators_used": int(model.best_iteration_) if hasattr(model, "best_iteration_") else None,
    }
    with open(metrics_dir / "tier24_lightgbm.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("wrote: %s", metrics_dir / "tier24_lightgbm.json")


if __name__ == "__main__":
    main()
