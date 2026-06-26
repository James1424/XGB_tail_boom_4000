from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .model_config import (
    ABLATION_GROUPS,
    FEATURE_GROUP_WEIGHTS,
    FEATURE_WEIGHT_PROFILES,
    HYPERPARAMETER_ABLATION_PROFILES,
    MAIN_WEIGHT_PROFILE,
    FEATURE_LIST_FILE,
    MAIN_MODEL_FILE,
    MAIN_SEEDS,
    MODEL_DIR,
    MODEL_PARAMS,
    OUTPUT_DIR,
    OUTPUT_FILES,
    TARGET_LABEL,
    TEST_START,
    TRAIN_END,
    TRAINING_CURVE_ROUNDS,
    VALID_END,
    VALID_START,
)
from .model_features import columns_matching, infer_feature_columns, load_panel
from .model_metrics import dataset_metrics, monthly_top_table, recent_top3_backtest_months, strategy_return_row, topk_metrics

SCORE_COL = "xgb_boom_probability"


def split_panel(df: pd.DataFrame):
    train = df[(df["month"] <= pd.Timestamp(TRAIN_END)) & df[TARGET_LABEL].notna()].copy()
    valid = df[(df["month"] >= pd.Timestamp(VALID_START)) & (df["month"] <= pd.Timestamp(VALID_END)) & df[TARGET_LABEL].notna()].copy()
    test = df[(df["month"] >= pd.Timestamp(TEST_START)) & df[TARGET_LABEL].notna()].copy()
    latest_month = df["month"].max()
    latest = df[df["month"] == latest_month].copy()
    return train, valid, test, latest


def feature_group(feature: str) -> str:
    for group, patterns in ABLATION_GROUPS.items():
        if any(p in feature for p in patterns):
            return group
    return "unclassified"


def manual_feature_weight(feature: str, group_weights: dict | None = None) -> float:
    weights = FEATURE_GROUP_WEIGHTS if group_weights is None else group_weights
    feature_overrides = weights.get("_feature_overrides", {}) if isinstance(weights, dict) else {}
    if feature in feature_overrides:
        return float(feature_overrides[feature])
    return float(weights.get(feature_group(feature), weights.get("unclassified", 1.0)))


def feature_tier(weight: float) -> str:
    if weight >= 1.20:
        return "tier_1_core_tail"
    if weight >= 1.10:
        return "tier_2_tail_context"
    if weight >= 1.00:
        return "tier_3_standard"
    return "tier_4_downweighted_context"


def manual_feature_weight_table(features: list[str], group_weights: dict | None = None) -> pd.DataFrame:
    rows = []
    for i, f in enumerate(features, start=1):
        w = manual_feature_weight(f, group_weights=group_weights)
        rows.append({
            "index": i,
            "feature": f,
            "tier": feature_tier(w),
            "feature_group": feature_group(f),
            "feature_weight": w,
        })
    return pd.DataFrame(rows)


def make_reference_downweighted_weights(df: pd.DataFrame, target_col: str = TARGET_LABEL) -> np.ndarray:
    y = df[target_col].fillna(0).astype(int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    base_pos = max(1.0, neg / max(pos, 1))
    w = np.where(y == 1, base_pos, 0.55).astype(float)

    if "future_max_return_1_3m" in df.columns:
        fm = df["future_max_return_1_3m"].fillna(0)
        w[(y == 0) & (fm <= 0)] *= 0.55
        w[(y == 0) & (fm > 0.20)] *= 1.25
    for col, mult in [
        ("label_boom40_top10_1_3m", 1.25),
        ("label_boom50_top5_1_3m", 1.50),
        ("label_mega100_1_3m", 2.00),
    ]:
        if col in df.columns:
            w[df[col].fillna(0).astype(int) == 1] *= mult
    return w


def fit_xgb(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    seed: int,
    params: dict | None = None,
    group_weights: dict | None = None,
) -> XGBClassifier:
    p = dict(MODEL_PARAMS if params is None else params)
    p["random_state"] = seed
    model = XGBClassifier(**p)
    X_train = train[features].replace([np.inf, -np.inf], np.nan)
    y_train = train[TARGET_LABEL].astype(int)
    w_train = make_reference_downweighted_weights(train)
    X_valid = valid[features].replace([np.inf, -np.inf], np.nan)
    y_valid = valid[TARGET_LABEL].astype(int)
    fweights = np.array([manual_feature_weight(f, group_weights=group_weights) for f in features], dtype=float)
    try:
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            feature_weights=fweights,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
    except TypeError:
        # Older xgboost builds may not expose feature_weights in the sklearn wrapper.
        model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model


def predict(model: XGBClassifier, df: pd.DataFrame, features: list[str], score_col: str = SCORE_COL, rounds: int | None = None) -> pd.DataFrame:
    out = df.copy()
    X = out[features].replace([np.inf, -np.inf], np.nan)
    if rounds is None:
        out[score_col] = model.predict_proba(X)[:, 1]
    else:
        out[score_col] = model.predict_proba(X, iteration_range=(0, rounds))[:, 1]
    return out


def final_train_valid_test_metrics(model: XGBClassifier, train, valid, test, features) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_train = predict(model, train, features, SCORE_COL)
    pred_valid = predict(model, valid, features, SCORE_COL)
    pred_test = predict(model, test, features, SCORE_COL)
    rows = [
        dataset_metrics(pred_train, SCORE_COL, "train"),
        dataset_metrics(pred_valid, SCORE_COL, "validation"),
        dataset_metrics(pred_test, SCORE_COL, "test"),
    ]
    return pd.DataFrame(rows), pred_train, pred_valid, pred_test


def training_curve_every_100_rounds(model: XGBClassifier, train, valid, test, features) -> pd.DataFrame:
    rows = []
    datasets = [("train", train), ("validation", valid), ("test", test)]
    max_rounds = int(model.get_params().get("n_estimators", MODEL_PARAMS["n_estimators"]))
    for r in [x for x in TRAINING_CURVE_ROUNDS if x <= max_rounds]:
        for dataset, df in datasets:
            pred = predict(model, df, features, SCORE_COL, rounds=r)
            row = {"boosting_round": r, "dataset": dataset}
            # Keep this compact: PR-AUC/AUC and Top-3 tail behavior are the most useful curve diagnostics.
            m = dataset_metrics(pred, SCORE_COL, dataset)
            for key in [
                "prauc", "auc", "precision_at_top3", "top3_hit30_rate", "top3_hit50_rate",
                "monthly_any_top3_hit50_rate", "top3_avg_future_return_1m", "top3_avg_future_max_return_1_3m",
            ]:
                if key in m:
                    row[key] = m[key]
            rows.append(row)
    return pd.DataFrame(rows)


def train_main(train, valid, test, latest, features):
    model = fit_xgb(train, valid, features, seed=42, params=MODEL_PARAMS)
    final_metrics, pred_train, pred_valid, pred_test = final_train_valid_test_metrics(model, train, valid, test, features)
    pred_latest = predict(model, latest, features, SCORE_COL)

    test_metrics = final_metrics[final_metrics["dataset"] == "test"].iloc[0].to_dict()
    main_result = {
        "model": "reference_downweighted_xgb_classifier",
        "target": TARGET_LABEL,
        "main_weight_profile": MAIN_WEIGHT_PROFILE,
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "feature_count": int(len(features)),
        **{k: v for k, v in test_metrics.items() if k not in {"dataset", "rows", "months", "positive_rows", "positive_rate"}},
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MAIN_MODEL_FILE))
    FEATURE_LIST_FILE.write_text("\n".join(features), encoding="utf-8")
    return model, pred_train, pred_valid, pred_test, pred_latest, final_metrics, main_result


def sort_strategy_table(out: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        "total_return_1m_rebalanced",
        "avg_monthly_return_1m",
        "avg_future_max_return_1_3m",
        "avg_boom_hit_rate",
    ]
    sort_cols = [c for c in sort_cols if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last").reset_index(drop=True)
    return out


def baseline_comparison(test_full: pd.DataFrame, test_pred: pd.DataFrame) -> pd.DataFrame:
    """Compare XGB Top-3 with independent full-universe baseline Top-3 strategies.

    XGB uses model predictions. Each baseline is recomputed independently on the
    full clean test panel using only its own score column. The comparison table
    is sorted by realized 1-month rebalanced total return first.
    """
    rows = [strategy_return_row("xgb_boom_probability", test_pred, SCORE_COL, k=3)]
    baselines = {
        "baseline_mom_3m": "mom_3m",
        "baseline_mom_5m": "mom_5m",
        "baseline_mom_6m": "mom_6m",
        "baseline_rel_mom_6m_vs_qqq": "rel_mom_6m_vs_qqq",
        "baseline_core_mom_456_avg": "core_mom_456_avg",
        "baseline_return_vol_ratio_6m": "return_vol_ratio_6m",
        "baseline_mom_6m_acceleration": "mom_6m_acceleration",
        "baseline_mom_4m": "mom_4m",
    }
    for name, col in baselines.items():
        if col in test_full.columns:
            tmp = test_full.copy()
            tmp[col] = tmp[col].fillna(tmp[col].median())
            rows.append(strategy_return_row(name, tmp, col, k=3))
    return sort_strategy_table(pd.DataFrame(rows))


def feature_weight_ablation_summary(train, valid, test, features) -> pd.DataFrame:
    """Train one seed per manual feature-weight profile and compare Top-3 outcomes.

    This tests whether giving core_momentum a stronger XGBoost feature-sampling
    prior improves realized Top-3 returns and right-tail capture.
    """
    rows = []
    for profile_name, group_weights in FEATURE_WEIGHT_PROFILES.items():
        m = fit_xgb(train, valid, features, seed=42, params=MODEL_PARAMS, group_weights=group_weights)
        score_col = f"weight_profile_{profile_name}_score"
        pt = predict(m, test, features, score_col)
        row = {
            "weight_profile": profile_name,
            "is_main_profile": profile_name == MAIN_WEIGHT_PROFILE,
            "core_momentum_group_weight": group_weights.get("core_momentum", 1.0),
            "mom_4m_weight": manual_feature_weight("mom_4m", group_weights),
            "mom_5m_weight": manual_feature_weight("mom_5m", group_weights),
            "mom_6m_weight": manual_feature_weight("mom_6m", group_weights),
            "core_mom_456_avg_weight": manual_feature_weight("core_mom_456_avg", group_weights),
            "mom_6m_acceleration_weight": manual_feature_weight("mom_6m_acceleration", group_weights),
            "relative_strength_weight": group_weights.get("relative_strength", 1.0),
            "volatility_frequency_weight": group_weights.get("volatility_frequency", 1.0),
            "etf_source_weight": group_weights.get("etf_source", 1.0),
        }
        row.update(strategy_return_row(profile_name, pt, score_col, k=3))
        row.update(topk_metrics(pt, score_col, TARGET_LABEL))
        rows.append(row)
    out = pd.DataFrame(rows)
    return sort_strategy_table(out)


def hyperparameter_ablation_summary(train, valid, test, features) -> pd.DataFrame:
    """Compare deeper/slower XGBoost parameter profiles on the same features and weights.

    This tests the user's hypothesis that more boosting rounds, deeper trees,
    and a lower learning rate may learn finer right-tail boom patterns. It uses
    the main feature-weight profile for all rows so the only changing factor is
    the XGBoost hyperparameter profile.
    """
    rows = []
    for profile_name, params in HYPERPARAMETER_ABLATION_PROFILES.items():
        m = fit_xgb(
            train,
            valid,
            features,
            seed=42,
            params=params,
            group_weights=FEATURE_GROUP_WEIGHTS,
        )
        score_col = f"param_profile_{profile_name}_score"
        pt = predict(m, test, features, score_col)
        row = {
            "param_profile": profile_name,
            "is_main_params": profile_name == "reference_2000_d4_lr0015",
            "n_estimators": params.get("n_estimators"),
            "max_depth": params.get("max_depth"),
            "learning_rate": params.get("learning_rate"),
            "min_child_weight": params.get("min_child_weight"),
            "reg_alpha": params.get("reg_alpha"),
            "reg_lambda": params.get("reg_lambda"),
            "subsample": params.get("subsample"),
            "colsample_bytree": params.get("colsample_bytree"),
            "colsample_bylevel": params.get("colsample_bylevel"),
            "colsample_bynode": params.get("colsample_bynode"),
        }
        row.update(strategy_return_row(profile_name, pt, score_col, k=3))
        row.update(topk_metrics(pt, score_col, TARGET_LABEL))
        rows.append(row)
    return sort_strategy_table(pd.DataFrame(rows))


def five_seed_stability_and_importance(train, valid, test, latest, features):
    rows = []
    latest_scores = latest[["month", "ticker"]].copy()
    importances = []
    for seed in MAIN_SEEDS:
        m = fit_xgb(train, valid, features, seed=seed, params=MODEL_PARAMS)
        score_col = f"seed_{seed}_score"
        pt = predict(m, test, features, score_col)
        pl = predict(m, latest, features, score_col)
        row = {"seed": seed}
        row.update(topk_metrics(pt, score_col, TARGET_LABEL))
        rows.append(row)
        latest_scores = latest_scores.merge(pl[["month", "ticker", score_col]], on=["month", "ticker"], how="left")
        importances.append(pd.Series(m.feature_importances_, index=features, name=str(seed)))

    seed_cols = [c for c in latest_scores.columns if c.endswith("_score")]
    latest_scores["five_seed_avg_score"] = latest_scores[seed_cols].mean(axis=1)
    latest_scores["five_seed_score_std"] = latest_scores[seed_cols].std(axis=1)

    imp_mat = pd.concat(importances, axis=1)
    # Manual weights also have an original feature-order column named "index".
    # Drop it before merging, then create a fresh rank index after sorting by
    # five-seed mean importance. Otherwise pandas raises:
    # ValueError: cannot insert index, already exists.
    weights = manual_feature_weight_table(features).drop(columns=["index"], errors="ignore")
    imp = pd.DataFrame({
        "feature": imp_mat.index,
        "mean": imp_mat.mean(axis=1).values,
        "std": imp_mat.std(axis=1).values,
        "min": imp_mat.min(axis=1).values,
        "max": imp_mat.max(axis=1).values,
    })
    imp = imp.merge(weights, on="feature", how="left")
    imp = imp.sort_values("mean", ascending=False).reset_index(drop=True)
    imp.insert(0, "index", range(1, len(imp) + 1))
    imp = imp[["index", "feature", "tier", "feature_group", "feature_weight", "mean", "std", "min", "max"]]
    return pd.DataFrame(rows), latest_scores, imp


def ablation_summary(train, valid, test, features, reference_top3: float | None = None):
    rows = []
    for group, patterns in ABLATION_GROUPS.items():
        drop_cols = columns_matching(features, patterns)
        if not drop_cols:
            continue
        keep = [f for f in features if f not in drop_cols]
        if len(keep) < 10:
            continue
        m = fit_xgb(train, valid, keep, seed=42, params=MODEL_PARAMS)
        score_col = f"abl_{group}_score"
        pt = predict(m, test, keep, score_col)
        met = topk_metrics(pt, score_col, TARGET_LABEL)
        row = {
            "ablation": f"drop_{group}",
            "dropped_features": len(drop_cols),
            "kept_features": len(keep),
        }
        row.update(met)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and reference_top3 is not None and "top3_avg_future_max_return_1_3m" in out.columns:
        out["delta_top3_future_max_vs_main"] = out["top3_avg_future_max_return_1_3m"] - reference_top3
        out = out.sort_values("delta_top3_future_max_vs_main")
    return out


def make_latest_candidates(main_latest: pd.DataFrame, five_seed_latest: pd.DataFrame) -> pd.DataFrame:
    out = main_latest.copy()
    out = out.merge(five_seed_latest[["month", "ticker", "five_seed_avg_score", "five_seed_score_std"]], on=["month", "ticker"], how="left")
    out["ensemble_score"] = out[[SCORE_COL, "five_seed_avg_score"]].mean(axis=1)
    keep_cols = ["month", "ticker", "ensemble_score", SCORE_COL, "five_seed_avg_score", "five_seed_score_std"]
    context_cols = ["mom_6m", "mom_3m", "rel_mom_6m_vs_qqq", "liquid_vol_score", "avg_dollar_volume_3m", "large_move_freq_3m", "source_count", "source_weight_sum", "theme_count"]
    keep_cols += [c for c in context_cols if c in out.columns and c not in keep_cols]
    keep_cols = [c for c in keep_cols if c in out.columns]
    final = out.sort_values("ensemble_score", ascending=False).head(30)[keep_cols].copy()
    final["rank"] = range(1, len(final) + 1)
    return final[["rank"] + [c for c in final.columns if c != "rank"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=100, help="Deprecated: kept for workflow compatibility. Training curve uses every 100 boosting rounds.")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-hyperparam-ablation", action="store_true", help="Skip the slower deep-tree hyperparameter ablation.")
    parser.add_argument("--skip-rounds", action="store_true", help="Deprecated: kept for compatibility; ignored.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_panel()
    if TARGET_LABEL not in df.columns:
        raise ValueError(f"Missing target label {TARGET_LABEL}")
    features = infer_feature_columns(df)
    train, valid, test, latest = split_panel(df)
    if train.empty or valid.empty or test.empty:
        raise ValueError("Train/validation/test split has empty segment. Check month coverage and labels.")

    model, pred_train, pred_valid, pred_test, pred_latest, final_metrics, main_result = train_main(train, valid, test, latest, features)
    final_metrics.to_csv(OUTPUT_FILES["final_metrics"], index=False)
    pd.DataFrame([main_result]).to_csv(OUTPUT_FILES["main_result"], index=False)

    curve = training_curve_every_100_rounds(model, train, valid, test, features)
    curve.to_csv(OUTPUT_FILES["training_curve"], index=False)

    baseline = baseline_comparison(test, pred_test)
    baseline.to_csv(OUTPUT_FILES["strategy_baseline"], index=False)

    weight_ablation = feature_weight_ablation_summary(train, valid, test, features)
    weight_ablation.to_csv(OUTPUT_FILES["feature_weight_ablation"], index=False)

    if not args.skip_hyperparam_ablation:
        hp_ablation = hyperparameter_ablation_summary(train, valid, test, features)
        hp_ablation.to_csv(OUTPUT_FILES["hyperparameter_ablation"], index=False)
    else:
        pd.DataFrame().to_csv(OUTPUT_FILES["hyperparameter_ablation"], index=False)

    five_report, five_latest, five_imp = five_seed_stability_and_importance(train, valid, test, latest, features)
    five_report.to_csv(OUTPUT_FILES["five_seed"], index=False)
    five_imp.to_csv(OUTPUT_FILES["five_seed_feature_importance"], index=False)

    manual_feature_weight_table(features).to_csv(OUTPUT_FILES["manual_feature_weights"], index=False)

    if not args.skip_ablation:
        ref_val = main_result.get("top3_avg_future_max_return_1_3m")
        abl = ablation_summary(train, valid, test, features, reference_top3=ref_val)
        abl.to_csv(OUTPUT_FILES["ablation"], index=False)
    else:
        pd.DataFrame().to_csv(OUTPUT_FILES["ablation"], index=False)

    latest_candidates = make_latest_candidates(pred_latest, five_latest)
    latest_candidates.to_csv(OUTPUT_FILES["latest_live"], index=False)

    top_monthly = monthly_top_table(pred_test, SCORE_COL, k=10)
    top_monthly.to_csv(OUTPUT_FILES["monthly_top"], index=False)
    recent_top3_backtest_months(pred_test, SCORE_COL, months=12).to_csv(OUTPUT_FILES["recent_top3"], index=False)
    pred_test.to_csv(OUTPUT_FILES["full_predictions"], index=False)

    metrics_json = {
        "target_label": TARGET_LABEL,
        "main_weight_profile": MAIN_WEIGHT_PROFILE,
        "feature_group_weights": FEATURE_GROUP_WEIGHTS,
        "model_params": MODEL_PARAMS,
        "features": features,
        "main_result": main_result,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "test_rows": len(test),
        "latest_month": str(latest["month"].max().date()),
        "training_curve_rounds": TRAINING_CURVE_ROUNDS,
        "hyperparameter_ablation_profiles": HYPERPARAMETER_ABLATION_PROFILES,
    }
    OUTPUT_FILES["metrics_json"].write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    print("Training complete. Key outputs written to outputs/ and models/.")


if __name__ == "__main__":
    main()
