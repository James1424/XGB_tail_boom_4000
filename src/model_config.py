from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

# Full training panel path. In the integrated workflow, build_panel writes this file to outputs/.
# IMPORTANT: panel_head_20000.csv is only a human-readable sample and must never be used for training.
PANEL_FILE = OUTPUT_DIR / "clean_monthly_panel.csv"
PANEL_FALLBACKS = [
    DATA_DIR / "clean_monthly_panel.csv",
]
MIN_TRAINING_PANEL_ROWS = 30000

TARGET_LABEL = "label_boom30_top10_1_3m"
AUX_LABELS = [
    "label_top10_1_3m",
    "label_top5_1_3m",
    "label_boom40_top10_1_3m",
    "label_boom50_top5_1_3m",
    "label_mega100_1_3m",
]
FUTURE_RETURN_COLS = ["future_return_1m", "future_return_2m", "future_return_3m", "future_max_return_1_3m"]
LEAKAGE_COLUMNS = FUTURE_RETURN_COLS + [
    "future_max_return_1_3m_pct_rank",
    "monthly_top10_threshold_1_3m",
    "monthly_top5_threshold_1_3m",
    TARGET_LABEL,
] + AUX_LABELS
ID_COLUMNS = ["month", "ticker", "adj_close", "sources", "categories"]

TRAIN_END = "2021-12-31"
VALID_START = "2022-01-01"
VALID_END = "2023-12-31"
TEST_START = "2024-01-01"

TOP_KS = [3, 5, 10]
MAIN_SEEDS = [7, 42, 202, 777, 2026]
TRAINING_CURVE_ROUNDS = list(range(100, 4001, 100))

# Main model configuration.
#
# Previous ablations showed that the deep/slow 4000-round configuration beat
# both the old 2000-round reference and the 6000-round stress test on realized
# Top-3 strategy return. Therefore the production baseline is now fixed at
# 4000 trees, depth 5, learning_rate 0.008. Future ablations should test around
# this baseline instead of treating 2000 rounds as the reference.
MODEL_PARAMS = {
    "n_estimators": 4000,
    "max_depth": 5,
    "learning_rate": 0.008,
    "subsample": 0.85,
    "colsample_bytree": 0.90,
    "colsample_bylevel": 0.85,
    "colsample_bynode": 0.85,
    "min_child_weight": 3,
    "reg_alpha": 0.05,
    "reg_lambda": 2.00,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

# Hyperparameter ablation profiles. These are trained only for comparison in
# outputs/hyperparameter_ablation_summary.csv.
#
# The main question is now local: with depth=5 and learning_rate=0.008, where is
# the best boosting-round sweet spot? Keep the main feature-weight profile fixed
# across all rows so this table isolates training rounds. The old 2000-round
# setting is kept as a legacy anchor for comparison.
HYPERPARAMETER_ABLATION_PROFILES = {
    "legacy_2000_d4_lr0015": {
        **MODEL_PARAMS,
        "n_estimators": 2000,
        "max_depth": 4,
        "learning_rate": 0.015,
        "min_child_weight": 4,
        "reg_lambda": 1.50,
    },
    "rounds_3000_d5_lr0008": {
        **MODEL_PARAMS,
        "n_estimators": 3000,
    },
    "main_4000_d5_lr0008": dict(MODEL_PARAMS),
    "rounds_5000_d5_lr0008": {
        **MODEL_PARAMS,
        "n_estimators": 5000,
    },
    "rounds_6000_d5_lr0008": {
        **MODEL_PARAMS,
        "n_estimators": 6000,
    },
}


ABLATION_GROUPS = {
    "core_momentum": ["mom_4m", "mom_5m", "mom_6m", "core_mom_456", "mom_6m_"],
    "other_momentum": ["mom_1m", "mom_2m", "mom_3m", "mom_7m", "mom_9m", "mom_12m"],
    "relative_strength": ["rel_mom_"],
    "trend": ["price_ma", "ma5_slope", "ma10_slope", "ma20_slope", "ma30_slope", "ma50_slope", "ma100_slope"],
    "risk_drawdown": ["drawdown", "volatility_", "return_vol_ratio"],
    "volatility_frequency": ["large_move_freq", "up_big_move_freq", "down_big_move_freq", "avg_abs_daily_return", "intraday_range"],
    "liquidity_size": ["avg_dollar_volume", "dollar_volume", "trading_day_count", "liquid_vol_score"],
    "volume_flow": ["volume_change", "volume_ratio", "volume_ma", "up_day_volume", "up_day_dollar"],
    "qqq_context": ["qqq_mom_"],
    "etf_source": ["source_count", "source_weight_sum", "theme_count", "in_"],
}

# Manual feature weights passed to XGBoost through feature_weights.
#
# The baseline table shows that raw 4m / 5m / 6m momentum and core_mom_456_avg
# are very strong standalone strategies. For this right-tail boom detector, the
# main profile therefore gives these exact core momentum features a much stronger
# feature-sampling prior than generic context features.
#
# XGBoost feature_weights are not linear coefficients. They bias column sampling
# toward selected features; the tree booster still decides whether the split is
# useful by gain. Very high profiles below are stress tests for whether the model
# should behave closer to a pure momentum ranker.
FEATURE_WEIGHT_PROFILES = {
    "balanced_original": {
        "core_momentum": 1.25,
        "relative_strength": 1.15,
        "volatility_frequency": 1.15,
        "liquidity_size": 1.10,
        "other_momentum": 1.05,
        "trend": 1.00,
        "risk_drawdown": 1.00,
        "volume_flow": 0.95,
        "qqq_context": 0.95,
        "etf_source": 0.90,
        "unclassified": 1.00,
        "_feature_overrides": {},
    },
    "core_momentum_heavy": {
        "core_momentum": 2.20,
        "relative_strength": 1.25,
        "volatility_frequency": 1.15,
        "liquidity_size": 1.05,
        "other_momentum": 1.00,
        "trend": 0.95,
        "risk_drawdown": 0.90,
        "volume_flow": 0.90,
        "qqq_context": 0.85,
        "etf_source": 0.80,
        "unclassified": 1.00,
        "_feature_overrides": {
            "mom_4m": 3.00,
            "mom_5m": 3.50,
            "mom_6m": 3.50,
            "core_mom_456_avg": 4.00,
            "core_mom_456_min": 3.00,
            "core_mom_456_max": 3.00,
            "core_mom_456_std": 2.20,
            "mom_4m_vs_6m": 2.20,
            "mom_5m_vs_6m": 2.20,
            "mom_6m_first3m": 2.40,
            "mom_6m_last3m": 2.40,
            "mom_6m_acceleration": 2.60,
        },
    },
    "core_momentum_ultra": {
        "core_momentum": 3.00,
        "relative_strength": 1.20,
        "volatility_frequency": 1.10,
        "liquidity_size": 1.00,
        "other_momentum": 0.95,
        "trend": 0.90,
        "risk_drawdown": 0.85,
        "volume_flow": 0.85,
        "qqq_context": 0.80,
        "etf_source": 0.75,
        "unclassified": 1.00,
        "_feature_overrides": {
            "mom_4m": 4.00,
            "mom_5m": 5.00,
            "mom_6m": 5.00,
            "core_mom_456_avg": 5.50,
            "core_mom_456_min": 4.00,
            "core_mom_456_max": 4.00,
            "core_mom_456_std": 2.50,
            "mom_4m_vs_6m": 2.80,
            "mom_5m_vs_6m": 2.80,
            "mom_6m_first3m": 3.20,
            "mom_6m_last3m": 3.20,
            "mom_6m_acceleration": 3.50,
        },
    },
    "core_momentum_max_stress": {
        "core_momentum": 5.00,
        "relative_strength": 1.10,
        "volatility_frequency": 1.00,
        "liquidity_size": 0.95,
        "other_momentum": 0.90,
        "trend": 0.80,
        "risk_drawdown": 0.75,
        "volume_flow": 0.75,
        "qqq_context": 0.70,
        "etf_source": 0.65,
        "unclassified": 1.00,
        "_feature_overrides": {
            "mom_4m": 10.00,
            "mom_5m": 15.00,
            "mom_6m": 15.00,
            "core_mom_456_avg": 20.00,
            "core_mom_456_min": 10.00,
            "core_mom_456_max": 10.00,
            "core_mom_456_std": 5.00,
            "mom_4m_vs_6m": 5.00,
            "mom_5m_vs_6m": 5.00,
            "mom_6m_first3m": 8.00,
            "mom_6m_last3m": 8.00,
            "mom_6m_acceleration": 8.00,
        },
    },
    "core_momentum_ranker_stress": {
        "core_momentum": 8.00,
        "relative_strength": 0.95,
        "volatility_frequency": 0.85,
        "liquidity_size": 0.80,
        "other_momentum": 0.75,
        "trend": 0.65,
        "risk_drawdown": 0.60,
        "volume_flow": 0.60,
        "qqq_context": 0.55,
        "etf_source": 0.50,
        "unclassified": 0.75,
        "_feature_overrides": {
            "mom_4m": 20.00,
            "mom_5m": 30.00,
            "mom_6m": 25.00,
            "core_mom_456_avg": 35.00,
            "core_mom_456_min": 16.00,
            "core_mom_456_max": 16.00,
            "core_mom_456_std": 6.00,
            "mom_4m_vs_6m": 8.00,
            "mom_5m_vs_6m": 8.00,
            "mom_6m_first3m": 12.00,
            "mom_6m_last3m": 12.00,
            "mom_6m_acceleration": 12.00,
        },
    },
}
MAIN_WEIGHT_PROFILE = "core_momentum_heavy"
FEATURE_GROUP_WEIGHTS = FEATURE_WEIGHT_PROFILES[MAIN_WEIGHT_PROFILE]

OUTPUT_FILES = {
    "final_metrics": OUTPUT_DIR / "final_train_validation_test_metrics.csv",
    "main_result": OUTPUT_DIR / "reference_downweighted_main_model_result.csv",
    "strategy_baseline": OUTPUT_DIR / "strategy_baseline_comparison.csv",
    "latest_live": OUTPUT_DIR / "latest_live_boom_candidates.csv",
    "recent_top3": OUTPUT_DIR / "recent_xgb_top3_backtest_months.csv",
    "ablation": OUTPUT_DIR / "ablation_ranked_summary.csv",
    "five_seed": OUTPUT_DIR / "five_seed_training_stability.csv",
    "training_curve": OUTPUT_DIR / "training_curve_metrics_every_100_rounds.csv",
    "five_seed_feature_importance": OUTPUT_DIR / "five_seed_average_feature_importance.csv",
    "manual_feature_weights": OUTPUT_DIR / "manual_feature_weights_used_by_xgboost.csv",
    "feature_weight_ablation": OUTPUT_DIR / "feature_weight_ablation_summary.csv",
    "hyperparameter_ablation": OUTPUT_DIR / "hyperparameter_ablation_summary.csv",
    "monthly_top": OUTPUT_DIR / "monthly_top_predictions.csv",
    "full_predictions": OUTPUT_DIR / "full_predictions.csv",
    "metrics_json": OUTPUT_DIR / "model_metrics.json",
}
MAIN_MODEL_FILE = MODEL_DIR / "xgb_tail_event_classifier.json"
FEATURE_LIST_FILE = MODEL_DIR / "selected_features.txt"
