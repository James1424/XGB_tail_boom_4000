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
# Previous hyperparameter ablations showed that the deep/slow 4000-round
# configuration beat the old 2000-round reference and the 6000-round stress test
# on realized Top-3 strategy return. Therefore the production baseline is now
# fixed at 4000 trees, depth 5, and learning_rate 0.008.
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

# Hyperparameter ablation profiles.
#
# Keep the main feature-weight profile fixed and test the local boosting-round
# sweet spot around the new 4000-round baseline. The legacy 2000-round profile is
# retained only as an old-reference anchor.
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
# Important: XGBoost feature_weights are not linear coefficients. They bias
# column sampling toward selected features; the booster still chooses splits by
# gain. Here the former core_momentum_max_stress profile is intentionally kept as
# the main/original profile. All lighter profiles from the previous experiment
# were removed. The four additional profiles below are more aggressive stress
# tests that try to push XGB closer to the strong standalone momentum baselines
# such as baseline_mom_5m, baseline_core_mom_456_avg, and baseline_mom_4m.
FEATURE_WEIGHT_PROFILES = {
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
    "core_momentum_aggressive_1": {
        "core_momentum": 7.00,
        "relative_strength": 1.00,
        "volatility_frequency": 0.90,
        "liquidity_size": 0.85,
        "other_momentum": 0.80,
        "trend": 0.70,
        "risk_drawdown": 0.65,
        "volume_flow": 0.65,
        "qqq_context": 0.60,
        "etf_source": 0.55,
        "unclassified": 0.90,
        "_feature_overrides": {
            "mom_4m": 16.00,
            "mom_5m": 24.00,
            "mom_6m": 24.00,
            "core_mom_456_avg": 32.00,
            "core_mom_456_min": 16.00,
            "core_mom_456_max": 16.00,
            "core_mom_456_std": 7.00,
            "mom_4m_vs_6m": 7.00,
            "mom_5m_vs_6m": 7.00,
            "mom_6m_first3m": 12.00,
            "mom_6m_last3m": 12.00,
            "mom_6m_acceleration": 12.00,
        },
    },
    "core_momentum_aggressive_2": {
        "core_momentum": 10.00,
        "relative_strength": 0.90,
        "volatility_frequency": 0.80,
        "liquidity_size": 0.75,
        "other_momentum": 0.70,
        "trend": 0.60,
        "risk_drawdown": 0.55,
        "volume_flow": 0.55,
        "qqq_context": 0.50,
        "etf_source": 0.45,
        "unclassified": 0.80,
        "_feature_overrides": {
            "mom_4m": 25.00,
            "mom_5m": 38.00,
            "mom_6m": 35.00,
            "core_mom_456_avg": 50.00,
            "core_mom_456_min": 25.00,
            "core_mom_456_max": 25.00,
            "core_mom_456_std": 9.00,
            "mom_4m_vs_6m": 10.00,
            "mom_5m_vs_6m": 10.00,
            "mom_6m_first3m": 18.00,
            "mom_6m_last3m": 18.00,
            "mom_6m_acceleration": 18.00,
        },
    },
    "core_momentum_aggressive_3": {
        "core_momentum": 14.00,
        "relative_strength": 0.80,
        "volatility_frequency": 0.70,
        "liquidity_size": 0.65,
        "other_momentum": 0.60,
        "trend": 0.50,
        "risk_drawdown": 0.45,
        "volume_flow": 0.45,
        "qqq_context": 0.40,
        "etf_source": 0.35,
        "unclassified": 0.70,
        "_feature_overrides": {
            "mom_4m": 40.00,
            "mom_5m": 60.00,
            "mom_6m": 55.00,
            "core_mom_456_avg": 80.00,
            "core_mom_456_min": 40.00,
            "core_mom_456_max": 40.00,
            "core_mom_456_std": 12.00,
            "mom_4m_vs_6m": 15.00,
            "mom_5m_vs_6m": 15.00,
            "mom_6m_first3m": 28.00,
            "mom_6m_last3m": 28.00,
            "mom_6m_acceleration": 28.00,
        },
    },
    "core_momentum_pure_ranker_stress": {
        "core_momentum": 20.00,
        "relative_strength": 0.70,
        "volatility_frequency": 0.55,
        "liquidity_size": 0.50,
        "other_momentum": 0.50,
        "trend": 0.35,
        "risk_drawdown": 0.30,
        "volume_flow": 0.30,
        "qqq_context": 0.25,
        "etf_source": 0.20,
        "unclassified": 0.50,
        "_feature_overrides": {
            "mom_4m": 60.00,
            "mom_5m": 100.00,
            "mom_6m": 90.00,
            "core_mom_456_avg": 130.00,
            "core_mom_456_min": 60.00,
            "core_mom_456_max": 60.00,
            "core_mom_456_std": 18.00,
            "mom_4m_vs_6m": 22.00,
            "mom_5m_vs_6m": 22.00,
            "mom_6m_first3m": 45.00,
            "mom_6m_last3m": 45.00,
            "mom_6m_acceleration": 45.00,
        },
    },
}
MAIN_WEIGHT_PROFILE = "core_momentum_max_stress"
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
