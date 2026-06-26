from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

# Preferred full panel path. In the integrated workflow, build_panel writes this file to outputs/.
PANEL_FILE = OUTPUT_DIR / "clean_monthly_panel.csv"
PANEL_FALLBACKS = [
    DATA_DIR / "clean_monthly_panel.csv",
    PROJECT_ROOT / "outputs" / "panel_head_20000.csv",
    DATA_DIR / "panel_head_20000.csv",
]

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
TRAINING_CURVE_ROUNDS = list(range(100, 2001, 100))

# User-requested main model configuration.
MODEL_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 4,
    "learning_rate": 0.015,
    "subsample": 0.85,
    "colsample_bytree": 0.90,
    "colsample_bylevel": 0.85,
    "colsample_bynode": 0.85,
    "min_child_weight": 4,
    "reg_alpha": 0.05,
    "reg_lambda": 1.50,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
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
FEATURE_GROUP_WEIGHTS = {
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
}

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
    "monthly_top": OUTPUT_DIR / "monthly_top_predictions.csv",
    "full_predictions": OUTPUT_DIR / "full_predictions.csv",
    "metrics_json": OUTPUT_DIR / "model_metrics.json",
}
MAIN_MODEL_FILE = MODEL_DIR / "xgb_tail_event_classifier.json"
FEATURE_LIST_FILE = MODEL_DIR / "selected_features.txt"
README_FILE = PROJECT_ROOT / "README.md"
