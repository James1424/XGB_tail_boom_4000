from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .model_config import MAIN_WEIGHT_PROFILE, MODEL_PARAMS, OUTPUT_FILES, README_FILE, TARGET_LABEL

PERCENT_HINTS = (
    "return", "rate", "precision", "hit", "probability", "score", "weight", "mean", "std", "min", "max", "importance",
)
RAW_FLOAT_COLUMNS = {"prauc", "auc", "feature_weight"}
INTEGER_COLUMNS = {"rank", "index", "seed", "months", "rows", "positive_rows", "boosting_round", "dropped_features", "kept_features", "top3_rows", "top5_rows", "top10_rows", "feature_count", "train_rows", "valid_rows", "test_rows"}


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def fmt_value(col: str, value) -> str:
    if pd.isna(value):
        return ""
    lc = col.lower()
    if lc in INTEGER_COLUMNS:
        try:
            return f"{int(round(float(value)))}"
        except Exception:
            return str(value)
    if lc in RAW_FLOAT_COLUMNS or lc.endswith("auc"):
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)
    if any(h in lc for h in PERCENT_HINTS):
        try:
            return f"{float(value) * 100:.2f}%"
        except Exception:
            return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def md_table(df: pd.DataFrame, max_rows: int = 20, cols: list[str] | None = None, sort_by: list[str] | None = None, ascending=True) -> str:
    if df.empty:
        return "_No data available yet._"
    x = df.copy()
    if sort_by:
        existing = [c for c in sort_by if c in x.columns]
        if existing:
            x = x.sort_values(existing, ascending=ascending)
    if cols:
        x = x[[c for c in cols if c in x.columns]]
    x = x.head(max_rows)
    for c in x.columns:
        x[c] = x[c].map(lambda v, col=c: fmt_value(col, v))
    return x.to_markdown(index=False)


def params_block() -> str:
    display = {k: v for k, v in MODEL_PARAMS.items() if k != "tree_method"}
    return json.dumps(display, indent=4)


def main():
    final_metrics = read_csv(OUTPUT_FILES["final_metrics"])
    main_result = read_csv(OUTPUT_FILES["main_result"])
    latest = read_csv(OUTPUT_FILES["latest_live"])
    baseline = read_csv(OUTPUT_FILES["strategy_baseline"])
    recent_top3 = read_csv(OUTPUT_FILES["recent_top3"])
    ablation = read_csv(OUTPUT_FILES["ablation"])
    five = read_csv(OUTPUT_FILES["five_seed"])
    curve = read_csv(OUTPUT_FILES["training_curve"])
    five_imp = read_csv(OUTPUT_FILES["five_seed_feature_importance"])
    manual_weights = read_csv(OUTPUT_FILES["manual_feature_weights"])
    weight_ablation = read_csv(OUTPUT_FILES["feature_weight_ablation"])
    hyperparam_ablation = read_csv(OUTPUT_FILES["hyperparameter_ablation"])

    metrics = {}
    if OUTPUT_FILES["metrics_json"].exists():
        try:
            metrics = json.loads(OUTPUT_FILES["metrics_json"].read_text(encoding="utf-8"))
        except Exception:
            metrics = {}

    text = f"""# Integrated ETF Tail Boom Prediction Project

This repository builds the ETF/index monthly panel and immediately trains a right-tail XGBoost boom detector in the same workflow. Large panel and prediction files are uploaded as GitHub Actions artifacts; README reports, compact CSV summaries, and the model JSON are committed to the repository.

Core target: `{TARGET_LABEL}`. A positive label means future 1–3 month max return is in the monthly top 10% and at least +30%.

## Model parameters

Main feature-weight profile: `{MAIN_WEIGHT_PROFILE}`.

```json
{params_block()}
```

## Run

```bash
pip install -r requirements.txt
python run_all.py
```

GitHub Actions workflow:

```text
Build Panel and Train Tail Boom Model
```

## Leakage rule

The following columns must never be used as model inputs: `future_return_*`, `future_max_return_1_3m`, `future_max_return_1_3m_pct_rank`, monthly thresholds, and every `label_*` column. The training code automatically excludes them and only keeps numeric feature columns.

## Final train / validation / test metrics

{md_table(final_metrics, max_rows=10, cols=['dataset','rows','months','positive_rows','positive_rate','prauc','auc','precision_at_top3','top3_hit30_rate','top3_hit50_rate','monthly_any_top3_hit50_rate','top3_avg_future_return_1m','top3_avg_future_max_return_1_3m'])}

## Training curve metrics every 100 rounds

This table evaluates the same final model at each 100-tree checkpoint on train, validation, and test. It is meant to show whether test performance is stable or only appears near the final number of trees.

{md_table(curve, max_rows=90, cols=['boosting_round','dataset','prauc','auc','precision_at_top3','top3_hit30_rate','top3_hit50_rate','monthly_any_top3_hit50_rate','top3_avg_future_return_1m','top3_avg_future_max_return_1_3m'])}

## Reference-downweighted main model result

The main model is an XGBoost classifier with reference-downweighted sample weights. Easy/reference negatives are downweighted so they do not dominate the right-tail learning objective; stronger boom labels receive extra weight.

{md_table(main_result, max_rows=5, cols=['model','target','train_rows','valid_rows','test_rows','feature_count','prauc','auc','precision_at_top3','top3_hit30_rate','top3_hit50_rate','monthly_any_top3_hit50_rate','top3_avg_future_return_1m','top3_avg_future_max_return_1_3m'])}

## Latest live boom candidates

Latest month candidates are ranked by an ensemble of the main model and five-seed average score.

{md_table(latest, max_rows=30, cols=['rank','month','ticker','ensemble_score','xgb_boom_probability','five_seed_avg_score','five_seed_score_std','mom_6m','mom_3m','rel_mom_6m_vs_qqq','liquid_vol_score','avg_dollar_volume_3m'])}

## Feature weight profile ablation

This section compares manual XGBoost `feature_weights` profiles. The heavier profiles test whether the strong standalone 4m / 5m / 6m / 456 momentum baselines should receive a much stronger feature-sampling prior. The table is sorted by `total_return_1m_rebalanced`, then monthly return, then future max return.

{md_table(weight_ablation, max_rows=20, cols=['weight_profile','is_main_profile','core_momentum_group_weight','mom_4m_weight','mom_5m_weight','mom_6m_weight','core_mom_456_avg_weight','mom_6m_acceleration_weight','months','total_return_1m_rebalanced','annualized_return_1m_rebalanced','avg_monthly_return_1m','avg_future_max_return_1_3m','avg_boom_hit_rate','prauc','auc','precision_at_top3','top3_hit30_rate','top3_hit50_rate','monthly_any_top3_hit50_rate'])}

## Hyperparameter ablation: deeper trees, more rounds, lower learning rate

This section tests whether a deeper and slower XGBoost can learn subtler pre-boom interactions. All rows use the same features and the same main feature-weight profile; only the XGBoost hyperparameters change. The table is sorted by realized strategy performance.

{md_table(hyperparam_ablation, max_rows=10, cols=['param_profile','is_main_params','n_estimators','max_depth','learning_rate','min_child_weight','reg_alpha','reg_lambda','subsample','colsample_bytree','months','total_return_1m_rebalanced','annualized_return_1m_rebalanced','avg_monthly_return_1m','avg_future_max_return_1_3m','avg_boom_hit_rate','prauc','auc','precision_at_top3','top3_hit30_rate','top3_hit50_rate','monthly_any_top3_hit50_rate'])}

## Strategy and baseline comparison

`xgb_boom_probability` is computed on model prediction rows. `baseline_*` strategies are computed independently on the full clean test panel, requiring only the baseline score column and future-return labels. This keeps baseline returns fixed when model feature sets change.

{md_table(baseline, max_rows=20, cols=['strategy','months','total_return_1m_rebalanced','annualized_return_1m_rebalanced','avg_monthly_return_1m','avg_future_max_return_1_3m','avg_boom_hit_rate'])}

## Recent XGB Top-3 backtest months

{md_table(recent_top3, max_rows=12, cols=['month','selected_tickers','avg_score','return_1m','future_max_return_1_3m','boom_hit_rate'])}

## Ablation ranked summary

Each row drops one feature group and retrains the model. Negative `delta_top3_future_max_vs_main` means the removed group was useful for Top-3 tail capture.

{md_table(ablation, max_rows=30, cols=['ablation','dropped_features','kept_features','top3_avg_future_max_return_1_3m','precision_at_top3','top3_hit30_rate','top3_hit50_rate','monthly_any_top3_hit50_rate','delta_top3_future_max_vs_main'])}

## Five-seed training stability

This retrains the same main model with five seeds and checks whether Top-K performance is stable.

{md_table(five, max_rows=10, cols=['seed','prauc','auc','precision_at_top3','top3_hit30_rate','top3_hit50_rate','monthly_any_top3_hit50_rate','top3_avg_future_return_1m','top3_avg_future_max_return_1_3m'])}

## Five-seed average feature importance

Feature importance is averaged across the five seed models. `feature_weight` is the manual XGBoost feature weight used during training, not a learned importance.

{md_table(five_imp, max_rows=50, cols=['index','feature','tier','feature_group','feature_weight','mean','std','min','max'])}

## Manual feature weights used by XGBoost

{md_table(manual_weights, max_rows=80, cols=['index','feature','tier','feature_group','feature_weight'])}

## Output files

```text
outputs/final_train_validation_test_metrics.csv
outputs/training_curve_metrics_every_100_rounds.csv
outputs/reference_downweighted_main_model_result.csv
outputs/latest_live_boom_candidates.csv
outputs/strategy_baseline_comparison.csv
outputs/feature_weight_ablation_summary.csv
outputs/hyperparameter_ablation_summary.csv
outputs/recent_xgb_top3_backtest_months.csv
outputs/ablation_ranked_summary.csv
outputs/five_seed_training_stability.csv
outputs/five_seed_average_feature_importance.csv
outputs/manual_feature_weights_used_by_xgboost.csv
outputs/monthly_top_predictions.csv
outputs/model_metrics.json
models/xgb_tail_event_classifier.json
models/selected_features.txt
```

## Artifact files

Large files are not committed to GitHub. They are uploaded as workflow artifacts:

```text
full-panel-files:
  outputs/raw_monthly_panel.csv
  outputs/clean_monthly_panel.csv
  data/daily_prices.csv.gz

full-prediction-files:
  outputs/full_predictions.csv
  outputs/monthly_top_predictions.csv
```

## Notes

- The model is a stock-ranking tool, not a guarantee of future return.
- The main decision output should be monthly Top-3 / Top-5 candidates, not raw probability calibration.
- The panel file is normally too large to commit. Keep it as a GitHub Actions artifact or local file; commit only summaries and model artifacts.
"""
    README_FILE.write_text(text, encoding="utf-8")
    print(f"Updated {README_FILE}")


if __name__ == "__main__":
    main()
