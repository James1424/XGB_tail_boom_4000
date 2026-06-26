# Integrated ETF Tail Boom Prediction Project

This repository builds the ETF/index monthly panel and immediately trains a right-tail XGBoost boom detector in the same workflow. Large panel and prediction files are uploaded as GitHub Actions artifacts; README reports, compact CSV summaries, and the model JSON are committed to the repository.

Core target: `label_boom30_top10_1_3m`. A positive label means future 1–3 month max return is in the monthly top 10% and at least +30%.

## Model parameters

```json
{
    "n_estimators": 2000,
    "max_depth": 4,
    "learning_rate": 0.015,
    "subsample": 0.85,
    "colsample_bytree": 0.9,
    "colsample_bylevel": 0.85,
    "colsample_bynode": 0.85,
    "min_child_weight": 4,
    "reg_alpha": 0.05,
    "reg_lambda": 1.5,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1
}
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

_No data available yet._

## Training curve metrics every 100 rounds

This table evaluates the same final model at each 100-tree checkpoint on train, validation, and test. It is meant to show whether test performance is stable or only appears near the final number of trees.

_No data available yet._

## Reference-downweighted main model result

The main model is an XGBoost classifier with reference-downweighted sample weights. Easy/reference negatives are downweighted so they do not dominate the right-tail learning objective; stronger boom labels receive extra weight.

_No data available yet._

## Latest live boom candidates

Latest month candidates are ranked by an ensemble of the main model and five-seed average score.

_No data available yet._

## Strategy and baseline comparison

`xgb_boom_probability` is computed on model prediction rows. `baseline_*` strategies are computed independently on the full clean test panel, requiring only the baseline score column and future-return labels. This keeps baseline returns fixed when model feature sets change.

_No data available yet._

## Recent XGB Top-3 backtest months

_No data available yet._

## Ablation ranked summary

Each row drops one feature group and retrains the model. Negative `delta_top3_future_max_vs_main` means the removed group was useful for Top-3 tail capture.

_No data available yet._

## Five-seed training stability

This retrains the same main model with five seeds and checks whether Top-K performance is stable.

_No data available yet._

## Five-seed average feature importance

Feature importance is averaged across the five seed models. `feature_weight` is the manual XGBoost feature weight used during training, not a learned importance.

_No data available yet._

## Manual feature weights used by XGBoost

_No data available yet._

## Output files

```text
outputs/final_train_validation_test_metrics.csv
outputs/training_curve_metrics_every_100_rounds.csv
outputs/reference_downweighted_main_model_result.csv
outputs/latest_live_boom_candidates.csv
outputs/strategy_baseline_comparison.csv
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
