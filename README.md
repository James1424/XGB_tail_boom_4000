# Integrated ETF Tail Boom Prediction Project

This repository builds the ETF/index monthly panel and immediately trains a right-tail XGBoost boom detector in the same workflow. Large panel and prediction files are uploaded as GitHub Actions artifacts; README reports, compact CSV summaries, and the model JSON are committed to the repository.

Core target: `label_boom30_top10_1_3m`. A positive label means future 1–3 month max return is in the monthly top 10% and at least +30%.

## Model parameters

Main feature-weight profile: `momentum_boosted`.

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

| dataset    |   rows |   months |   positive_rows | positive_rate   |   prauc |    auc | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   | top3_avg_future_return_1m   | top3_avg_future_max_return_1_3m   |
|:-----------|-------:|---------:|----------------:|:----------------|--------:|-------:|:--------------------|:------------------|:------------------|:------------------------------|:----------------------------|:----------------------------------|
| train      |  28791 |       72 |            1228 | 4.27%           |  0.8451 | 0.9926 | 88.43%              | 88.43%            | 54.17%            | 86.11%                        | 19.80%                      | 60.32%                            |
| validation |   9964 |       24 |             529 | 5.31%           |  0.2052 | 0.8    | 33.33%              | 34.72%            | 12.50%            | 29.17%                        | 3.36%                       | 21.78%                            |
| test       |  12636 |       30 |             752 | 5.95%           |  0.2435 | 0.8256 | 28.89%              | 30.00%            | 26.67%            | 53.33%                        | 7.13%                       | 28.73%                            |

## Training curve metrics every 100 rounds

This table evaluates the same final model at each 100-tree checkpoint on train, validation, and test. It is meant to show whether test performance is stable or only appears near the final number of trees.

|   boosting_round | dataset    |   prauc |    auc | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   | top3_avg_future_return_1m   | top3_avg_future_max_return_1_3m   |
|-----------------:|:-----------|--------:|-------:|:--------------------|:------------------|:------------------|:------------------------------|:----------------------------|:----------------------------------|
|              100 | train      |  0.3069 | 0.8926 | 45.37%              | 45.37%            | 29.63%            | 61.11%                        | 9.24%                       | 36.63%                            |
|              100 | validation |  0.1971 | 0.8047 | 33.33%              | 34.72%            | 16.67%            | 41.67%                        | 1.29%                       | 26.63%                            |
|              100 | test       |  0.2793 | 0.8322 | 35.56%              | 35.56%            | 27.78%            | 53.33%                        | 9.50%                       | 35.92%                            |
|              200 | train      |  0.3377 | 0.9067 | 47.69%              | 47.69%            | 33.33%            | 69.44%                        | 9.59%                       | 38.86%                            |
|              200 | validation |  0.1992 | 0.8107 | 36.11%              | 37.50%            | 16.67%            | 41.67%                        | 4.97%                       | 29.25%                            |
|              200 | test       |  0.2784 | 0.8329 | 37.78%              | 37.78%            | 30.00%            | 56.67%                        | 7.99%                       | 35.63%                            |
|              300 | train      |  0.3677 | 0.9179 | 50.46%              | 50.46%            | 37.04%            | 75.00%                        | 10.47%                      | 42.61%                            |
|              300 | validation |  0.207  | 0.8124 | 31.94%              | 33.33%            | 15.28%            | 33.33%                        | 3.22%                       | 24.47%                            |
|              300 | test       |  0.2753 | 0.8331 | 41.11%              | 41.11%            | 31.11%            | 53.33%                        | 10.71%                      | 40.19%                            |
|              400 | train      |  0.4067 | 0.9292 | 53.24%              | 53.24%            | 37.96%            | 75.00%                        | 11.58%                      | 43.73%                            |
|              400 | validation |  0.2117 | 0.8128 | 34.72%              | 36.11%            | 16.67%            | 41.67%                        | 2.21%                       | 26.81%                            |
|              400 | test       |  0.2705 | 0.8332 | 41.11%              | 41.11%            | 30.00%            | 56.67%                        | 11.22%                      | 36.09%                            |
|              500 | train      |  0.4368 | 0.9386 | 55.09%              | 55.09%            | 38.89%            | 79.17%                        | 13.04%                      | 45.17%                            |
|              500 | validation |  0.212  | 0.8146 | 34.72%              | 36.11%            | 13.89%            | 37.50%                        | 3.18%                       | 24.63%                            |
|              500 | test       |  0.2661 | 0.8328 | 37.78%              | 37.78%            | 27.78%            | 56.67%                        | 10.67%                      | 37.03%                            |
|              600 | train      |  0.4648 | 0.9465 | 57.41%              | 57.41%            | 41.20%            | 80.56%                        | 12.98%                      | 46.80%                            |
|              600 | validation |  0.2103 | 0.8136 | 34.72%              | 36.11%            | 12.50%            | 37.50%                        | 1.84%                       | 24.42%                            |
|              600 | test       |  0.266  | 0.833  | 37.78%              | 37.78%            | 26.67%            | 53.33%                        | 9.20%                       | 32.84%                            |
|              700 | train      |  0.4966 | 0.9532 | 58.80%              | 58.80%            | 42.13%            | 80.56%                        | 14.22%                      | 47.83%                            |
|              700 | validation |  0.2092 | 0.8124 | 33.33%              | 34.72%            | 13.89%            | 37.50%                        | 2.75%                       | 24.81%                            |
|              700 | test       |  0.2642 | 0.8324 | 35.56%              | 35.56%            | 26.67%            | 53.33%                        | 9.82%                       | 35.22%                            |
|              800 | train      |  0.5318 | 0.9593 | 62.04%              | 62.04%            | 43.98%            | 81.94%                        | 14.82%                      | 49.04%                            |
|              800 | validation |  0.2076 | 0.8111 | 31.94%              | 33.33%            | 12.50%            | 37.50%                        | 2.76%                       | 23.81%                            |
|              800 | test       |  0.2624 | 0.8322 | 33.33%              | 34.44%            | 25.56%            | 53.33%                        | 7.90%                       | 31.40%                            |
|              900 | train      |  0.5619 | 0.9645 | 65.74%              | 65.74%            | 44.91%            | 81.94%                        | 15.06%                      | 50.06%                            |
|              900 | validation |  0.2061 | 0.8106 | 31.94%              | 33.33%            | 12.50%            | 37.50%                        | 2.02%                       | 25.19%                            |
|              900 | test       |  0.2605 | 0.8312 | 33.33%              | 33.33%            | 27.78%            | 50.00%                        | 7.67%                       | 32.37%                            |
|             1000 | train      |  0.5948 | 0.969  | 68.52%              | 68.52%            | 46.30%            | 81.94%                        | 16.14%                      | 50.96%                            |
|             1000 | validation |  0.2055 | 0.8091 | 34.72%              | 36.11%            | 15.28%            | 41.67%                        | 3.04%                       | 26.41%                            |
|             1000 | test       |  0.2591 | 0.8307 | 30.00%              | 30.00%            | 25.56%            | 50.00%                        | 5.55%                       | 28.67%                            |
|             1100 | train      |  0.6234 | 0.9729 | 71.30%              | 71.30%            | 49.07%            | 84.72%                        | 16.10%                      | 52.25%                            |
|             1100 | validation |  0.2057 | 0.8078 | 33.33%              | 34.72%            | 15.28%            | 41.67%                        | 2.13%                       | 25.76%                            |
|             1100 | test       |  0.2569 | 0.8303 | 30.00%              | 31.11%            | 25.56%            | 53.33%                        | 7.70%                       | 30.48%                            |
|             1200 | train      |  0.6523 | 0.9763 | 74.07%              | 74.07%            | 49.07%            | 84.72%                        | 17.10%                      | 53.96%                            |
|             1200 | validation |  0.2054 | 0.8073 | 33.33%              | 34.72%            | 13.89%            | 37.50%                        | 3.14%                       | 23.84%                            |
|             1200 | test       |  0.2551 | 0.8298 | 30.00%              | 31.11%            | 25.56%            | 53.33%                        | 8.20%                       | 29.72%                            |
|             1300 | train      |  0.6802 | 0.9793 | 75.46%              | 75.46%            | 49.54%            | 84.72%                        | 17.45%                      | 55.32%                            |
|             1300 | validation |  0.2054 | 0.8061 | 30.56%              | 31.94%            | 11.11%            | 29.17%                        | 2.71%                       | 20.24%                            |
|             1300 | test       |  0.2546 | 0.8293 | 28.89%              | 30.00%            | 25.56%            | 56.67%                        | 7.77%                       | 30.52%                            |
|             1400 | train      |  0.7046 | 0.9819 | 76.85%              | 76.85%            | 50.00%            | 84.72%                        | 18.15%                      | 55.94%                            |
|             1400 | validation |  0.2059 | 0.8052 | 30.56%              | 31.94%            | 11.11%            | 29.17%                        | 1.72%                       | 20.70%                            |
|             1400 | test       |  0.2526 | 0.8287 | 28.89%              | 30.00%            | 25.56%            | 56.67%                        | 8.24%                       | 30.88%                            |
|             1500 | train      |  0.7343 | 0.9844 | 78.70%              | 78.70%            | 50.00%            | 84.72%                        | 18.49%                      | 56.01%                            |
|             1500 | validation |  0.2056 | 0.8045 | 29.17%              | 30.56%            | 9.72%             | 25.00%                        | 1.63%                       | 19.81%                            |
|             1500 | test       |  0.2496 | 0.8276 | 30.00%              | 30.00%            | 26.67%            | 56.67%                        | 7.97%                       | 31.98%                            |
|             1600 | train      |  0.7578 | 0.9864 | 81.94%              | 81.94%            | 51.39%            | 86.11%                        | 18.80%                      | 57.46%                            |
|             1600 | validation |  0.2051 | 0.8035 | 29.17%              | 30.56%            | 9.72%             | 25.00%                        | 1.08%                       | 19.07%                            |
|             1600 | test       |  0.2488 | 0.8273 | 31.11%              | 31.11%            | 27.78%            | 56.67%                        | 8.88%                       | 32.86%                            |
|             1700 | train      |  0.7825 | 0.9883 | 84.26%              | 84.26%            | 51.85%            | 86.11%                        | 18.95%                      | 58.20%                            |
|             1700 | validation |  0.2052 | 0.8032 | 29.17%              | 30.56%            | 9.72%             | 25.00%                        | 1.53%                       | 19.93%                            |
|             1700 | test       |  0.2471 | 0.827  | 28.89%              | 30.00%            | 26.67%            | 56.67%                        | 9.02%                       | 31.18%                            |
|             1800 | train      |  0.804  | 0.9899 | 85.65%              | 85.65%            | 52.78%            | 86.11%                        | 19.00%                      | 58.59%                            |
|             1800 | validation |  0.2051 | 0.8019 | 31.94%              | 33.33%            | 11.11%            | 25.00%                        | 2.30%                       | 21.00%                            |
|             1800 | test       |  0.2472 | 0.8267 | 27.78%              | 28.89%            | 24.44%            | 53.33%                        | 7.30%                       | 27.65%                            |
|             1900 | train      |  0.8258 | 0.9913 | 86.57%              | 86.57%            | 53.24%            | 86.11%                        | 19.31%                      | 59.24%                            |
|             1900 | validation |  0.2048 | 0.8011 | 30.56%              | 31.94%            | 9.72%             | 25.00%                        | 1.93%                       | 19.45%                            |
|             1900 | test       |  0.2448 | 0.8263 | 28.89%              | 30.00%            | 26.67%            | 53.33%                        | 6.98%                       | 29.18%                            |
|             2000 | train      |  0.8451 | 0.9926 | 88.43%              | 88.43%            | 54.17%            | 86.11%                        | 19.80%                      | 60.32%                            |
|             2000 | validation |  0.2052 | 0.8    | 33.33%              | 34.72%            | 12.50%            | 29.17%                        | 3.36%                       | 21.78%                            |
|             2000 | test       |  0.2435 | 0.8256 | 28.89%              | 30.00%            | 26.67%            | 53.33%                        | 7.13%                       | 28.73%                            |

## Reference-downweighted main model result

The main model is an XGBoost classifier with reference-downweighted sample weights. Easy/reference negatives are downweighted so they do not dominate the right-tail learning objective; stronger boom labels receive extra weight.

| model                                 | target                  |   train_rows |   valid_rows |   test_rows |   feature_count |   prauc |    auc | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   | top3_avg_future_return_1m   | top3_avg_future_max_return_1_3m   |
|:--------------------------------------|:------------------------|-------------:|-------------:|------------:|----------------:|--------:|-------:|:--------------------|:------------------|:------------------|:------------------------------|:----------------------------|:----------------------------------|
| reference_downweighted_xgb_classifier | label_boom30_top10_1_3m |        28791 |         9964 |       12636 |             110 |  0.2435 | 0.8256 | 28.89%              | 30.00%            | 26.67%            | 53.33%                        | 7.13%                       | 28.73%                            |

## Latest live boom candidates

Latest month candidates are ranked by an ensemble of the main model and five-seed average score.

|   rank | month      | ticker   | ensemble_score   | xgb_boom_probability   | five_seed_avg_score   | five_seed_score_std   |   mom_6m |   mom_3m |   rel_mom_6m_vs_qqq | liquid_vol_score   |   avg_dollar_volume_3m |
|-------:|:-----------|:---------|:-----------------|:-----------------------|:----------------------|:----------------------|---------:|---------:|--------------------:|:-------------------|-----------------------:|
|      1 | 2026-06-30 | FDXF     | 98.28%           | 98.23%                 | 98.33%                | 0.20%                 |          |          |                     | 0.00%              |                        |
|      2 | 2026-06-30 | AMD      | 96.90%           | 97.07%                 | 96.73%                | 0.49%                 |   1.4325 |   1.5608 |              1.2685 | 96.48%             |            1.42692e+10 |
|      3 | 2026-06-30 | RKLB     | 95.54%           | 95.71%                 | 95.37%                | 1.21%                 |   0.2265 |   0.3323 |              0.0625 | 97.61%             |            2.72435e+09 |
|      4 | 2026-06-30 | ALAB     | 94.40%           | 94.16%                 | 94.65%                | 1.19%                 |   1.3233 |   2.5265 |              1.1593 | 95.25%             |            1.5709e+09  |
|      5 | 2026-06-30 | ECHO     | 94.32%           | 94.58%                 | 94.06%                | 0.51%                 |          |          |                     | 0.00%              |                        |
|      6 | 2026-06-30 | TER      | 93.86%           | 93.46%                 | 94.26%                | 0.75%                 |   1.2752 |   0.4849 |              1.1112 | 93.31%             |            1.49257e+09 |
|      7 | 2026-06-30 | ENPH     | 93.52%           | 94.27%                 | 92.78%                | 1.94%                 |   0.4778 |   0.2527 |              0.3138 | 79.88%             |            3.6444e+08  |
|      8 | 2026-06-30 | DDOG     | 92.86%           | 93.88%                 | 91.85%                | 2.03%                 |   0.7191 |   0.9803 |              0.5551 | 89.61%             |            1.05701e+09 |
|      9 | 2026-06-30 | PLUG     | 92.34%           | 93.01%                 | 91.67%                | 1.71%                 |   0.2893 |   0.1238 |              0.1253 | 73.60%             |            2.36014e+08 |
|     10 | 2026-06-30 | ASML     | 91.84%           | 92.00%                 | 91.69%                | 0.91%                 |   0.6868 |   0.3645 |              0.5228 | 89.93%             |            2.94967e+09 |
|     11 | 2026-06-30 | ANET     | 91.56%           | 91.35%                 | 91.77%                | 1.19%                 |   0.2067 |   0.2878 |              0.0427 | 90.06%             |            1.42067e+09 |
|     12 | 2026-06-30 | MSTR     | 90.87%           | 92.09%                 | 89.65%                | 2.72%                 |  -0.4389 |  -0.3168 |             -0.6029 | 95.95%             |            2.7694e+09  |
|     13 | 2026-06-30 | SMCI     | 90.82%           | 91.55%                 | 90.09%                | 1.52%                 |   0.0565 |   0.3581 |             -0.1075 | 94.83%             |            1.57075e+09 |
|     14 | 2026-06-30 | DELL     | 90.54%           | 91.12%                 | 89.96%                | 1.72%                 |   2.1206 |   1.3829 |              1.9566 | 93.96%             |            2.66274e+09 |
|     15 | 2026-06-30 | MU       | 89.30%           | 88.55%                 | 90.06%                | 1.61%                 |   3.1519 |   2.5061 |              2.9879 | 99.28%             |            3.79625e+10 |
|     16 | 2026-06-30 | FIX      | 89.05%           | 88.47%                 | 89.63%                | 1.36%                 |   1.0929 |   0.4157 |              0.9289 | 85.09%             |            7.70275e+08 |
|     17 | 2026-06-30 | JBL      | 88.70%           | 89.06%                 | 88.33%                | 0.56%                 |   0.5856 |   0.3607 |              0.4216 | 70.95%             |            4.2772e+08  |
|     18 | 2026-06-30 | CDNS     | 87.80%           | 88.10%                 | 87.50%                | 0.76%                 |   0.1954 |   0.3447 |              0.0314 | 76.75%             |            8.44929e+08 |
|     19 | 2026-06-30 | FDS      | 87.77%           | 88.56%                 | 86.97%                | 2.08%                 |  -0.203  |   0.0605 |             -0.367  | 64.28%             |            2.16777e+08 |
|     20 | 2026-06-30 | MPWR     | 86.93%           | 87.74%                 | 86.12%                | 2.93%                 |   0.4851 |   0.2287 |              0.3211 | 89.08%             |            1.08927e+09 |
|     21 | 2026-06-30 | CRWD     | 86.60%           | 85.64%                 | 87.57%                | 2.31%                 |   0.4841 |   0.782  |              0.3201 | 88.31%             |            1.88014e+09 |
|     22 | 2026-06-30 | COIN     | 86.30%           | 85.48%                 | 87.13%                | 1.32%                 |  -0.3428 |  -0.1488 |             -0.5068 | 93.02%             |            1.7273e+09  |
|     23 | 2026-06-30 | CIEN     | 84.75%           | 83.07%                 | 86.44%                | 2.85%                 |   1.0154 |   0.2141 |              0.8514 | 93.10%             |            1.3761e+09  |
|     24 | 2026-06-30 | LRCX     | 83.97%           | 82.78%                 | 85.16%                | 2.41%                 |   1.2284 |   0.7832 |              1.0644 | 94.74%             |            3.0744e+09  |
|     25 | 2026-06-30 | MELI     | 83.31%           | 84.64%                 | 81.99%                | 2.55%                 |  -0.1608 |  -0.0224 |             -0.3248 | 73.85%             |            9.07892e+08 |
|     26 | 2026-06-30 | LEU      | 83.15%           | 83.07%                 | 83.24%                | 1.58%                 |  -0.3185 |  -0.0469 |             -0.4825 | 67.65%             |            1.68235e+08 |
|     27 | 2026-06-30 | CEG      | 82.86%           | 80.88%                 | 84.85%                | 2.78%                 |  -0.2487 |  -0.0508 |             -0.4127 | 83.57%             |            9.77595e+08 |
|     28 | 2026-06-30 | COHR     | 82.79%           | 82.73%                 | 82.86%                | 1.77%                 |   1.0809 |   0.6124 |              0.9169 | 95.45%             |            2.10977e+09 |
|     29 | 2026-06-30 | SHOP     | 82.75%           | 82.93%                 | 82.57%                | 1.01%                 |  -0.2755 |  -0.0169 |             -0.4395 | 89.66%             |            1.13523e+09 |
|     30 | 2026-06-30 | MRVL     | 82.43%           | 81.92%                 | 82.93%                | 3.35%                 |   2.1487 |   1.6995 |              1.9847 | 96.95%             |            9.43631e+09 |

## Feature weight profile ablation

This section compares three manual XGBoost `feature_weights` profiles. Baseline uses the original mild core-momentum boost; boosted is the main model profile; aggressive tests whether core momentum is being over-emphasized. The table is sorted by `total_return_1m_rebalanced`, then monthly return, then future max return.

| weight_profile      | is_main_profile   | core_momentum_weight   | relative_strength_weight   | volatility_frequency_weight   | etf_source_weight   |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |   prauc |    auc | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   |
|:--------------------|:------------------|:-----------------------|:---------------------------|:------------------------------|:--------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:-----------------------------|:--------------------|--------:|-------:|:--------------------|:------------------|:------------------|:------------------------------|
| momentum_aggressive | False             | 200.00%                | 135.00%                    | 125.00%                       | 80.00%              |       29 | 414.46%                      | 96.95%                            | 7.96%                   | 30.68%                       | 32.18%              |  0.2413 | 0.8257 | 31.11%              | 32.22%            | 27.78%            | 53.33%                        |
| momentum_boosted    | True              | 160.00%                | 125.00%                    | 120.00%                       | 85.00%              |       29 | 323.90%                      | 81.78%                            | 7.13%                   | 28.73%                       | 29.89%              |  0.2435 | 0.8256 | 28.89%              | 30.00%            | 26.67%            | 53.33%                        |
| balanced_original   | False             | 125.00%                | 115.00%                    | 115.00%                       | 90.00%              |       29 | 160.71%                      | 48.66%                            | 4.95%                   | 28.94%                       | 29.89%              |  0.2432 | 0.8256 | 28.89%              | 28.89%            | 26.67%            | 46.67%                        |

## Strategy and baseline comparison

`xgb_boom_probability` is computed on model prediction rows. `baseline_*` strategies are computed independently on the full clean test panel, requiring only the baseline score column and future-return labels. This keeps baseline returns fixed when model feature sets change.

| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:-----------------------------|:--------------------|
| baseline_mom_5m              |       29 | 1332.51%                     | 200.87%                           | 11.33%                  | 38.32%                       | 50.57%              |
| baseline_mom_4m              |       29 | 1308.00%                     | 198.73%                           | 10.95%                  | 38.53%                       | 50.57%              |
| baseline_core_mom_456_avg    |       29 | 1152.81%                     | 184.64%                           | 10.52%                  | 37.92%                       | 50.57%              |
| baseline_mom_6m              |       29 | 823.99%                      | 150.95%                           | 9.48%                   | 34.89%                       | 47.13%              |
| baseline_rel_mom_6m_vs_qqq   |       29 | 823.99%                      | 150.95%                           | 9.48%                   | 34.89%                       | 47.13%              |
| baseline_mom_6m_acceleration |       29 | 496.81%                      | 109.43%                           | 7.97%                   | 28.61%                       | 35.63%              |
| baseline_mom_3m              |       29 | 378.93%                      | 91.20%                            | 6.94%                   | 30.66%                       | 43.68%              |
| xgb_boom_probability         |       29 | 323.90%                      | 81.78%                            | 7.13%                   | 28.73%                       | 29.89%              |
| baseline_return_vol_ratio_6m |       29 | 105.91%                      | 34.83%                            | 3.10%                   | 19.16%                       | 24.14%              |

## Recent XGB Top-3 backtest months

| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-07-31 | SEDG, LEU, ENPH    | 95.94%      | 13.99%      | 43.76%                   | 66.67%          |
| 2025-08-31 | ALAB, SEDG, UUUU   | 97.30%      | 16.55%      | 31.43%                   | 33.33%          |
| 2025-09-30 | SEDG, LEU, ENPH    | 95.88%      | -0.15%      | 2.60%                    | 0.00%           |
| 2025-10-31 | LEU, SEDG, TSLA    | 95.60%      | -10.37%     | -7.22%                   | 0.00%           |
| 2025-11-30 | Q, ALAB, LEU       | 97.57%      | -0.04%      | 23.11%                   | 33.33%          |
| 2025-12-31 | SEDG, LEU, COHR    | 97.58%      | 12.29%      | 43.96%                   | 66.67%          |
| 2026-01-31 | SEDG, RKLB, MSTR   | 97.76%      | -4.27%      | 26.17%                   | 33.33%          |
| 2026-02-28 | LEU, RKLB, MSTR    | 97.45%      | -8.34%      | 46.51%                   | 33.33%          |
| 2026-03-31 | ALAB, LEU, MSTR    | 97.00%      | 43.93%      | 102.25%                  | 33.33%          |
| 2026-04-30 | ALAB, SEDG, ENPH   | 97.54%      | 87.20%      | 94.67%                   | 100.00%         |
| 2026-05-31 | ALAB, ENPH, COIN   | 96.57%      | -13.12%     | -13.12%                  | 0.00%           |
| 2026-06-30 | FDXF, AMD, RKLB    | 97.00%      |             |                          | 0.00%           |

## Ablation ranked summary

Each row drops one feature group and retrains the model. Negative `delta_top3_future_max_vs_main` means the removed group was useful for Top-3 tail capture.

| ablation                  |   dropped_features |   kept_features | top3_avg_future_max_return_1_3m   | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   | delta_top3_future_max_vs_main   |
|:--------------------------|-------------------:|----------------:|:----------------------------------|:--------------------|:------------------|:------------------|:------------------------------|:--------------------------------|
| drop_volatility_frequency |                 11 |              99 | 24.00%                            | 28.89%              | 28.89%            | 24.44%            | 43.33%                        | -4.73%                          |
| drop_etf_source           |                  6 |             104 | 27.03%                            | 30.00%              | 30.00%            | 24.44%            | 53.33%                        | -1.70%                          |
| drop_trend                |                 12 |              98 | 28.20%                            | 28.89%              | 28.89%            | 25.56%            | 46.67%                        | -0.53%                          |
| drop_other_momentum       |                 13 |              97 | 28.77%                            | 27.78%              | 28.89%            | 25.56%            | 53.33%                        | 0.04%                           |
| drop_core_momentum        |                 14 |              96 | 28.90%                            | 33.33%              | 33.33%            | 24.44%            | 43.33%                        | 0.17%                           |
| drop_risk_drawdown        |                 19 |              91 | 29.47%                            | 28.89%              | 28.89%            | 25.56%            | 50.00%                        | 0.73%                           |
| drop_liquidity_size       |                 11 |              99 | 29.57%                            | 31.11%              | 32.22%            | 24.44%            | 56.67%                        | 0.84%                           |
| drop_qqq_context          |                  4 |             106 | 31.77%                            | 36.67%              | 36.67%            | 32.22%            | 60.00%                        | 3.04%                           |
| drop_volume_flow          |                  9 |             101 | 32.51%                            | 34.44%              | 34.44%            | 30.00%            | 56.67%                        | 3.77%                           |
| drop_relative_strength    |                  4 |             106 | 35.20%                            | 34.44%              | 34.44%            | 31.11%            | 60.00%                        | 6.47%                           |

## Five-seed training stability

This retrains the same main model with five seeds and checks whether Top-K performance is stable.

|   seed |   prauc |    auc | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   | top3_avg_future_return_1m   | top3_avg_future_max_return_1_3m   |
|-------:|--------:|-------:|:--------------------|:------------------|:------------------|:------------------------------|:----------------------------|:----------------------------------|
|      7 |  0.2434 | 0.8259 | 30.00%              | 30.00%            | 27.78%            | 50.00%                        | 5.80%                       | 29.76%                            |
|     42 |  0.2435 | 0.8256 | 28.89%              | 30.00%            | 26.67%            | 53.33%                        | 7.13%                       | 28.73%                            |
|    202 |  0.2389 | 0.8252 | 26.67%              | 26.67%            | 25.56%            | 46.67%                        | 4.45%                       | 28.34%                            |
|    777 |  0.2449 | 0.8259 | 32.22%              | 32.22%            | 28.89%            | 53.33%                        | 6.83%                       | 33.55%                            |
|   2026 |  0.2409 | 0.8264 | 27.78%              | 28.89%            | 24.44%            | 46.67%                        | 4.80%                       | 29.11%                            |

## Five-seed average feature importance

Feature importance is averaged across the five seed models. `feature_weight` is the manual XGBoost feature weight used during training, not a learned importance.

|   index | feature                  | tier                        | feature_group        |   feature_weight | mean   | std   | min   | max   |
|--------:|:-------------------------|:----------------------------|:---------------------|-----------------:|:-------|:------|:------|:------|
|       1 | intraday_range_mean_3m   | tier_1_core_tail            | volatility_frequency |             1.2  | 8.14%  | 1.06% | 7.28% | 9.86% |
|       2 | volatility_6m            | tier_4_downweighted_context | risk_drawdown        |             0.95 | 7.32%  | 0.69% | 6.60% | 8.28% |
|       3 | intraday_range_mean_6m   | tier_1_core_tail            | volatility_frequency |             1.2  | 4.20%  | 0.34% | 3.63% | 4.46% |
|       4 | atr_14_to_price          | tier_3_standard             | unclassified         |             1    | 2.49%  | 0.09% | 2.37% | 2.61% |
|       5 | in_manual_core           | tier_4_downweighted_context | etf_source           |             0.85 | 2.09%  | 0.11% | 2.02% | 2.29% |
|       6 | avg_abs_daily_return_6m  | tier_1_core_tail            | volatility_frequency |             1.2  | 1.51%  | 0.44% | 0.98% | 2.04% |
|       7 | liquid_vol_score         | tier_2_tail_context         | liquidity_size       |             1.1  | 1.45%  | 0.04% | 1.40% | 1.49% |
|       8 | source_count             | tier_4_downweighted_context | etf_source           |             0.85 | 1.38%  | 0.15% | 1.21% | 1.57% |
|       9 | volatility_3m            | tier_4_downweighted_context | risk_drawdown        |             0.95 | 1.34%  | 0.13% | 1.17% | 1.47% |
|      10 | source_weight_sum        | tier_4_downweighted_context | etf_source           |             0.85 | 1.30%  | 0.20% | 1.02% | 1.56% |
|      11 | theme_count              | tier_4_downweighted_context | etf_source           |             0.85 | 1.26%  | 0.19% | 1.03% | 1.46% |
|      12 | log_avg_dollar_volume_3m | tier_2_tail_context         | liquidity_size       |             1.1  | 1.15%  | 0.21% | 0.85% | 1.38% |
|      13 | large_move_freq_6m       | tier_1_core_tail            | volatility_frequency |             1.2  | 1.07%  | 0.06% | 0.98% | 1.12% |
|      14 | qqq_mom_6m               | tier_1_core_tail            | core_momentum        |             1.6  | 1.06%  | 0.03% | 1.03% | 1.08% |
|      15 | ma100_slope_1m           | tier_3_standard             | trend                |             1    | 1.06%  | 0.04% | 1.00% | 1.09% |
|      16 | qqq_mom_1m               | tier_2_tail_context         | other_momentum       |             1.1  | 1.04%  | 0.06% | 0.97% | 1.13% |
|      17 | avg_dollar_volume_3m     | tier_2_tail_context         | liquidity_size       |             1.1  | 1.03%  | 0.06% | 0.94% | 1.09% |
|      18 | in_large_cap_core        | tier_4_downweighted_context | etf_source           |             0.85 | 1.03%  | 0.20% | 0.70% | 1.23% |
|      19 | avg_dollar_volume_6m     | tier_2_tail_context         | liquidity_size       |             1.1  | 0.96%  | 0.05% | 0.90% | 1.04% |
|      20 | drawdown_3m_abs          | tier_4_downweighted_context | risk_drawdown        |             0.95 | 0.92%  | 0.05% | 0.85% | 0.98% |
|      21 | rel_mom_6m_vs_qqq        | tier_1_core_tail            | core_momentum        |             1.6  | 0.88%  | 0.06% | 0.79% | 0.92% |
|      22 | avg_dollar_volume_1m     | tier_2_tail_context         | liquidity_size       |             1.1  | 0.86%  | 0.03% | 0.81% | 0.89% |
|      23 | drawdown_12m             | tier_4_downweighted_context | risk_drawdown        |             0.95 | 0.85%  | 0.05% | 0.81% | 0.94% |
|      24 | drawdown_12m_abs         | tier_4_downweighted_context | risk_drawdown        |             0.95 | 0.85%  | 0.04% | 0.79% | 0.89% |
|      25 | qqq_mom_12m              | tier_2_tail_context         | other_momentum       |             1.1  | 0.84%  | 0.02% | 0.83% | 0.86% |
|      26 | volatility_3m_to_6m      | tier_4_downweighted_context | risk_drawdown        |             0.95 | 0.82%  | 0.03% | 0.80% | 0.87% |
|      27 | drawdown_6m              | tier_4_downweighted_context | risk_drawdown        |             0.95 | 0.82%  | 0.07% | 0.72% | 0.89% |
|      28 | drawdown_3m              | tier_4_downweighted_context | risk_drawdown        |             0.95 | 0.80%  | 0.08% | 0.74% | 0.94% |
|      29 | mom_6m_first3m           | tier_1_core_tail            | core_momentum        |             1.6  | 0.80%  | 0.04% | 0.75% | 0.84% |
|      30 | qqq_mom_3m               | tier_2_tail_context         | other_momentum       |             1.1  | 0.78%  | 0.02% | 0.75% | 0.81% |
|      31 | ret_lag_6m_std           | tier_3_standard             | unclassified         |             1    | 0.77%  | 0.05% | 0.72% | 0.83% |
|      32 | mom_6m                   | tier_1_core_tail            | core_momentum        |             1.6  | 0.76%  | 0.05% | 0.72% | 0.81% |
|      33 | core_mom_456_min         | tier_1_core_tail            | core_momentum        |             1.6  | 0.76%  | 0.03% | 0.72% | 0.81% |
|      34 | drawdown_6m_abs          | tier_4_downweighted_context | risk_drawdown        |             0.95 | 0.75%  | 0.09% | 0.68% | 0.89% |
|      35 | down_big_move_freq_6m    | tier_1_core_tail            | volatility_frequency |             1.2  | 0.75%  | 0.05% | 0.70% | 0.80% |
|      36 | ret_lag_6m_mean          | tier_3_standard             | unclassified         |             1    | 0.75%  | 0.07% | 0.67% | 0.84% |
|      37 | ret_lag_6m_max           | tier_3_standard             | unclassified         |             1    | 0.74%  | 0.02% | 0.72% | 0.76% |
|      38 | drawdown_1m_abs          | tier_4_downweighted_context | risk_drawdown        |             0.95 | 0.73%  | 0.04% | 0.69% | 0.78% |
|      39 | mom_9m                   | tier_2_tail_context         | other_momentum       |             1.1  | 0.73%  | 0.04% | 0.69% | 0.79% |
|      40 | days_since_3m_low_norm   | tier_3_standard             | unclassified         |             1    | 0.73%  | 0.04% | 0.69% | 0.78% |
|      41 | mom_3m_vs_6m             | tier_2_tail_context         | other_momentum       |             1.1  | 0.73%  | 0.04% | 0.69% | 0.77% |
|      42 | mom_4m                   | tier_1_core_tail            | core_momentum        |             1.6  | 0.73%  | 0.05% | 0.68% | 0.81% |
|      43 | core_mom_456_avg         | tier_1_core_tail            | core_momentum        |             1.6  | 0.73%  | 0.09% | 0.61% | 0.83% |
|      44 | ma30_slope_1m            | tier_3_standard             | trend                |             1    | 0.72%  | 0.02% | 0.71% | 0.75% |
|      45 | volatility_1m_to_6m      | tier_4_downweighted_context | risk_drawdown        |             0.95 | 0.70%  | 0.04% | 0.65% | 0.76% |
|      46 | volatility_1m            | tier_4_downweighted_context | risk_drawdown        |             0.95 | 0.70%  | 0.02% | 0.67% | 0.73% |
|      47 | mom_5m                   | tier_1_core_tail            | core_momentum        |             1.6  | 0.70%  | 0.04% | 0.65% | 0.74% |
|      48 | mom_12m                  | tier_2_tail_context         | other_momentum       |             1.1  | 0.69%  | 0.03% | 0.65% | 0.72% |
|      49 | atr_14_to_100d           | tier_3_standard             | unclassified         |             1    | 0.68%  | 0.05% | 0.62% | 0.75% |
|      50 | drawdown_1m              | tier_4_downweighted_context | risk_drawdown        |             0.95 | 0.67%  | 0.07% | 0.60% | 0.78% |

## Manual feature weights used by XGBoost

|   index | feature                       | tier                        | feature_group        |   feature_weight |
|--------:|:------------------------------|:----------------------------|:---------------------|-----------------:|
|       1 | rel_mom_12m_vs_qqq            | tier_2_tail_context         | other_momentum       |             1.1  |
|       2 | mom_12m                       | tier_2_tail_context         | other_momentum       |             1.1  |
|       3 | up_day_volume_ratio_3m        | tier_4_downweighted_context | volume_flow          |             0.95 |
|       4 | up_day_dollar_volume_ratio_3m | tier_2_tail_context         | liquidity_size       |             1.1  |
|       5 | mom_9m                        | tier_2_tail_context         | other_momentum       |             1.1  |
|       6 | mom_7m                        | tier_2_tail_context         | other_momentum       |             1.1  |
|       7 | mom_6m                        | tier_1_core_tail            | core_momentum        |             1.6  |
|       8 | mom_4m_vs_6m                  | tier_1_core_tail            | core_momentum        |             1.6  |
|       9 | mom_5m_vs_6m                  | tier_1_core_tail            | core_momentum        |             1.6  |
|      10 | mom_6m_first3m                | tier_1_core_tail            | core_momentum        |             1.6  |
|      11 | mom_6m_acceleration           | tier_1_core_tail            | core_momentum        |             1.6  |
|      12 | ret_lag_6m                    | tier_3_standard             | unclassified         |             1    |
|      13 | rel_mom_6m_vs_qqq             | tier_1_core_tail            | core_momentum        |             1.6  |
|      14 | mom_3m_vs_6m                  | tier_2_tail_context         | other_momentum       |             1.1  |
|      15 | return_vol_ratio_6m           | tier_4_downweighted_context | risk_drawdown        |             0.95 |
|      16 | drawdown_12m                  | tier_4_downweighted_context | risk_drawdown        |             0.95 |
|      17 | drawdown_12m_abs              | tier_4_downweighted_context | risk_drawdown        |             0.95 |
|      18 | ma100_slope_1m                | tier_3_standard             | trend                |             1    |
|      19 | mom_5m                        | tier_1_core_tail            | core_momentum        |             1.6  |
|      20 | core_mom_456_std              | tier_1_core_tail            | core_momentum        |             1.6  |
|      21 | ret_lag_5m                    | tier_3_standard             | unclassified         |             1    |
|      22 | volume_ma3_to_12m             | tier_4_downweighted_context | volume_flow          |             0.95 |
|      23 | price_ma100_ratio             | tier_3_standard             | trend                |             1    |
|      24 | volume_change_3m              | tier_4_downweighted_context | volume_flow          |             0.95 |
|      25 | dollar_volume_change_3m       | tier_2_tail_context         | liquidity_size       |             1.1  |
|      26 | drawdown_change_3m            | tier_4_downweighted_context | risk_drawdown        |             0.95 |
|      27 | mom_4m                        | tier_1_core_tail            | core_momentum        |             1.6  |
|      28 | core_mom_456_avg              | tier_1_core_tail            | core_momentum        |             1.6  |
|      29 | core_mom_456_min              | tier_1_core_tail            | core_momentum        |             1.6  |
|      30 | core_mom_456_max              | tier_1_core_tail            | core_momentum        |             1.6  |
|      31 | ret_lag_4m                    | tier_3_standard             | unclassified         |             1    |
|      32 | ma50_slope_1m                 | tier_3_standard             | trend                |             1    |
|      33 | mom_6m_last3m                 | tier_1_core_tail            | core_momentum        |             1.6  |
|      34 | ret_lag_3m                    | tier_3_standard             | unclassified         |             1    |
|      35 | rel_mom_3m_vs_qqq             | tier_2_tail_context         | other_momentum       |             1.1  |
|      36 | mom_3m                        | tier_2_tail_context         | other_momentum       |             1.1  |
|      37 | volatility_6m                 | tier_4_downweighted_context | risk_drawdown        |             0.95 |
|      38 | return_vol_ratio_3m           | tier_4_downweighted_context | risk_drawdown        |             0.95 |
|      39 | volatility_1m_to_6m           | tier_4_downweighted_context | risk_drawdown        |             0.95 |
|      40 | volatility_3m_to_6m           | tier_4_downweighted_context | risk_drawdown        |             0.95 |
|      41 | avg_abs_daily_return_6m       | tier_1_core_tail            | volatility_frequency |             1.2  |
|      42 | drawdown_6m                   | tier_4_downweighted_context | risk_drawdown        |             0.95 |
|      43 | drawdown_6m_abs               | tier_4_downweighted_context | risk_drawdown        |             0.95 |
|      44 | recovery_from_6m_low          | tier_3_standard             | unclassified         |             1    |
|      45 | avg_dollar_volume_6m          | tier_2_tail_context         | liquidity_size       |             1.1  |
|      46 | dollar_volume_3m_to_6m        | tier_2_tail_context         | liquidity_size       |             1.1  |
|      47 | large_move_freq_6m            | tier_1_core_tail            | volatility_frequency |             1.2  |
|      48 | up_big_move_freq_6m           | tier_1_core_tail            | volatility_frequency |             1.2  |
|      49 | down_big_move_freq_6m         | tier_1_core_tail            | volatility_frequency |             1.2  |
|      50 | intraday_range_mean_6m        | tier_1_core_tail            | volatility_frequency |             1.2  |
|      51 | volume_change_1m              | tier_4_downweighted_context | volume_flow          |             0.95 |
|      52 | dollar_volume_change_1m       | tier_2_tail_context         | liquidity_size       |             1.1  |
|      53 | drawdown_change_1m            | tier_4_downweighted_context | risk_drawdown        |             0.95 |
|      54 | ma30_slope_1m                 | tier_3_standard             | trend                |             1    |
|      55 | price_ma50_ratio              | tier_3_standard             | trend                |             1    |
|      56 | atr_14_to_100d                | tier_3_standard             | unclassified         |             1    |
|      57 | volume_ma1_to_6m              | tier_4_downweighted_context | volume_flow          |             0.95 |
|      58 | ret_lag_2m                    | tier_3_standard             | unclassified         |             1    |
|      59 | ret_lag_6m_std                | tier_3_standard             | unclassified         |             1    |
|      60 | mom_2m                        | tier_2_tail_context         | other_momentum       |             1.1  |
|      61 | ma20_slope_1m                 | tier_3_standard             | trend                |             1    |
|      62 | volume_ratio_3m               | tier_4_downweighted_context | volume_flow          |             0.95 |
|      63 | volatility_3m                 | tier_4_downweighted_context | risk_drawdown        |             0.95 |
|      64 | volatility_1m_to_3m           | tier_4_downweighted_context | risk_drawdown        |             0.95 |
|      65 | avg_abs_daily_return_3m       | tier_1_core_tail            | volatility_frequency |             1.2  |
|      66 | ma10_slope_1m                 | tier_3_standard             | trend                |             1    |
|      67 | drawdown_3m                   | tier_4_downweighted_context | risk_drawdown        |             0.95 |
|      68 | drawdown_3m_abs               | tier_4_downweighted_context | risk_drawdown        |             0.95 |
|      69 | recovery_from_3m_low          | tier_3_standard             | unclassified         |             1    |
|      70 | days_since_3m_high_norm       | tier_3_standard             | unclassified         |             1    |
|      71 | days_since_3m_low_norm        | tier_3_standard             | unclassified         |             1    |
|      72 | avg_dollar_volume_3m          | tier_2_tail_context         | liquidity_size       |             1.1  |
|      73 | log_avg_dollar_volume_3m      | tier_2_tail_context         | liquidity_size       |             1.1  |
|      74 | large_move_freq_3m            | tier_1_core_tail            | volatility_frequency |             1.2  |
|      75 | up_big_move_freq_3m           | tier_1_core_tail            | volatility_frequency |             1.2  |
|      76 | down_big_move_freq_3m         | tier_1_core_tail            | volatility_frequency |             1.2  |
|      77 | intraday_range_mean_3m        | tier_1_core_tail            | volatility_frequency |             1.2  |
|      78 | price_ma30_ratio              | tier_3_standard             | trend                |             1    |
|      79 | ma5_slope_1m                  | tier_3_standard             | trend                |             1    |
|      80 | return_vol_ratio_1m           | tier_4_downweighted_context | risk_drawdown        |             0.95 |

## Output files

```text
outputs/final_train_validation_test_metrics.csv
outputs/training_curve_metrics_every_100_rounds.csv
outputs/reference_downweighted_main_model_result.csv
outputs/latest_live_boom_candidates.csv
outputs/strategy_baseline_comparison.csv
outputs/feature_weight_ablation_summary.csv
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
