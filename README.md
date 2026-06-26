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

| dataset    |   rows |   months |   positive_rows | positive_rate   |   prauc |    auc | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   | top3_avg_future_return_1m   | top3_avg_future_max_return_1_3m   |
|:-----------|-------:|---------:|----------------:|:----------------|--------:|-------:|:--------------------|:------------------|:------------------|:------------------------------|:----------------------------|:----------------------------------|
| train      |  28791 |       72 |            1228 | 4.27%           |  0.846  | 0.9926 | 89.35%              | 89.35%            | 52.31%            | 84.72%                        | 19.41%                      | 60.01%                            |
| validation |   9964 |       24 |             529 | 5.31%           |  0.203  | 0.8029 | 34.72%              | 36.11%            | 13.89%            | 37.50%                        | 1.27%                       | 24.17%                            |
| test       |  12636 |       30 |             751 | 5.94%           |  0.2469 | 0.8261 | 35.56%              | 35.56%            | 28.89%            | 60.00%                        | 8.61%                       | 34.52%                            |

## Training curve metrics every 100 rounds

This table evaluates the same final model at each 100-tree checkpoint on train, validation, and test. It is meant to show whether test performance is stable or only appears near the final number of trees.

|   boosting_round | dataset    |   prauc |    auc | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   | top3_avg_future_return_1m   | top3_avg_future_max_return_1_3m   |
|-----------------:|:-----------|--------:|-------:|:--------------------|:------------------|:------------------|:------------------------------|:----------------------------|:----------------------------------|
|              100 | train      |  0.3089 | 0.8927 | 45.37%              | 45.37%            | 30.09%            | 63.89%                        | 9.26%                       | 37.27%                            |
|              100 | validation |  0.1943 | 0.805  | 34.72%              | 36.11%            | 20.83%            | 41.67%                        | 3.10%                       | 29.12%                            |
|              100 | test       |  0.2774 | 0.831  | 36.67%              | 36.67%            | 31.11%            | 56.67%                        | 7.90%                       | 33.55%                            |
|              200 | train      |  0.3365 | 0.9062 | 46.76%              | 46.76%            | 32.87%            | 66.67%                        | 9.99%                       | 39.72%                            |
|              200 | validation |  0.1989 | 0.8102 | 34.72%              | 36.11%            | 19.44%            | 45.83%                        | 5.07%                       | 31.19%                            |
|              200 | test       |  0.2772 | 0.8317 | 38.89%              | 38.89%            | 30.00%            | 56.67%                        | 7.20%                       | 37.02%                            |
|              300 | train      |  0.3722 | 0.9177 | 51.39%              | 51.39%            | 37.50%            | 76.39%                        | 10.78%                      | 41.75%                            |
|              300 | validation |  0.2078 | 0.8123 | 33.33%              | 34.72%            | 18.06%            | 41.67%                        | 4.11%                       | 28.26%                            |
|              300 | test       |  0.273  | 0.8312 | 42.22%              | 42.22%            | 28.89%            | 60.00%                        | 9.70%                       | 37.49%                            |
|              400 | train      |  0.3998 | 0.9282 | 54.17%              | 54.17%            | 37.96%            | 75.00%                        | 11.85%                      | 43.90%                            |
|              400 | validation |  0.2094 | 0.8149 | 30.56%              | 31.94%            | 13.89%            | 41.67%                        | 3.18%                       | 23.51%                            |
|              400 | test       |  0.2699 | 0.833  | 40.00%              | 40.00%            | 26.67%            | 56.67%                        | 10.71%                      | 37.16%                            |
|              500 | train      |  0.4324 | 0.9377 | 57.41%              | 57.41%            | 41.20%            | 79.17%                        | 13.02%                      | 46.95%                            |
|              500 | validation |  0.2098 | 0.8157 | 37.50%              | 38.89%            | 16.67%            | 45.83%                        | 4.72%                       | 27.25%                            |
|              500 | test       |  0.2691 | 0.8332 | 34.44%              | 34.44%            | 24.44%            | 53.33%                        | 9.42%                       | 35.27%                            |
|              600 | train      |  0.4599 | 0.9453 | 57.87%              | 57.87%            | 40.74%            | 76.39%                        | 12.40%                      | 45.41%                            |
|              600 | validation |  0.212  | 0.8159 | 37.50%              | 38.89%            | 16.67%            | 45.83%                        | 4.78%                       | 27.10%                            |
|              600 | test       |  0.2688 | 0.8332 | 37.78%              | 37.78%            | 27.78%            | 53.33%                        | 9.74%                       | 35.83%                            |
|              700 | train      |  0.4935 | 0.9526 | 60.65%              | 60.65%            | 41.20%            | 77.78%                        | 13.64%                      | 47.54%                            |
|              700 | validation |  0.2132 | 0.8152 | 37.50%              | 38.89%            | 19.44%            | 50.00%                        | 3.68%                       | 29.29%                            |
|              700 | test       |  0.2669 | 0.8334 | 35.56%              | 35.56%            | 25.56%            | 53.33%                        | 7.98%                       | 31.44%                            |
|              800 | train      |  0.5232 | 0.9585 | 62.96%              | 62.96%            | 42.13%            | 76.39%                        | 14.22%                      | 48.23%                            |
|              800 | validation |  0.2124 | 0.8144 | 37.50%              | 38.89%            | 18.06%            | 45.83%                        | 4.01%                       | 29.10%                            |
|              800 | test       |  0.2658 | 0.8334 | 36.67%              | 36.67%            | 27.78%            | 56.67%                        | 7.83%                       | 32.91%                            |
|              900 | train      |  0.5599 | 0.9639 | 66.20%              | 66.20%            | 43.98%            | 77.78%                        | 14.49%                      | 49.14%                            |
|              900 | validation |  0.2093 | 0.8127 | 38.89%              | 40.28%            | 18.06%            | 50.00%                        | 2.64%                       | 28.60%                            |
|              900 | test       |  0.2619 | 0.833  | 31.11%              | 31.11%            | 24.44%            | 53.33%                        | 7.20%                       | 29.70%                            |
|             1000 | train      |  0.5893 | 0.9684 | 67.13%              | 67.13%            | 44.91%            | 79.17%                        | 15.53%                      | 50.03%                            |
|             1000 | validation |  0.2093 | 0.8119 | 36.11%              | 37.50%            | 15.28%            | 41.67%                        | 0.60%                       | 24.23%                            |
|             1000 | test       |  0.2593 | 0.8317 | 32.22%              | 32.22%            | 25.56%            | 53.33%                        | 8.26%                       | 31.09%                            |
|             1100 | train      |  0.6231 | 0.9726 | 71.30%              | 71.30%            | 46.76%            | 81.94%                        | 15.97%                      | 53.16%                            |
|             1100 | validation |  0.2086 | 0.81   | 37.50%              | 38.89%            | 16.67%            | 45.83%                        | 1.34%                       | 25.30%                            |
|             1100 | test       |  0.2581 | 0.8313 | 28.89%              | 28.89%            | 24.44%            | 53.33%                        | 7.31%                       | 29.22%                            |
|             1200 | train      |  0.6551 | 0.9762 | 76.39%              | 76.39%            | 49.54%            | 84.72%                        | 16.86%                      | 55.68%                            |
|             1200 | validation |  0.2074 | 0.8092 | 37.50%              | 38.89%            | 18.06%            | 50.00%                        | 1.47%                       | 26.52%                            |
|             1200 | test       |  0.2575 | 0.831  | 31.11%              | 31.11%            | 26.67%            | 60.00%                        | 7.42%                       | 30.58%                            |
|             1300 | train      |  0.6838 | 0.9793 | 75.93%              | 75.93%            | 50.00%            | 84.72%                        | 16.75%                      | 54.58%                            |
|             1300 | validation |  0.2064 | 0.809  | 37.50%              | 38.89%            | 16.67%            | 45.83%                        | 2.16%                       | 25.50%                            |
|             1300 | test       |  0.2565 | 0.8302 | 30.00%              | 30.00%            | 26.67%            | 56.67%                        | 7.45%                       | 31.78%                            |
|             1400 | train      |  0.7114 | 0.982  | 75.93%              | 75.93%            | 50.93%            | 84.72%                        | 17.10%                      | 55.01%                            |
|             1400 | validation |  0.2056 | 0.8081 | 36.11%              | 37.50%            | 15.28%            | 41.67%                        | 1.46%                       | 24.38%                            |
|             1400 | test       |  0.2542 | 0.8299 | 30.00%              | 30.00%            | 25.56%            | 56.67%                        | 6.86%                       | 30.86%                            |
|             1500 | train      |  0.7366 | 0.9844 | 79.17%              | 79.17%            | 50.93%            | 84.72%                        | 17.83%                      | 56.93%                            |
|             1500 | validation |  0.2077 | 0.8074 | 33.33%              | 34.72%            | 15.28%            | 41.67%                        | 1.87%                       | 24.79%                            |
|             1500 | test       |  0.2526 | 0.8292 | 30.00%              | 30.00%            | 27.78%            | 56.67%                        | 7.23%                       | 31.20%                            |
|             1600 | train      |  0.7609 | 0.9865 | 80.56%              | 80.56%            | 50.46%            | 84.72%                        | 17.81%                      | 56.07%                            |
|             1600 | validation |  0.2073 | 0.8069 | 33.33%              | 34.72%            | 13.89%            | 41.67%                        | 1.15%                       | 23.50%                            |
|             1600 | test       |  0.2517 | 0.8285 | 31.11%              | 31.11%            | 26.67%            | 56.67%                        | 7.42%                       | 31.38%                            |
|             1700 | train      |  0.7832 | 0.9882 | 81.94%              | 81.94%            | 50.93%            | 84.72%                        | 18.27%                      | 56.79%                            |
|             1700 | validation |  0.2065 | 0.8057 | 34.72%              | 36.11%            | 15.28%            | 37.50%                        | 0.85%                       | 24.53%                            |
|             1700 | test       |  0.2495 | 0.8279 | 31.11%              | 31.11%            | 27.78%            | 60.00%                        | 7.79%                       | 32.26%                            |
|             1800 | train      |  0.8063 | 0.9899 | 86.11%              | 86.11%            | 51.85%            | 84.72%                        | 18.73%                      | 58.50%                            |
|             1800 | validation |  0.2045 | 0.8045 | 33.33%              | 34.72%            | 13.89%            | 37.50%                        | 0.23%                       | 22.96%                            |
|             1800 | test       |  0.2482 | 0.8271 | 31.11%              | 31.11%            | 27.78%            | 60.00%                        | 8.17%                       | 32.64%                            |
|             1900 | train      |  0.8259 | 0.9913 | 87.50%              | 87.50%            | 51.85%            | 84.72%                        | 19.05%                      | 58.66%                            |
|             1900 | validation |  0.2041 | 0.8042 | 33.33%              | 34.72%            | 13.89%            | 37.50%                        | 0.68%                       | 23.55%                            |
|             1900 | test       |  0.248  | 0.8268 | 32.22%              | 32.22%            | 27.78%            | 60.00%                        | 8.53%                       | 32.72%                            |
|             2000 | train      |  0.846  | 0.9926 | 89.35%              | 89.35%            | 52.31%            | 84.72%                        | 19.41%                      | 60.01%                            |
|             2000 | validation |  0.203  | 0.8029 | 34.72%              | 36.11%            | 13.89%            | 37.50%                        | 1.27%                       | 24.17%                            |
|             2000 | test       |  0.2469 | 0.8261 | 35.56%              | 35.56%            | 28.89%            | 60.00%                        | 8.61%                       | 34.52%                            |

## Reference-downweighted main model result

The main model is an XGBoost classifier with reference-downweighted sample weights. Easy/reference negatives are downweighted so they do not dominate the right-tail learning objective; stronger boom labels receive extra weight.

| model                                 | target                  |   train_rows |   valid_rows |   test_rows |   feature_count |   prauc |    auc | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   | top3_avg_future_return_1m   | top3_avg_future_max_return_1_3m   |
|:--------------------------------------|:------------------------|-------------:|-------------:|------------:|----------------:|--------:|-------:|:--------------------|:------------------|:------------------|:------------------------------|:----------------------------|:----------------------------------|
| reference_downweighted_xgb_classifier | label_boom30_top10_1_3m |        28791 |         9964 |       12636 |             110 |  0.2469 | 0.8261 | 35.56%              | 35.56%            | 28.89%            | 60.00%                        | 8.61%                       | 34.52%                            |

## Latest live boom candidates

Latest month candidates are ranked by an ensemble of the main model and five-seed average score.

|   rank | month      | ticker   | ensemble_score   | xgb_boom_probability   | five_seed_avg_score   | five_seed_score_std   |   mom_6m |   mom_3m |   rel_mom_6m_vs_qqq | liquid_vol_score   |   avg_dollar_volume_3m |
|-------:|:-----------|:---------|:-----------------|:-----------------------|:----------------------|:----------------------|---------:|---------:|--------------------:|:-------------------|-----------------------:|
|      1 | 2026-06-30 | FDXF     | 98.29%           | 98.12%                 | 98.45%                | 0.30%                 |          |          |                     | 0.00%              |                        |
|      2 | 2026-06-30 | RKLB     | 96.50%           | 97.07%                 | 95.93%                | 1.10%                 |   0.2321 |   0.3384 |              0.0675 | 97.61%             |            2.72136e+09 |
|      3 | 2026-06-30 | AMD      | 95.38%           | 95.49%                 | 95.28%                | 0.75%                 |   1.4286 |   1.5567 |              1.264  | 96.48%             |            1.42445e+10 |
|      4 | 2026-06-30 | ECHO     | 94.20%           | 94.27%                 | 94.12%                | 0.55%                 |          |          |                     | 0.00%              |                        |
|      5 | 2026-06-30 | MSTR     | 93.26%           | 93.72%                 | 92.79%                | 0.87%                 |  -0.4325 |  -0.309  |             -0.597  | 95.95%             |            2.76479e+09 |
|      6 | 2026-06-30 | TER      | 93.11%           | 92.12%                 | 94.10%                | 1.39%                 |   1.2491 |   0.4678 |              1.0845 | 93.36%             |            1.48931e+09 |
|      7 | 2026-06-30 | ASML     | 92.00%           | 92.49%                 | 91.51%                | 1.35%                 |   0.6845 |   0.3626 |              0.5199 | 89.93%             |            2.94521e+09 |
|      8 | 2026-06-30 | ALAB     | 91.84%           | 90.76%                 | 92.91%                | 1.27%                 |   1.2866 |   2.4708 |              1.122  | 95.19%             |            1.56882e+09 |
|      9 | 2026-06-30 | MU       | 91.30%           | 91.97%                 | 90.64%                | 1.07%                 |   3.1029 |   2.4647 |              2.9383 | 99.28%             |            3.78632e+10 |
|     10 | 2026-06-30 | SMCI     | 90.14%           | 91.37%                 | 88.91%                | 3.31%                 |   0.0632 |   0.3667 |             -0.1014 | 94.92%             |            1.56925e+09 |
|     11 | 2026-06-30 | JBL      | 89.34%           | 89.45%                 | 89.24%                | 0.76%                 |   0.5921 |   0.3663 |              0.4276 | 70.97%             |            4.26761e+08 |
|     12 | 2026-06-30 | PLUG     | 87.72%           | 87.03%                 | 88.41%                | 3.29%                 |   0.3046 |   0.1372 |              0.14   | 73.60%             |            2.3584e+08  |
|     13 | 2026-06-30 | DELL     | 87.42%           | 86.99%                 | 87.85%                | 3.50%                 |   2.1088 |   1.3739 |              1.9442 | 93.96%             |            2.65632e+09 |
|     14 | 2026-06-30 | CRWD     | 87.26%           | 87.75%                 | 86.77%                | 2.22%                 |   0.4992 |   0.8001 |              0.3346 | 88.59%             |            1.87811e+09 |
|     15 | 2026-06-30 | DDOG     | 87.12%           | 88.09%                 | 86.14%                | 3.58%                 |   0.7239 |   0.9859 |              0.5593 | 89.61%             |            1.05542e+09 |
|     16 | 2026-06-30 | ANET     | 86.33%           | 87.88%                 | 84.78%                | 3.45%                 |   0.2143 |   0.2959 |              0.0497 | 90.06%             |            1.4192e+09  |
|     17 | 2026-06-30 | ENPH     | 85.44%           | 86.59%                 | 84.28%                | 2.24%                 |   0.4839 |   0.2579 |              0.3194 | 79.88%             |            3.6414e+08  |
|     18 | 2026-06-30 | GNRC     | 84.70%           | 85.36%                 | 84.05%                | 1.82%                 |   1.0596 |   0.4379 |              0.8951 | 66.42%             |            2.11693e+08 |
|     19 | 2026-06-30 | MRVL     | 84.69%           | 85.11%                 | 84.28%                | 2.45%                 |   2.1581 |   1.7076 |              1.9936 | 96.95%             |            9.42458e+09 |
|     20 | 2026-06-30 | FIX      | 84.50%           | 82.86%                 | 86.13%                | 2.44%                 |   1.0698 |   0.4001 |              0.9053 | 85.07%             |            7.67792e+08 |
|     21 | 2026-06-30 | CDNS     | 82.47%           | 82.72%                 | 82.22%                | 1.93%                 |   0.1946 |   0.3439 |              0.0301 | 76.82%             |            8.43571e+08 |
|     22 | 2026-06-30 | COIN     | 81.79%           | 82.07%                 | 81.52%                | 1.32%                 |  -0.3433 |  -0.1495 |             -0.5079 | 93.02%             |            1.72581e+09 |
|     23 | 2026-06-30 | LRCX     | 81.76%           | 81.73%                 | 81.78%                | 2.27%                 |   1.2284 |   0.7832 |              1.0638 | 94.74%             |            3.06915e+09 |
|     24 | 2026-06-30 | MPWR     | 81.20%           | 78.11%                 | 84.30%                | 3.55%                 |   0.479  |   0.2236 |              0.3144 | 89.11%             |            1.08669e+09 |
|     25 | 2026-06-30 | CPAY     | 79.63%           | 79.03%                 | 80.24%                | 1.21%                 |   0.1025 |   0.1401 |             -0.0621 | 52.99%             |            1.92791e+08 |
|     26 | 2026-06-30 | AXON     | 79.17%           | 79.75%                 | 78.59%                | 2.97%                 |  -0.1666 |   0.1145 |             -0.3311 | 80.96%             |            4.80346e+08 |
|     27 | 2026-06-30 | CIEN     | 78.97%           | 76.80%                 | 81.14%                | 2.94%                 |   1.0136 |   0.213  |              0.8491 | 93.10%             |            1.37409e+09 |
|     28 | 2026-06-30 | SEDG     | 78.70%           | 78.61%                 | 78.79%                | 6.41%                 |   0.8003 |   0.0174 |              0.6358 | 71.35%             |            2.00998e+08 |
|     29 | 2026-06-30 | CEG      | 77.82%           | 82.38%                 | 73.27%                | 5.18%                 |  -0.2485 |  -0.0505 |             -0.413  | 83.61%             |            9.76678e+08 |
|     30 | 2026-06-30 | CRL      | 76.78%           | 80.52%                 | 73.04%                | 6.53%                 |   0.0613 |   0.2273 |             -0.1033 | 55.64%             |            1.50726e+08 |

## Strategy and baseline comparison

`xgb_boom_probability` is computed on model prediction rows. `baseline_*` strategies are computed independently on the full clean test panel, requiring only the baseline score column and future-return labels. This keeps baseline returns fixed when model feature sets change.

| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:-----------------------------|:--------------------|
| xgb_boom_probability         |       29 | 523.51%                      | 113.26%                           | 8.61%                   | 34.52%                       | 36.78%              |
| baseline_mom_3m              |       29 | 376.41%                      | 90.78%                            | 6.93%                   | 30.64%                       | 43.68%              |
| baseline_mom_5m              |       29 | 1326.78%                     | 200.38%                           | 11.31%                  | 38.23%                       | 50.57%              |
| baseline_mom_6m              |       29 | 818.88%                      | 150.37%                           | 9.47%                   | 34.78%                       | 47.13%              |
| baseline_rel_mom_6m_vs_qqq   |       29 | 818.88%                      | 150.37%                           | 9.47%                   | 34.78%                       | 47.13%              |
| baseline_core_mom_456_avg    |       29 | 1147.80%                     | 184.17%                           | 10.50%                  | 37.83%                       | 50.57%              |
| baseline_return_vol_ratio_6m |       29 | 104.61%                      | 34.48%                            | 3.08%                   | 19.16%                       | 24.14%              |
| baseline_mom_6m_acceleration |       29 | 493.67%                      | 108.97%                           | 7.95%                   | 28.60%                       | 35.63%              |
| baseline_mom_4m              |       29 | 1310.64%                     | 198.96%                           | 10.96%                  | 38.51%                       | 50.57%              |

## Recent XGB Top-3 backtest months

| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-07-31 | SEDG, ENPH, LEU    | 94.36%      | 13.99%      | 43.76%                   | 66.67%          |
| 2025-08-31 | UUUU, SEDG, RKLB   | 96.97%      | 13.59%      | 38.80%                   | 33.33%          |
| 2025-09-30 | LEU, SEDG, SMCI    | 97.30%      | 7.24%       | 8.54%                    | 0.00%           |
| 2025-10-31 | LEU, SEDG, TSLA    | 96.45%      | -10.37%     | -7.22%                   | 0.00%           |
| 2025-11-30 | ALAB, RKLB, LEU    | 97.37%      | 21.58%      | 34.30%                   | 33.33%          |
| 2025-12-31 | LEU, SEDG, PLUG    | 97.92%      | 9.84%       | 35.43%                   | 33.33%          |
| 2026-01-31 | RKLB, SEDG, MRVL   | 97.32%      | 1.40%       | 59.12%                   | 66.67%          |
| 2026-02-28 | RKLB, LEU, MSTR    | 97.62%      | -8.34%      | 46.51%                   | 33.33%          |
| 2026-03-31 | LEU, ALAB, SEDG    | 97.41%      | 27.72%      | 106.05%                  | 66.67%          |
| 2026-04-30 | ALAB, SEDG, ENPH   | 96.36%      | 87.20%      | 93.63%                   | 100.00%         |
| 2026-05-31 | ENPH, ALAB, SEDG   | 95.92%      | -17.15%     | -17.15%                  | 0.00%           |
| 2026-06-30 | FDXF, RKLB, AMD    | 96.89%      |             |                          | 0.00%           |

## Ablation ranked summary

Each row drops one feature group and retrains the model. Negative `delta_top3_future_max_vs_main` means the removed group was useful for Top-3 tail capture.

| ablation                  |   dropped_features |   kept_features | top3_avg_future_max_return_1_3m   | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   | delta_top3_future_max_vs_main   |
|:--------------------------|-------------------:|----------------:|:----------------------------------|:--------------------|:------------------|:------------------|:------------------------------|:--------------------------------|
| drop_volatility_frequency |                 11 |              99 | 24.25%                            | 26.67%              | 27.78%            | 21.11%            | 46.67%                        | -10.26%                         |
| drop_liquidity_size       |                 11 |              99 | 26.22%                            | 27.78%              | 30.00%            | 23.33%            | 53.33%                        | -8.30%                          |
| drop_qqq_context          |                  4 |             106 | 27.06%                            | 28.89%              | 28.89%            | 23.33%            | 43.33%                        | -7.46%                          |
| drop_trend                |                 12 |              98 | 27.72%                            | 28.89%              | 28.89%            | 25.56%            | 46.67%                        | -6.80%                          |
| drop_etf_source           |                  6 |             104 | 27.99%                            | 30.00%              | 30.00%            | 26.67%            | 53.33%                        | -6.53%                          |
| drop_other_momentum       |                 13 |              97 | 29.31%                            | 32.22%              | 33.33%            | 28.89%            | 56.67%                        | -5.21%                          |
| drop_volume_flow          |                  9 |             101 | 31.49%                            | 34.44%              | 34.44%            | 28.89%            | 50.00%                        | -3.02%                          |
| drop_relative_strength    |                  4 |             106 | 31.82%                            | 33.33%              | 34.44%            | 26.67%            | 56.67%                        | -2.70%                          |
| drop_risk_drawdown        |                 19 |              91 | 32.40%                            | 35.56%              | 35.56%            | 28.89%            | 53.33%                        | -2.12%                          |
| drop_core_momentum        |                 14 |              96 | 34.51%                            | 33.33%              | 34.44%            | 28.89%            | 53.33%                        | -0.01%                          |

## Five-seed training stability

This retrains the same main model with five seeds and checks whether Top-K performance is stable.

|   seed |   prauc |    auc | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   | top3_avg_future_return_1m   | top3_avg_future_max_return_1_3m   |
|-------:|--------:|-------:|:--------------------|:------------------|:------------------|:------------------------------|:----------------------------|:----------------------------------|
|      7 |  0.2439 | 0.8254 | 32.22%              | 32.22%            | 26.67%            | 50.00%                        | 6.89%                       | 30.21%                            |
|     42 |  0.2469 | 0.8261 | 35.56%              | 35.56%            | 28.89%            | 60.00%                        | 8.61%                       | 34.52%                            |
|    202 |  0.2461 | 0.826  | 33.33%              | 34.44%            | 28.89%            | 53.33%                        | 5.50%                       | 29.77%                            |
|    777 |  0.2458 | 0.8264 | 31.11%              | 31.11%            | 27.78%            | 53.33%                        | 7.54%                       | 29.97%                            |
|   2026 |  0.2448 | 0.8277 | 28.89%              | 28.89%            | 25.56%            | 50.00%                        | 4.89%                       | 28.22%                            |

## Five-seed average feature importance

Feature importance is averaged across the five seed models. `feature_weight` is the manual XGBoost feature weight used during training, not a learned importance.

|   index | feature                  | tier                        | feature_group        |   feature_weight | mean   | std   | min   | max   |
|--------:|:-------------------------|:----------------------------|:---------------------|-----------------:|:-------|:------|:------|:------|
|       1 | intraday_range_mean_3m   | tier_2_tail_context         | volatility_frequency |             1.15 | 8.25%  | 0.36% | 7.78% | 8.75% |
|       2 | volatility_6m            | tier_3_standard             | risk_drawdown        |             1    | 7.17%  | 0.41% | 6.74% | 7.60% |
|       3 | intraday_range_mean_6m   | tier_2_tail_context         | volatility_frequency |             1.15 | 4.07%  | 0.34% | 3.59% | 4.47% |
|       4 | atr_14_to_price          | tier_3_standard             | unclassified         |             1    | 2.47%  | 0.14% | 2.30% | 2.65% |
|       5 | in_manual_core           | tier_4_downweighted_context | etf_source           |             0.9  | 2.08%  | 0.06% | 1.99% | 2.17% |
|       6 | theme_count              | tier_4_downweighted_context | etf_source           |             0.9  | 1.55%  | 0.23% | 1.24% | 1.81% |
|       7 | liquid_vol_score         | tier_2_tail_context         | liquidity_size       |             1.1  | 1.44%  | 0.02% | 1.43% | 1.47% |
|       8 | volatility_3m            | tier_3_standard             | risk_drawdown        |             1    | 1.44%  | 0.32% | 1.18% | 1.98% |
|       9 | avg_abs_daily_return_6m  | tier_2_tail_context         | volatility_frequency |             1.15 | 1.35%  | 0.18% | 1.07% | 1.54% |
|      10 | source_count             | tier_4_downweighted_context | etf_source           |             0.9  | 1.33%  | 0.06% | 1.25% | 1.41% |
|      11 | source_weight_sum        | tier_4_downweighted_context | etf_source           |             0.9  | 1.19%  | 0.20% | 0.95% | 1.45% |
|      12 | log_avg_dollar_volume_3m | tier_2_tail_context         | liquidity_size       |             1.1  | 1.11%  | 0.13% | 0.91% | 1.23% |
|      13 | in_large_cap_core        | tier_4_downweighted_context | etf_source           |             0.9  | 1.09%  | 0.12% | 0.93% | 1.26% |
|      14 | large_move_freq_6m       | tier_2_tail_context         | volatility_frequency |             1.15 | 1.06%  | 0.10% | 0.92% | 1.20% |
|      15 | ma100_slope_1m           | tier_3_standard             | trend                |             1    | 1.04%  | 0.07% | 0.94% | 1.12% |
|      16 | qqq_mom_1m               | tier_3_standard             | other_momentum       |             1.05 | 1.03%  | 0.02% | 1.01% | 1.05% |
|      17 | qqq_mom_6m               | tier_1_core_tail            | core_momentum        |             1.25 | 1.03%  | 0.04% | 0.99% | 1.09% |
|      18 | avg_dollar_volume_3m     | tier_2_tail_context         | liquidity_size       |             1.1  | 1.01%  | 0.04% | 0.97% | 1.04% |
|      19 | avg_dollar_volume_6m     | tier_2_tail_context         | liquidity_size       |             1.1  | 0.97%  | 0.03% | 0.93% | 1.02% |
|      20 | in_core_growth           | tier_4_downweighted_context | etf_source           |             0.9  | 0.97%  | 0.36% | 0.57% | 1.43% |
|      21 | drawdown_3m_abs          | tier_3_standard             | risk_drawdown        |             1    | 0.88%  | 0.05% | 0.81% | 0.94% |
|      22 | rel_mom_6m_vs_qqq        | tier_1_core_tail            | core_momentum        |             1.25 | 0.88%  | 0.09% | 0.74% | 0.97% |
|      23 | drawdown_12m_abs         | tier_3_standard             | risk_drawdown        |             1    | 0.87%  | 0.03% | 0.84% | 0.91% |
|      24 | drawdown_12m             | tier_3_standard             | risk_drawdown        |             1    | 0.87%  | 0.05% | 0.81% | 0.92% |
|      25 | qqq_mom_12m              | tier_3_standard             | other_momentum       |             1.05 | 0.87%  | 0.02% | 0.85% | 0.89% |
|      26 | avg_dollar_volume_1m     | tier_2_tail_context         | liquidity_size       |             1.1  | 0.87%  | 0.04% | 0.84% | 0.93% |
|      27 | volatility_3m_to_6m      | tier_3_standard             | risk_drawdown        |             1    | 0.84%  | 0.04% | 0.79% | 0.89% |
|      28 | mom_3m_vs_6m             | tier_3_standard             | other_momentum       |             1.05 | 0.83%  | 0.07% | 0.75% | 0.91% |
|      29 | qqq_mom_3m               | tier_3_standard             | other_momentum       |             1.05 | 0.80%  | 0.04% | 0.77% | 0.86% |
|      30 | mom_6m_first3m           | tier_1_core_tail            | core_momentum        |             1.25 | 0.78%  | 0.06% | 0.70% | 0.85% |
|      31 | drawdown_6m_abs          | tier_3_standard             | risk_drawdown        |             1    | 0.77%  | 0.09% | 0.67% | 0.88% |
|      32 | drawdown_6m              | tier_3_standard             | risk_drawdown        |             1    | 0.76%  | 0.02% | 0.74% | 0.78% |
|      33 | ret_lag_6m_std           | tier_3_standard             | unclassified         |             1    | 0.75%  | 0.03% | 0.71% | 0.80% |
|      34 | core_mom_456_min         | tier_1_core_tail            | core_momentum        |             1.25 | 0.75%  | 0.04% | 0.70% | 0.79% |
|      35 | mom_6m                   | tier_1_core_tail            | core_momentum        |             1.25 | 0.75%  | 0.09% | 0.62% | 0.86% |
|      36 | core_mom_456_avg         | tier_1_core_tail            | core_momentum        |             1.25 | 0.75%  | 0.03% | 0.72% | 0.78% |
|      37 | down_big_move_freq_6m    | tier_2_tail_context         | volatility_frequency |             1.15 | 0.75%  | 0.04% | 0.70% | 0.79% |
|      38 | ret_lag_6m_max           | tier_3_standard             | unclassified         |             1    | 0.74%  | 0.02% | 0.72% | 0.76% |
|      39 | ret_lag_6m_mean          | tier_3_standard             | unclassified         |             1    | 0.74%  | 0.06% | 0.66% | 0.83% |
|      40 | days_since_3m_low_norm   | tier_3_standard             | unclassified         |             1    | 0.74%  | 0.01% | 0.72% | 0.75% |
|      41 | drawdown_1m_abs          | tier_3_standard             | risk_drawdown        |             1    | 0.73%  | 0.03% | 0.70% | 0.77% |
|      42 | drawdown_3m              | tier_3_standard             | risk_drawdown        |             1    | 0.72%  | 0.06% | 0.62% | 0.79% |
|      43 | mom_9m                   | tier_3_standard             | other_momentum       |             1.05 | 0.72%  | 0.03% | 0.70% | 0.75% |
|      44 | mom_4m                   | tier_1_core_tail            | core_momentum        |             1.25 | 0.71%  | 0.05% | 0.64% | 0.77% |
|      45 | mom_12m                  | tier_3_standard             | other_momentum       |             1.05 | 0.70%  | 0.02% | 0.68% | 0.72% |
|      46 | ma30_slope_1m            | tier_3_standard             | trend                |             1    | 0.70%  | 0.03% | 0.66% | 0.74% |
|      47 | core_mom_456_max         | tier_1_core_tail            | core_momentum        |             1.25 | 0.69%  | 0.07% | 0.61% | 0.76% |
|      48 | ma50_slope_1m            | tier_3_standard             | trend                |             1    | 0.69%  | 0.05% | 0.64% | 0.75% |
|      49 | volatility_1m            | tier_3_standard             | risk_drawdown        |             1    | 0.69%  | 0.02% | 0.66% | 0.72% |
|      50 | mom_5m                   | tier_1_core_tail            | core_momentum        |             1.25 | 0.68%  | 0.02% | 0.66% | 0.72% |

## Manual feature weights used by XGBoost

|   index | feature                       | tier                        | feature_group        |   feature_weight |
|--------:|:------------------------------|:----------------------------|:---------------------|-----------------:|
|       1 | rel_mom_12m_vs_qqq            | tier_3_standard             | other_momentum       |             1.05 |
|       2 | mom_12m                       | tier_3_standard             | other_momentum       |             1.05 |
|       3 | up_day_volume_ratio_3m        | tier_4_downweighted_context | volume_flow          |             0.95 |
|       4 | up_day_dollar_volume_ratio_3m | tier_2_tail_context         | liquidity_size       |             1.1  |
|       5 | mom_9m                        | tier_3_standard             | other_momentum       |             1.05 |
|       6 | mom_7m                        | tier_3_standard             | other_momentum       |             1.05 |
|       7 | mom_6m                        | tier_1_core_tail            | core_momentum        |             1.25 |
|       8 | mom_4m_vs_6m                  | tier_1_core_tail            | core_momentum        |             1.25 |
|       9 | mom_5m_vs_6m                  | tier_1_core_tail            | core_momentum        |             1.25 |
|      10 | mom_6m_first3m                | tier_1_core_tail            | core_momentum        |             1.25 |
|      11 | mom_6m_acceleration           | tier_1_core_tail            | core_momentum        |             1.25 |
|      12 | ret_lag_6m                    | tier_3_standard             | unclassified         |             1    |
|      13 | rel_mom_6m_vs_qqq             | tier_1_core_tail            | core_momentum        |             1.25 |
|      14 | mom_3m_vs_6m                  | tier_3_standard             | other_momentum       |             1.05 |
|      15 | return_vol_ratio_6m           | tier_3_standard             | risk_drawdown        |             1    |
|      16 | drawdown_12m                  | tier_3_standard             | risk_drawdown        |             1    |
|      17 | drawdown_12m_abs              | tier_3_standard             | risk_drawdown        |             1    |
|      18 | ma100_slope_1m                | tier_3_standard             | trend                |             1    |
|      19 | mom_5m                        | tier_1_core_tail            | core_momentum        |             1.25 |
|      20 | core_mom_456_std              | tier_1_core_tail            | core_momentum        |             1.25 |
|      21 | ret_lag_5m                    | tier_3_standard             | unclassified         |             1    |
|      22 | volume_ma3_to_12m             | tier_4_downweighted_context | volume_flow          |             0.95 |
|      23 | price_ma100_ratio             | tier_3_standard             | trend                |             1    |
|      24 | volume_change_3m              | tier_4_downweighted_context | volume_flow          |             0.95 |
|      25 | dollar_volume_change_3m       | tier_2_tail_context         | liquidity_size       |             1.1  |
|      26 | drawdown_change_3m            | tier_3_standard             | risk_drawdown        |             1    |
|      27 | mom_4m                        | tier_1_core_tail            | core_momentum        |             1.25 |
|      28 | core_mom_456_avg              | tier_1_core_tail            | core_momentum        |             1.25 |
|      29 | core_mom_456_min              | tier_1_core_tail            | core_momentum        |             1.25 |
|      30 | core_mom_456_max              | tier_1_core_tail            | core_momentum        |             1.25 |
|      31 | ret_lag_4m                    | tier_3_standard             | unclassified         |             1    |
|      32 | ma50_slope_1m                 | tier_3_standard             | trend                |             1    |
|      33 | mom_6m_last3m                 | tier_1_core_tail            | core_momentum        |             1.25 |
|      34 | ret_lag_3m                    | tier_3_standard             | unclassified         |             1    |
|      35 | rel_mom_3m_vs_qqq             | tier_3_standard             | other_momentum       |             1.05 |
|      36 | mom_3m                        | tier_3_standard             | other_momentum       |             1.05 |
|      37 | volatility_6m                 | tier_3_standard             | risk_drawdown        |             1    |
|      38 | return_vol_ratio_3m           | tier_3_standard             | risk_drawdown        |             1    |
|      39 | volatility_1m_to_6m           | tier_3_standard             | risk_drawdown        |             1    |
|      40 | volatility_3m_to_6m           | tier_3_standard             | risk_drawdown        |             1    |
|      41 | avg_abs_daily_return_6m       | tier_2_tail_context         | volatility_frequency |             1.15 |
|      42 | drawdown_6m                   | tier_3_standard             | risk_drawdown        |             1    |
|      43 | drawdown_6m_abs               | tier_3_standard             | risk_drawdown        |             1    |
|      44 | recovery_from_6m_low          | tier_3_standard             | unclassified         |             1    |
|      45 | avg_dollar_volume_6m          | tier_2_tail_context         | liquidity_size       |             1.1  |
|      46 | dollar_volume_3m_to_6m        | tier_2_tail_context         | liquidity_size       |             1.1  |
|      47 | large_move_freq_6m            | tier_2_tail_context         | volatility_frequency |             1.15 |
|      48 | up_big_move_freq_6m           | tier_2_tail_context         | volatility_frequency |             1.15 |
|      49 | down_big_move_freq_6m         | tier_2_tail_context         | volatility_frequency |             1.15 |
|      50 | intraday_range_mean_6m        | tier_2_tail_context         | volatility_frequency |             1.15 |
|      51 | volume_change_1m              | tier_4_downweighted_context | volume_flow          |             0.95 |
|      52 | dollar_volume_change_1m       | tier_2_tail_context         | liquidity_size       |             1.1  |
|      53 | drawdown_change_1m            | tier_3_standard             | risk_drawdown        |             1    |
|      54 | ma30_slope_1m                 | tier_3_standard             | trend                |             1    |
|      55 | price_ma50_ratio              | tier_3_standard             | trend                |             1    |
|      56 | atr_14_to_100d                | tier_3_standard             | unclassified         |             1    |
|      57 | volume_ma1_to_6m              | tier_4_downweighted_context | volume_flow          |             0.95 |
|      58 | ret_lag_2m                    | tier_3_standard             | unclassified         |             1    |
|      59 | ret_lag_6m_std                | tier_3_standard             | unclassified         |             1    |
|      60 | mom_2m                        | tier_3_standard             | other_momentum       |             1.05 |
|      61 | ma20_slope_1m                 | tier_3_standard             | trend                |             1    |
|      62 | volume_ratio_3m               | tier_4_downweighted_context | volume_flow          |             0.95 |
|      63 | volatility_3m                 | tier_3_standard             | risk_drawdown        |             1    |
|      64 | volatility_1m_to_3m           | tier_3_standard             | risk_drawdown        |             1    |
|      65 | avg_abs_daily_return_3m       | tier_2_tail_context         | volatility_frequency |             1.15 |
|      66 | ma10_slope_1m                 | tier_3_standard             | trend                |             1    |
|      67 | drawdown_3m                   | tier_3_standard             | risk_drawdown        |             1    |
|      68 | drawdown_3m_abs               | tier_3_standard             | risk_drawdown        |             1    |
|      69 | recovery_from_3m_low          | tier_3_standard             | unclassified         |             1    |
|      70 | days_since_3m_high_norm       | tier_3_standard             | unclassified         |             1    |
|      71 | days_since_3m_low_norm        | tier_3_standard             | unclassified         |             1    |
|      72 | avg_dollar_volume_3m          | tier_2_tail_context         | liquidity_size       |             1.1  |
|      73 | log_avg_dollar_volume_3m      | tier_2_tail_context         | liquidity_size       |             1.1  |
|      74 | large_move_freq_3m            | tier_2_tail_context         | volatility_frequency |             1.15 |
|      75 | up_big_move_freq_3m           | tier_2_tail_context         | volatility_frequency |             1.15 |
|      76 | down_big_move_freq_3m         | tier_2_tail_context         | volatility_frequency |             1.15 |
|      77 | intraday_range_mean_3m        | tier_2_tail_context         | volatility_frequency |             1.15 |
|      78 | price_ma30_ratio              | tier_3_standard             | trend                |             1    |
|      79 | ma5_slope_1m                  | tier_3_standard             | trend                |             1    |
|      80 | return_vol_ratio_1m           | tier_3_standard             | risk_drawdown        |             1    |

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
