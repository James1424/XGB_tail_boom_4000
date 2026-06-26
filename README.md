# Integrated ETF Tail Boom Prediction Project

This repository builds the ETF/index monthly panel and immediately trains a right-tail XGBoost boom detector in the same workflow. Large panel and prediction files are uploaded as GitHub Actions artifacts; README reports, compact CSV summaries, and the model JSON are committed to the repository.

Core target: `label_boom30_top10_1_3m`. A positive label means future 1–3 month max return is in the monthly top 10% and at least +30%.

## Model parameters

Main feature-weight profile: `core_momentum_heavy`.

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

## README archive

Every successful workflow run saves a numbered copy of the generated root `README.md` into `readme_archive/`. The archive index is stored at `readme_archive/README_ARCHIVE_INDEX.md`, with files named like `README_0001_YYYYMMDD_HHMMSS_UTC.md`.

## Leakage rule

The following columns must never be used as model inputs: `future_return_*`, `future_max_return_1_3m`, `future_max_return_1_3m_pct_rank`, monthly thresholds, and every `label_*` column. The training code automatically excludes them and only keeps numeric feature columns.

## Final train / validation / test metrics

| dataset    |   rows |   months |   positive_rows | positive_rate   |   prauc |    auc | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   | top3_avg_future_return_1m   | top3_avg_future_max_return_1_3m   |
|:-----------|-------:|---------:|----------------:|:----------------|--------:|-------:|:--------------------|:------------------|:------------------|:------------------------------|:----------------------------|:----------------------------------|
| train      |  23153 |       72 |            1042 | 4.50%           |  0.9155 | 0.996  | 93.52%              | 93.52%            | 52.31%            | 81.94%                        | 19.48%                      | 59.80%                            |
| validation |   8060 |       24 |             442 | 5.48%           |  0.2075 | 0.8017 | 37.50%              | 40.28%            | 19.44%            | 54.17%                        | 6.07%                       | 28.70%                            |
| test       |  10236 |       30 |             651 | 6.36%           |  0.2287 | 0.8143 | 37.78%              | 40.00%            | 25.56%            | 60.00%                        | 6.44%                       | 33.31%                            |

## Training curve metrics every 100 rounds

This table evaluates the same final model at each 100-tree checkpoint on train, validation, and test. It is meant to show whether test performance is stable or only appears near the final number of trees.

|   boosting_round | dataset    |   prauc |    auc | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   | top3_avg_future_return_1m   | top3_avg_future_max_return_1_3m   |
|-----------------:|:-----------|--------:|-------:|:--------------------|:------------------|:------------------|:------------------------------|:----------------------------|:----------------------------------|
|              100 | train      |  0.316  | 0.89   | 44.44%              | 44.44%            | 27.31%            | 62.50%                        | 7.12%                       | 32.68%                            |
|              100 | validation |  0.1955 | 0.805  | 34.72%              | 36.11%            | 18.06%            | 37.50%                        | 3.44%                       | 29.35%                            |
|              100 | test       |  0.2905 | 0.8329 | 41.11%              | 41.11%            | 31.11%            | 50.00%                        | 12.61%                      | 42.23%                            |
|              200 | train      |  0.3527 | 0.9059 | 47.69%              | 48.15%            | 30.09%            | 62.50%                        | 7.77%                       | 34.36%                            |
|              200 | validation |  0.2018 | 0.8083 | 34.72%              | 36.11%            | 20.83%            | 45.83%                        | 6.83%                       | 34.17%                            |
|              200 | test       |  0.2831 | 0.8336 | 42.22%              | 42.22%            | 31.11%            | 56.67%                        | 14.22%                      | 44.06%                            |
|              300 | train      |  0.3956 | 0.9192 | 50.46%              | 50.46%            | 30.56%            | 65.28%                        | 8.74%                       | 35.96%                            |
|              300 | validation |  0.2188 | 0.811  | 33.33%              | 36.11%            | 18.06%            | 45.83%                        | 7.09%                       | 30.45%                            |
|              300 | test       |  0.2714 | 0.8325 | 45.56%              | 45.56%            | 31.11%            | 56.67%                        | 14.79%                      | 44.30%                            |
|              400 | train      |  0.4302 | 0.9313 | 57.41%              | 57.41%            | 33.80%            | 69.44%                        | 10.43%                      | 39.73%                            |
|              400 | validation |  0.2232 | 0.8119 | 37.50%              | 40.28%            | 20.83%            | 50.00%                        | 6.05%                       | 31.83%                            |
|              400 | test       |  0.266  | 0.8318 | 43.33%              | 44.44%            | 30.00%            | 56.67%                        | 14.02%                      | 41.43%                            |
|              500 | train      |  0.4728 | 0.9414 | 59.72%              | 59.72%            | 37.04%            | 70.83%                        | 12.08%                      | 42.25%                            |
|              500 | validation |  0.2204 | 0.8113 | 37.50%              | 40.28%            | 19.44%            | 50.00%                        | 5.55%                       | 30.90%                            |
|              500 | test       |  0.2619 | 0.8307 | 38.89%              | 40.00%            | 27.78%            | 56.67%                        | 11.75%                      | 37.97%                            |
|              600 | train      |  0.5149 | 0.9502 | 64.35%              | 64.35%            | 38.89%            | 72.22%                        | 12.89%                      | 45.20%                            |
|              600 | validation |  0.2195 | 0.8122 | 41.67%              | 44.44%            | 20.83%            | 54.17%                        | 6.10%                       | 34.85%                            |
|              600 | test       |  0.26   | 0.8298 | 37.78%              | 38.89%            | 27.78%            | 56.67%                        | 10.15%                      | 36.85%                            |
|              700 | train      |  0.5561 | 0.958  | 68.06%              | 68.06%            | 39.81%            | 72.22%                        | 14.18%                      | 46.35%                            |
|              700 | validation |  0.2167 | 0.8112 | 40.28%              | 43.06%            | 22.22%            | 54.17%                        | 5.32%                       | 30.51%                            |
|              700 | test       |  0.2573 | 0.8291 | 35.56%              | 38.89%            | 27.78%            | 56.67%                        | 10.44%                      | 36.45%                            |
|              800 | train      |  0.5959 | 0.9645 | 70.83%              | 70.83%            | 41.67%            | 72.22%                        | 14.63%                      | 48.38%                            |
|              800 | validation |  0.2154 | 0.8105 | 36.11%              | 38.89%            | 20.83%            | 54.17%                        | 5.82%                       | 28.53%                            |
|              800 | test       |  0.2558 | 0.8281 | 37.78%              | 40.00%            | 26.67%            | 56.67%                        | 10.28%                      | 35.81%                            |
|              900 | train      |  0.6304 | 0.9698 | 73.61%              | 73.61%            | 44.44%            | 77.78%                        | 15.13%                      | 50.43%                            |
|              900 | validation |  0.215  | 0.8097 | 33.33%              | 36.11%            | 19.44%            | 50.00%                        | 4.87%                       | 28.48%                            |
|              900 | test       |  0.2534 | 0.8271 | 37.78%              | 40.00%            | 27.78%            | 56.67%                        | 10.27%                      | 36.81%                            |
|             1000 | train      |  0.6655 | 0.9747 | 75.00%              | 75.00%            | 45.37%            | 77.78%                        | 15.36%                      | 50.96%                            |
|             1000 | validation |  0.2118 | 0.8083 | 34.72%              | 37.50%            | 19.44%            | 54.17%                        | 5.00%                       | 29.66%                            |
|             1000 | test       |  0.2508 | 0.8264 | 37.78%              | 40.00%            | 27.78%            | 56.67%                        | 9.81%                       | 35.59%                            |
|             1100 | train      |  0.698  | 0.9787 | 76.85%              | 76.85%            | 46.30%            | 79.17%                        | 16.03%                      | 52.06%                            |
|             1100 | validation |  0.2112 | 0.808  | 34.72%              | 37.50%            | 18.06%            | 50.00%                        | 3.75%                       | 28.94%                            |
|             1100 | test       |  0.2475 | 0.8248 | 37.78%              | 40.00%            | 26.67%            | 56.67%                        | 8.38%                       | 33.28%                            |
|             1200 | train      |  0.7327 | 0.9822 | 79.63%              | 79.63%            | 47.22%            | 79.17%                        | 16.74%                      | 53.02%                            |
|             1200 | validation |  0.2125 | 0.8073 | 41.67%              | 43.06%            | 23.61%            | 58.33%                        | 5.65%                       | 34.18%                            |
|             1200 | test       |  0.2441 | 0.8233 | 41.11%              | 42.22%            | 28.89%            | 56.67%                        | 8.47%                       | 37.01%                            |
|             1300 | train      |  0.7615 | 0.985  | 82.87%              | 82.87%            | 49.07%            | 79.17%                        | 17.35%                      | 55.18%                            |
|             1300 | validation |  0.2114 | 0.8074 | 38.89%              | 40.28%            | 20.83%            | 58.33%                        | 5.03%                       | 32.55%                            |
|             1300 | test       |  0.2429 | 0.8223 | 40.00%              | 42.22%            | 26.67%            | 56.67%                        | 7.68%                       | 35.29%                            |
|             1400 | train      |  0.7911 | 0.9876 | 85.65%              | 85.65%            | 49.54%            | 80.56%                        | 17.68%                      | 56.36%                            |
|             1400 | validation |  0.21   | 0.8065 | 37.50%              | 40.28%            | 19.44%            | 54.17%                        | 4.87%                       | 31.48%                            |
|             1400 | test       |  0.2404 | 0.821  | 40.00%              | 42.22%            | 25.56%            | 56.67%                        | 7.26%                       | 35.16%                            |
|             1500 | train      |  0.8189 | 0.9897 | 87.96%              | 87.96%            | 49.54%            | 80.56%                        | 18.03%                      | 56.63%                            |
|             1500 | validation |  0.2095 | 0.8052 | 37.50%              | 40.28%            | 19.44%            | 54.17%                        | 5.19%                       | 31.71%                            |
|             1500 | test       |  0.2383 | 0.8201 | 38.89%              | 41.11%            | 25.56%            | 56.67%                        | 7.20%                       | 35.04%                            |
|             1600 | train      |  0.8425 | 0.9915 | 88.89%              | 88.89%            | 50.00%            | 80.56%                        | 18.30%                      | 57.42%                            |
|             1600 | validation |  0.2081 | 0.8041 | 34.72%              | 37.50%            | 19.44%            | 54.17%                        | 4.74%                       | 30.40%                            |
|             1600 | test       |  0.2374 | 0.8193 | 38.89%              | 42.22%            | 25.56%            | 56.67%                        | 8.07%                       | 35.73%                            |
|             1700 | train      |  0.8645 | 0.9929 | 90.74%              | 90.74%            | 50.46%            | 80.56%                        | 18.67%                      | 58.40%                            |
|             1700 | validation |  0.2081 | 0.8038 | 36.11%              | 38.89%            | 19.44%            | 54.17%                        | 4.52%                       | 27.58%                            |
|             1700 | test       |  0.2351 | 0.8182 | 38.89%              | 41.11%            | 26.67%            | 56.67%                        | 6.95%                       | 34.87%                            |
|             1800 | train      |  0.8849 | 0.9942 | 90.74%              | 90.74%            | 51.85%            | 80.56%                        | 19.10%                      | 58.56%                            |
|             1800 | validation |  0.2079 | 0.803  | 34.72%              | 37.50%            | 18.06%            | 54.17%                        | 4.42%                       | 26.82%                            |
|             1800 | test       |  0.2329 | 0.8171 | 37.78%              | 41.11%            | 26.67%            | 60.00%                        | 6.83%                       | 34.02%                            |
|             1900 | train      |  0.9009 | 0.9951 | 91.67%              | 91.67%            | 51.39%            | 80.56%                        | 19.24%                      | 58.63%                            |
|             1900 | validation |  0.2074 | 0.8022 | 34.72%              | 37.50%            | 19.44%            | 54.17%                        | 5.56%                       | 27.08%                            |
|             1900 | test       |  0.2309 | 0.8157 | 36.67%              | 37.78%            | 26.67%            | 60.00%                        | 7.02%                       | 34.04%                            |
|             2000 | train      |  0.9155 | 0.996  | 93.52%              | 93.52%            | 52.31%            | 81.94%                        | 19.48%                      | 59.80%                            |
|             2000 | validation |  0.2075 | 0.8017 | 37.50%              | 40.28%            | 19.44%            | 54.17%                        | 6.07%                       | 28.70%                            |
|             2000 | test       |  0.2287 | 0.8143 | 37.78%              | 40.00%            | 25.56%            | 60.00%                        | 6.44%                       | 33.31%                            |

## Reference-downweighted main model result

The main model is an XGBoost classifier with reference-downweighted sample weights. Easy/reference negatives are downweighted so they do not dominate the right-tail learning objective; stronger boom labels receive extra weight.

| model                                 | target                  |   train_rows |   valid_rows |   test_rows |   feature_count |   prauc |    auc | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   | top3_avg_future_return_1m   | top3_avg_future_max_return_1_3m   |
|:--------------------------------------|:------------------------|-------------:|-------------:|------------:|----------------:|--------:|-------:|:--------------------|:------------------|:------------------|:------------------------------|:----------------------------|:----------------------------------|
| reference_downweighted_xgb_classifier | label_boom30_top10_1_3m |        23153 |         8060 |       10236 |             110 |  0.2287 | 0.8143 | 37.78%              | 40.00%            | 25.56%            | 60.00%                        | 6.44%                       | 33.31%                            |

## Latest live boom candidates

Latest month candidates are ranked by an ensemble of the main model and five-seed average score.

|   rank | month      | ticker   | ensemble_score   | xgb_boom_probability   | five_seed_avg_score   | five_seed_score_std   |   mom_6m |   mom_3m |   rel_mom_6m_vs_qqq | liquid_vol_score   |   avg_dollar_volume_3m |
|-------:|:-----------|:---------|:-----------------|:-----------------------|:----------------------|:----------------------|---------:|---------:|--------------------:|:-------------------|-----------------------:|
|      1 | 2026-06-30 | FDXF     | 98.77%           | 98.76%                 | 98.78%                | 0.06%                 |          |          |                     | 0.00%              |                        |
|      2 | 2026-06-30 | ENPH     | 98.24%           | 98.36%                 | 98.11%                | 0.51%                 |   0.4846 |   0.2584 |              0.3317 | 79.18%             |            3.68208e+08 |
|      3 | 2026-06-30 | RKLB     | 97.65%           | 97.59%                 | 97.70%                | 0.53%                 |   0.2119 |   0.3164 |              0.0591 | 97.06%             |            2.74812e+09 |
|      4 | 2026-06-30 | AMD      | 96.98%           | 97.32%                 | 96.64%                | 0.94%                 |   1.4355 |   1.5639 |              1.2826 | 96.04%             |            1.45776e+10 |
|      5 | 2026-06-30 | ECHO     | 95.92%           | 96.22%                 | 95.61%                | 0.36%                 |          |          |                     | 0.00%              |                        |
|      6 | 2026-06-30 | ALAB     | 95.51%           | 95.82%                 | 95.19%                | 1.74%                 |   1.3548 |   2.5743 |              1.202  | 94.37%             |            1.59505e+09 |
|      7 | 2026-06-30 | MELI     | 94.12%           | 94.34%                 | 93.91%                | 0.57%                 |  -0.1684 |  -0.0312 |             -0.3212 | 70.91%             |            9.1538e+08  |
|      8 | 2026-06-30 | SHOP     | 93.79%           | 93.64%                 | 93.94%                | 0.65%                 |  -0.274  |  -0.0148 |             -0.4268 | 88.20%             |            1.1462e+09  |
|      9 | 2026-06-30 | DDOG     | 93.75%           | 93.30%                 | 94.20%                | 1.89%                 |   0.7631 |   1.0311 |              0.6103 | 88.61%             |            1.08612e+09 |
|     10 | 2026-06-30 | UUUU     | 93.68%           | 94.00%                 | 93.36%                | 1.15%                 |   0.0055 |  -0.1989 |             -0.1473 | 70.37%             |            1.95669e+08 |
|     11 | 2026-06-30 | SMCI     | 92.61%           | 91.78%                 | 93.43%                | 1.27%                 |   0.0465 |   0.3452 |             -0.1064 | 93.98%             |            1.5851e+09  |
|     12 | 2026-06-30 | COIN     | 92.25%           | 92.55%                 | 91.96%                | 1.24%                 |  -0.3409 |  -0.1463 |             -0.4937 | 91.97%             |            1.74282e+09 |
|     13 | 2026-06-30 | TER      | 91.42%           | 91.15%                 | 91.69%                | 1.84%                 |   1.2588 |   0.4741 |              1.106  | 92.41%             |            1.55052e+09 |
|     14 | 2026-06-30 | CEG      | 91.29%           | 91.86%                 | 90.72%                | 0.94%                 |  -0.2505 |  -0.0531 |             -0.4033 | 81.59%             |            9.87511e+08 |
|     15 | 2026-06-30 | DELL     | 91.18%           | 91.56%                 | 90.79%                | 1.42%                 |   2.1973 |   1.4415 |              2.0445 | 93.07%             |            2.70043e+09 |
|     16 | 2026-06-30 | MU       | 91.03%           | 91.87%                 | 90.20%                | 4.32%                 |   2.969  |   2.3517 |              2.8162 | 99.25%             |            3.88476e+10 |
|     17 | 2026-06-30 | SEDG     | 90.92%           | 90.85%                 | 90.98%                | 0.73%                 |   0.7938 |   0.0137 |              0.6409 | 71.23%             |            2.01712e+08 |
|     18 | 2026-06-30 | ANET     | 90.27%           | 89.65%                 | 90.89%                | 1.64%                 |   0.2028 |   0.2836 |              0.05   | 88.52%             |            1.44319e+09 |
|     19 | 2026-06-30 | CIEN     | 89.99%           | 90.11%                 | 89.88%                | 1.83%                 |   1.0503 |   0.2351 |              0.8975 | 92.06%             |            1.39184e+09 |
|     20 | 2026-06-30 | CRWD     | 89.64%           | 88.97%                 | 90.31%                | 2.87%                 |   0.4956 |   0.7958 |              0.3428 | 86.49%             |            1.89664e+09 |
|     21 | 2026-06-30 | FIX      | 89.52%           | 88.36%                 | 90.67%                | 2.41%                 |   0.9886 |   0.3452 |              0.8358 | 83.82%             |            7.81268e+08 |
|     22 | 2026-06-30 | VEEV     | 88.81%           | 90.34%                 | 87.28%                | 3.57%                 |  -0.2324 |  -0.0245 |             -0.3852 | 71.42%             |            5.22386e+08 |
|     23 | 2026-06-30 | FDS      | 87.86%           | 87.08%                 | 88.63%                | 1.78%                 |  -0.1934 |   0.0732 |             -0.3462 | 63.16%             |            2.19055e+08 |
|     24 | 2026-06-30 | GNRC     | 87.85%           | 88.38%                 | 87.32%                | 1.76%                 |   1.0431 |   0.4264 |              0.8903 | 65.44%             |            2.13667e+08 |
|     25 | 2026-06-30 | IBKR     | 87.70%           | 87.51%                 | 87.89%                | 1.14%                 |   0.3996 |   0.3406 |              0.2468 | 64.85%             |            4.02271e+08 |
|     26 | 2026-06-30 | LRCX     | 87.42%           | 88.04%                 | 86.80%                | 1.75%                 |   1.2188 |   0.7755 |              1.066  | 94.09%             |            3.16272e+09 |
|     27 | 2026-06-30 | JBL      | 87.24%           | 86.64%                 | 87.84%                | 2.04%                 |   0.5735 |   0.3503 |              0.4207 | 68.93%             |            4.32856e+08 |
|     28 | 2026-06-30 | CDNS     | 86.86%           | 86.08%                 | 87.63%                | 1.19%                 |   0.207  |   0.3577 |              0.0541 | 74.46%             |            8.60065e+08 |
|     29 | 2026-06-30 | ASML     | 86.78%           | 87.40%                 | 86.16%                | 2.60%                 |   0.6833 |   0.3617 |              0.5305 | 88.35%             |            2.99827e+09 |
|     30 | 2026-06-30 | ZS       | 86.41%           | 86.91%                 | 85.91%                | 3.32%                 |  -0.412  |  -0.0572 |             -0.5648 | 82.09%             |            6.37904e+08 |

## Feature weight profile ablation

This section compares manual XGBoost `feature_weights` profiles. The heavier profiles test whether the strong standalone 4m / 5m / 6m / 456 momentum baselines should receive a much stronger feature-sampling prior. The table is sorted by `total_return_1m_rebalanced`, then monthly return, then future max return.

| weight_profile           | is_main_profile   | core_momentum_group_weight   | mom_4m_weight   | mom_5m_weight   | mom_6m_weight   | core_mom_456_avg_weight   | mom_6m_acceleration_weight   |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |   prauc |    auc | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   |
|:-------------------------|:------------------|:-----------------------------|:----------------|:----------------|:----------------|:--------------------------|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:-----------------------------|:--------------------|--------:|-------:|:--------------------|:------------------|:------------------|:------------------------------|
| core_momentum_max_stress | False             | 500.00%                      | 1000.00%        | 1500.00%        | 1500.00%        | 2000.00%                  | 800.00%                      |       29 | 544.88%                      | 116.25%                           | 8.64%                   | 40.20%                       | 45.98%              |  0.2264 | 0.8161 | 44.44%              | 46.67%            | 30.00%            | 63.33%                        |
| core_momentum_heavy      | True              | 220.00%                      | 300.00%         | 350.00%         | 350.00%         | 400.00%                   | 260.00%                      |       29 | 269.90%                      | 71.82%                            | 6.44%                   | 33.31%                       | 39.08%              |  0.2287 | 0.8143 | 37.78%              | 40.00%            | 25.56%            | 60.00%                        |
| core_momentum_ultra      | False             | 300.00%                      | 400.00%         | 500.00%         | 500.00%         | 550.00%                   | 350.00%                      |       29 | 248.07%                      | 67.55%                            | 6.70%                   | 34.89%                       | 40.23%              |  0.23   | 0.8163 | 38.89%              | 43.33%            | 26.67%            | 56.67%                        |
| balanced_original        | False             | 125.00%                      | 125.00%         | 125.00%         | 125.00%         | 125.00%                   | 125.00%                      |       29 | 154.92%                      | 47.29%                            | 4.95%                   | 29.29%                       | 37.93%              |  0.2238 | 0.8169 | 36.67%              | 40.00%            | 23.33%            | 50.00%                        |

## Hyperparameter ablation: deeper trees, more rounds, lower learning rate

This section tests whether a deeper and slower XGBoost can learn subtler pre-boom interactions. All rows use the same features and the same main feature-weight profile; only the XGBoost hyperparameters change. The table is sorted by realized strategy performance.

| param_profile                | is_main_params   |   n_estimators | max_depth   | learning_rate   | min_child_weight   |   reg_alpha |   reg_lambda |   subsample |   colsample_bytree |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |   prauc |    auc | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   |
|:-----------------------------|:-----------------|---------------:|:------------|:----------------|:-------------------|------------:|-------------:|------------:|-------------------:|---------:|:-----------------------------|:----------------------------------|:------------------------|:-----------------------------|:--------------------|--------:|-------:|:--------------------|:------------------|:------------------|:------------------------------|
| deep_slow_4000_d5_lr0008     | False            |           4000 | 500.00%     | 0.80%           | 300.00%            |        0.05 |          2   |        0.85 |               0.9  |       29 | 479.17%                      | 106.84%                           | 8.13%                   | 33.11%                       | 41.38%              |  0.2232 | 0.809  | 40.00%              | 43.33%            | 24.44%            | 50.00%                        |
| deeper_slower_6000_d6_lr0005 | False            |           6000 | 600.00%     | 0.50%           | 300.00%            |        0.08 |          2.5 |        0.82 |               0.85 |       29 | 403.20%                      | 95.15%                            | 7.50%                   | 34.63%                       | 37.93%              |  0.2273 | 0.8093 | 36.67%              | 40.00%            | 24.44%            | 53.33%                        |
| reference_2000_d4_lr0015     | True             |           2000 | 400.00%     | 1.50%           | 400.00%            |        0.05 |          1.5 |        0.85 |               0.9  |       29 | 269.90%                      | 71.82%                            | 6.44%                   | 33.31%                       | 39.08%              |  0.2287 | 0.8143 | 37.78%              | 40.00%            | 25.56%            | 60.00%                        |

## Strategy and baseline comparison

`xgb_boom_probability` is computed on model prediction rows. `baseline_*` strategies are computed independently on the full clean test panel, requiring only the baseline score column and future-return labels. This keeps baseline returns fixed when model feature sets change.

| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:-----------------------------|:--------------------|
| baseline_mom_5m              |       29 | 1631.56%                     | 225.43%                           | 12.20%                  | 42.89%                       | 51.72%              |
| baseline_core_mom_456_avg    |       29 | 1493.46%                     | 214.43%                           | 11.58%                  | 41.81%                       | 50.57%              |
| baseline_mom_4m              |       29 | 1242.22%                     | 192.88%                           | 10.82%                  | 37.38%                       | 50.57%              |
| baseline_mom_6m              |       29 | 1127.90%                     | 182.29%                           | 10.88%                  | 38.47%                       | 48.28%              |
| baseline_rel_mom_6m_vs_qqq   |       29 | 1127.90%                     | 182.29%                           | 10.88%                  | 38.47%                       | 48.28%              |
| baseline_mom_6m_acceleration |       29 | 592.60%                      | 122.73%                           | 9.29%                   | 29.02%                       | 34.48%              |
| baseline_mom_3m              |       29 | 468.73%                      | 105.29%                           | 7.71%                   | 29.92%                       | 42.53%              |
| xgb_boom_probability         |       29 | 269.90%                      | 71.82%                            | 6.44%                   | 33.31%                       | 39.08%              |
| baseline_return_vol_ratio_6m |       29 | 99.66%                       | 33.13%                            | 2.93%                   | 18.91%                       | 24.14%              |

## Recent XGB Top-3 backtest months

| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-07-31 | SEDG, FSLR, MSTR   | 94.17%      | 8.91%       | 26.73%                   | 66.67%          |
| 2025-08-31 | UUUU, SEDG, ALAB   | 95.99%      | 16.55%      | 31.43%                   | 33.33%          |
| 2025-09-30 | SEDG, TSLA, SMCI   | 93.72%      | 1.96%       | 3.26%                    | 0.00%           |
| 2025-10-31 | SEDG, DOW, LRCX    | 92.50%      | 1.55%       | 23.28%                   | 33.33%          |
| 2025-11-30 | Q, ENPH, SEDG      | 95.09%      | -3.08%      | 33.28%                   | 66.67%          |
| 2025-12-31 | SEDG, COHR, ALAB   | 96.46%      | 4.26%       | 35.93%                   | 66.67%          |
| 2026-01-31 | RKLB, SEDG, ENPH   | 96.52%      | 4.99%       | 27.43%                   | 33.33%          |
| 2026-02-28 | RKLB, SMCI, ENPH   | 94.09%      | -15.77%     | 70.55%                   | 100.00%         |
| 2026-03-31 | ENPH, RKLB, TER    | 93.46%      | 10.50%      | 83.88%                   | 66.67%          |
| 2026-04-30 | ALAB, RKLB, MU     | 95.49%      | 79.24%      | 98.00%                   | 100.00%         |
| 2026-05-31 | UUUU, FDXF, ENPH   | 95.57%      | -18.07%     | -18.07%                  | 0.00%           |
| 2026-06-30 | FDXF, ENPH, RKLB   | 98.24%      |             |                          | 0.00%           |

## Ablation ranked summary

Each row drops one feature group and retrains the model. Negative `delta_top3_future_max_vs_main` means the removed group was useful for Top-3 tail capture.

| ablation                  |   dropped_features |   kept_features | top3_avg_future_max_return_1_3m   | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   | delta_top3_future_max_vs_main   |
|:--------------------------|-------------------:|----------------:|:----------------------------------|:--------------------|:------------------|:------------------|:------------------------------|:--------------------------------|
| drop_volatility_frequency |                 11 |              99 | 31.82%                            | 34.44%              | 36.67%            | 24.44%            | 53.33%                        | -1.49%                          |
| drop_qqq_context          |                  4 |             106 | 32.33%                            | 38.89%              | 41.11%            | 25.56%            | 50.00%                        | -0.98%                          |
| drop_trend                |                 12 |              98 | 33.01%                            | 35.56%              | 38.89%            | 25.56%            | 56.67%                        | -0.30%                          |
| drop_etf_source           |                  6 |             104 | 33.24%                            | 37.78%              | 37.78%            | 27.78%            | 53.33%                        | -0.07%                          |
| drop_liquidity_size       |                 11 |              99 | 34.97%                            | 37.78%              | 40.00%            | 25.56%            | 53.33%                        | 1.66%                           |
| drop_relative_strength    |                  4 |             106 | 35.06%                            | 36.67%              | 38.89%            | 26.67%            | 53.33%                        | 1.75%                           |
| drop_core_momentum        |                 14 |              96 | 35.40%                            | 37.78%              | 41.11%            | 25.56%            | 56.67%                        | 2.08%                           |
| drop_risk_drawdown        |                 19 |              91 | 35.74%                            | 40.00%              | 42.22%            | 27.78%            | 56.67%                        | 2.43%                           |
| drop_other_momentum       |                 13 |              97 | 36.63%                            | 38.89%              | 42.22%            | 28.89%            | 56.67%                        | 3.32%                           |
| drop_volume_flow          |                  9 |             101 | 38.27%                            | 40.00%              | 42.22%            | 28.89%            | 56.67%                        | 4.96%                           |

## Five-seed training stability

This retrains the same main model with five seeds and checks whether Top-K performance is stable.

|   seed |   prauc |    auc | precision_at_top3   | top3_hit30_rate   | top3_hit50_rate   | monthly_any_top3_hit50_rate   | top3_avg_future_return_1m   | top3_avg_future_max_return_1_3m   |
|-------:|--------:|-------:|:--------------------|:------------------|:------------------|:------------------------------|:----------------------------|:----------------------------------|
|      7 |  0.2313 | 0.8167 | 42.22%              | 45.56%            | 26.67%            | 60.00%                        | 8.51%                       | 36.89%                            |
|     42 |  0.2287 | 0.8143 | 37.78%              | 40.00%            | 25.56%            | 60.00%                        | 6.44%                       | 33.31%                            |
|    202 |  0.2316 | 0.8153 | 42.22%              | 43.33%            | 26.67%            | 56.67%                        | 7.41%                       | 37.14%                            |
|    777 |  0.2284 | 0.8148 | 43.33%              | 45.56%            | 26.67%            | 56.67%                        | 8.82%                       | 37.16%                            |
|   2026 |  0.2263 | 0.8142 | 41.11%              | 44.44%            | 26.67%            | 60.00%                        | 8.13%                       | 34.72%                            |

## Five-seed average feature importance

Feature importance is averaged across the five seed models. `feature_weight` is the manual XGBoost feature weight used during training, not a learned importance.

|   index | feature                  | tier                        | feature_group        |   feature_weight | mean   | std   | min   | max   |
|--------:|:-------------------------|:----------------------------|:---------------------|-----------------:|:-------|:------|:------|:------|
|       1 | intraday_range_mean_3m   | tier_2_tail_context         | volatility_frequency |             1.15 | 7.63%  | 0.48% | 7.22% | 8.27% |
|       2 | volatility_6m            | tier_4_downweighted_context | risk_drawdown        |             0.9  | 5.55%  | 0.53% | 4.67% | 5.96% |
|       3 | intraday_range_mean_6m   | tier_2_tail_context         | volatility_frequency |             1.15 | 5.18%  | 0.36% | 4.79% | 5.60% |
|       4 | atr_14_to_price          | tier_3_standard             | unclassified         |             1    | 2.41%  | 0.18% | 2.16% | 2.66% |
|       5 | in_manual_core           | tier_4_downweighted_context | etf_source           |             0.8  | 1.91%  | 0.06% | 1.84% | 1.97% |
|       6 | avg_abs_daily_return_6m  | tier_2_tail_context         | volatility_frequency |             1.15 | 1.64%  | 0.41% | 1.17% | 2.25% |
|       7 | theme_count              | tier_4_downweighted_context | etf_source           |             0.8  | 1.53%  | 0.22% | 1.29% | 1.81% |
|       8 | volatility_3m            | tier_4_downweighted_context | risk_drawdown        |             0.9  | 1.52%  | 0.31% | 1.16% | 2.01% |
|       9 | source_count             | tier_4_downweighted_context | etf_source           |             0.8  | 1.42%  | 0.11% | 1.28% | 1.57% |
|      10 | liquid_vol_score         | tier_3_standard             | liquidity_size       |             1.05 | 1.36%  | 0.05% | 1.30% | 1.44% |
|      11 | log_avg_dollar_volume_3m | tier_3_standard             | liquidity_size       |             1.05 | 1.18%  | 0.07% | 1.10% | 1.29% |
|      12 | source_weight_sum        | tier_4_downweighted_context | etf_source           |             0.8  | 1.14%  | 0.13% | 0.95% | 1.30% |
|      13 | avg_dollar_volume_3m     | tier_3_standard             | liquidity_size       |             1.05 | 1.13%  | 0.08% | 1.02% | 1.21% |
|      14 | large_move_freq_6m       | tier_2_tail_context         | volatility_frequency |             1.15 | 1.09%  | 0.06% | 1.01% | 1.17% |
|      15 | qqq_mom_6m               | tier_1_core_tail            | core_momentum        |             2.2  | 1.01%  | 0.04% | 0.98% | 1.06% |
|      16 | ma100_slope_1m           | tier_4_downweighted_context | trend                |             0.95 | 1.01%  | 0.10% | 0.89% | 1.10% |
|      17 | avg_dollar_volume_6m     | tier_3_standard             | liquidity_size       |             1.05 | 0.98%  | 0.03% | 0.95% | 1.02% |
|      18 | drawdown_12m_abs         | tier_4_downweighted_context | risk_drawdown        |             0.9  | 0.98%  | 0.06% | 0.88% | 1.05% |
|      19 | volatility_3m_to_6m      | tier_4_downweighted_context | risk_drawdown        |             0.9  | 0.95%  | 0.04% | 0.91% | 1.02% |
|      20 | rel_mom_6m_vs_qqq        | tier_1_core_tail            | core_momentum        |             2.2  | 0.91%  | 0.04% | 0.86% | 0.97% |
|      21 | qqq_mom_1m               | tier_3_standard             | other_momentum       |             1    | 0.90%  | 0.03% | 0.88% | 0.96% |
|      22 | avg_dollar_volume_1m     | tier_3_standard             | liquidity_size       |             1.05 | 0.88%  | 0.07% | 0.77% | 0.94% |
|      23 | qqq_mom_3m               | tier_3_standard             | other_momentum       |             1    | 0.87%  | 0.02% | 0.85% | 0.90% |
|      24 | mom_4m                   | tier_1_core_tail            | core_momentum        |             3    | 0.86%  | 0.05% | 0.80% | 0.94% |
|      25 | drawdown_3m_abs          | tier_4_downweighted_context | risk_drawdown        |             0.9  | 0.86%  | 0.02% | 0.82% | 0.87% |
|      26 | drawdown_12m             | tier_4_downweighted_context | risk_drawdown        |             0.9  | 0.85%  | 0.04% | 0.81% | 0.91% |
|      27 | in_large_cap_core        | tier_4_downweighted_context | etf_source           |             0.8  | 0.85%  | 0.30% | 0.52% | 1.22% |
|      28 | ret_lag_6m_max           | tier_3_standard             | unclassified         |             1    | 0.84%  | 0.06% | 0.78% | 0.91% |
|      29 | avg_abs_daily_return_3m  | tier_2_tail_context         | volatility_frequency |             1.15 | 0.83%  | 0.26% | 0.69% | 1.30% |
|      30 | drawdown_1m_abs          | tier_4_downweighted_context | risk_drawdown        |             0.9  | 0.83%  | 0.06% | 0.74% | 0.88% |
|      31 | mom_6m                   | tier_1_core_tail            | core_momentum        |             3.5  | 0.79%  | 0.11% | 0.65% | 0.88% |
|      32 | mom_6m_first3m           | tier_1_core_tail            | core_momentum        |             2.4  | 0.79%  | 0.02% | 0.76% | 0.81% |
|      33 | ret_lag_6m_mean          | tier_3_standard             | unclassified         |             1    | 0.79%  | 0.08% | 0.73% | 0.93% |
|      34 | in_core_growth           | tier_4_downweighted_context | etf_source           |             0.8  | 0.78%  | 0.07% | 0.70% | 0.89% |
|      35 | mom_12m                  | tier_3_standard             | other_momentum       |             1    | 0.78%  | 0.04% | 0.72% | 0.82% |
|      36 | qqq_mom_12m              | tier_3_standard             | other_momentum       |             1    | 0.78%  | 0.03% | 0.75% | 0.80% |
|      37 | price_ma5_ratio          | tier_4_downweighted_context | trend                |             0.95 | 0.76%  | 0.04% | 0.69% | 0.78% |
|      38 | drawdown_3m              | tier_4_downweighted_context | risk_drawdown        |             0.9  | 0.75%  | 0.05% | 0.71% | 0.83% |
|      39 | core_mom_456_min         | tier_1_core_tail            | core_momentum        |             3    | 0.74%  | 0.05% | 0.67% | 0.81% |
|      40 | mom_6m_acceleration      | tier_1_core_tail            | core_momentum        |             2.6  | 0.74%  | 0.02% | 0.71% | 0.77% |
|      41 | price_ma30_ratio         | tier_4_downweighted_context | trend                |             0.95 | 0.74%  | 0.07% | 0.66% | 0.82% |
|      42 | ret_lag_6m_std           | tier_3_standard             | unclassified         |             1    | 0.74%  | 0.01% | 0.71% | 0.75% |
|      43 | drawdown_6m              | tier_4_downweighted_context | risk_drawdown        |             0.9  | 0.74%  | 0.07% | 0.66% | 0.85% |
|      44 | drawdown_1m              | tier_4_downweighted_context | risk_drawdown        |             0.9  | 0.73%  | 0.05% | 0.68% | 0.81% |
|      45 | drawdown_6m_abs          | tier_4_downweighted_context | risk_drawdown        |             0.9  | 0.72%  | 0.05% | 0.69% | 0.80% |
|      46 | core_mom_456_max         | tier_1_core_tail            | core_momentum        |             3    | 0.72%  | 0.05% | 0.68% | 0.79% |
|      47 | large_move_freq_3m       | tier_2_tail_context         | volatility_frequency |             1.15 | 0.71%  | 0.05% | 0.65% | 0.77% |
|      48 | volatility_1m            | tier_4_downweighted_context | risk_drawdown        |             0.9  | 0.71%  | 0.02% | 0.70% | 0.74% |
|      49 | core_mom_456_avg         | tier_1_core_tail            | core_momentum        |             4    | 0.71%  | 0.03% | 0.67% | 0.74% |
|      50 | mom_9m                   | tier_3_standard             | other_momentum       |             1    | 0.69%  | 0.03% | 0.65% | 0.73% |

## Manual feature weights used by XGBoost

|   index | feature                       | tier                        | feature_group        |   feature_weight |
|--------:|:------------------------------|:----------------------------|:---------------------|-----------------:|
|       1 | rel_mom_12m_vs_qqq            | tier_3_standard             | other_momentum       |             1    |
|       2 | mom_12m                       | tier_3_standard             | other_momentum       |             1    |
|       3 | up_day_volume_ratio_3m        | tier_4_downweighted_context | volume_flow          |             0.9  |
|       4 | up_day_dollar_volume_ratio_3m | tier_3_standard             | liquidity_size       |             1.05 |
|       5 | mom_9m                        | tier_3_standard             | other_momentum       |             1    |
|       6 | mom_7m                        | tier_3_standard             | other_momentum       |             1    |
|       7 | mom_6m                        | tier_1_core_tail            | core_momentum        |             3.5  |
|       8 | mom_4m_vs_6m                  | tier_1_core_tail            | core_momentum        |             2.2  |
|       9 | mom_5m_vs_6m                  | tier_1_core_tail            | core_momentum        |             2.2  |
|      10 | mom_6m_first3m                | tier_1_core_tail            | core_momentum        |             2.4  |
|      11 | mom_6m_acceleration           | tier_1_core_tail            | core_momentum        |             2.6  |
|      12 | ret_lag_6m                    | tier_3_standard             | unclassified         |             1    |
|      13 | rel_mom_6m_vs_qqq             | tier_1_core_tail            | core_momentum        |             2.2  |
|      14 | mom_3m_vs_6m                  | tier_3_standard             | other_momentum       |             1    |
|      15 | return_vol_ratio_6m           | tier_4_downweighted_context | risk_drawdown        |             0.9  |
|      16 | drawdown_12m                  | tier_4_downweighted_context | risk_drawdown        |             0.9  |
|      17 | drawdown_12m_abs              | tier_4_downweighted_context | risk_drawdown        |             0.9  |
|      18 | ma100_slope_1m                | tier_4_downweighted_context | trend                |             0.95 |
|      19 | mom_5m                        | tier_1_core_tail            | core_momentum        |             3.5  |
|      20 | core_mom_456_std              | tier_1_core_tail            | core_momentum        |             2.2  |
|      21 | ret_lag_5m                    | tier_3_standard             | unclassified         |             1    |
|      22 | volume_ma3_to_12m             | tier_4_downweighted_context | volume_flow          |             0.9  |
|      23 | price_ma100_ratio             | tier_4_downweighted_context | trend                |             0.95 |
|      24 | volume_change_3m              | tier_4_downweighted_context | volume_flow          |             0.9  |
|      25 | dollar_volume_change_3m       | tier_3_standard             | liquidity_size       |             1.05 |
|      26 | drawdown_change_3m            | tier_4_downweighted_context | risk_drawdown        |             0.9  |
|      27 | mom_4m                        | tier_1_core_tail            | core_momentum        |             3    |
|      28 | core_mom_456_avg              | tier_1_core_tail            | core_momentum        |             4    |
|      29 | core_mom_456_min              | tier_1_core_tail            | core_momentum        |             3    |
|      30 | core_mom_456_max              | tier_1_core_tail            | core_momentum        |             3    |
|      31 | ret_lag_4m                    | tier_3_standard             | unclassified         |             1    |
|      32 | ma50_slope_1m                 | tier_4_downweighted_context | trend                |             0.95 |
|      33 | mom_6m_last3m                 | tier_1_core_tail            | core_momentum        |             2.4  |
|      34 | ret_lag_3m                    | tier_3_standard             | unclassified         |             1    |
|      35 | rel_mom_3m_vs_qqq             | tier_3_standard             | other_momentum       |             1    |
|      36 | mom_3m                        | tier_3_standard             | other_momentum       |             1    |
|      37 | volatility_6m                 | tier_4_downweighted_context | risk_drawdown        |             0.9  |
|      38 | return_vol_ratio_3m           | tier_4_downweighted_context | risk_drawdown        |             0.9  |
|      39 | volatility_1m_to_6m           | tier_4_downweighted_context | risk_drawdown        |             0.9  |
|      40 | volatility_3m_to_6m           | tier_4_downweighted_context | risk_drawdown        |             0.9  |
|      41 | avg_abs_daily_return_6m       | tier_2_tail_context         | volatility_frequency |             1.15 |
|      42 | drawdown_6m                   | tier_4_downweighted_context | risk_drawdown        |             0.9  |
|      43 | drawdown_6m_abs               | tier_4_downweighted_context | risk_drawdown        |             0.9  |
|      44 | recovery_from_6m_low          | tier_3_standard             | unclassified         |             1    |
|      45 | avg_dollar_volume_6m          | tier_3_standard             | liquidity_size       |             1.05 |
|      46 | dollar_volume_3m_to_6m        | tier_3_standard             | liquidity_size       |             1.05 |
|      47 | large_move_freq_6m            | tier_2_tail_context         | volatility_frequency |             1.15 |
|      48 | up_big_move_freq_6m           | tier_2_tail_context         | volatility_frequency |             1.15 |
|      49 | down_big_move_freq_6m         | tier_2_tail_context         | volatility_frequency |             1.15 |
|      50 | intraday_range_mean_6m        | tier_2_tail_context         | volatility_frequency |             1.15 |
|      51 | volume_change_1m              | tier_4_downweighted_context | volume_flow          |             0.9  |
|      52 | dollar_volume_change_1m       | tier_3_standard             | liquidity_size       |             1.05 |
|      53 | drawdown_change_1m            | tier_4_downweighted_context | risk_drawdown        |             0.9  |
|      54 | ma30_slope_1m                 | tier_4_downweighted_context | trend                |             0.95 |
|      55 | price_ma50_ratio              | tier_4_downweighted_context | trend                |             0.95 |
|      56 | atr_14_to_100d                | tier_3_standard             | unclassified         |             1    |
|      57 | volume_ma1_to_6m              | tier_4_downweighted_context | volume_flow          |             0.9  |
|      58 | ret_lag_2m                    | tier_3_standard             | unclassified         |             1    |
|      59 | ret_lag_6m_std                | tier_3_standard             | unclassified         |             1    |
|      60 | mom_2m                        | tier_3_standard             | other_momentum       |             1    |
|      61 | ma20_slope_1m                 | tier_4_downweighted_context | trend                |             0.95 |
|      62 | volume_ratio_3m               | tier_4_downweighted_context | volume_flow          |             0.9  |
|      63 | volatility_3m                 | tier_4_downweighted_context | risk_drawdown        |             0.9  |
|      64 | volatility_1m_to_3m           | tier_4_downweighted_context | risk_drawdown        |             0.9  |
|      65 | avg_abs_daily_return_3m       | tier_2_tail_context         | volatility_frequency |             1.15 |
|      66 | ma10_slope_1m                 | tier_4_downweighted_context | trend                |             0.95 |
|      67 | drawdown_3m                   | tier_4_downweighted_context | risk_drawdown        |             0.9  |
|      68 | drawdown_3m_abs               | tier_4_downweighted_context | risk_drawdown        |             0.9  |
|      69 | recovery_from_3m_low          | tier_3_standard             | unclassified         |             1    |
|      70 | days_since_3m_high_norm       | tier_3_standard             | unclassified         |             1    |
|      71 | days_since_3m_low_norm        | tier_3_standard             | unclassified         |             1    |
|      72 | avg_dollar_volume_3m          | tier_3_standard             | liquidity_size       |             1.05 |
|      73 | log_avg_dollar_volume_3m      | tier_3_standard             | liquidity_size       |             1.05 |
|      74 | large_move_freq_3m            | tier_2_tail_context         | volatility_frequency |             1.15 |
|      75 | up_big_move_freq_3m           | tier_2_tail_context         | volatility_frequency |             1.15 |
|      76 | down_big_move_freq_3m         | tier_2_tail_context         | volatility_frequency |             1.15 |
|      77 | intraday_range_mean_3m        | tier_2_tail_context         | volatility_frequency |             1.15 |
|      78 | price_ma30_ratio              | tier_4_downweighted_context | trend                |             0.95 |
|      79 | ma5_slope_1m                  | tier_4_downweighted_context | trend                |             0.95 |
|      80 | return_vol_ratio_1m           | tier_4_downweighted_context | risk_drawdown        |             0.9  |

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
