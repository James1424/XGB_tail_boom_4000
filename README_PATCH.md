# XGB_tail_boom_predict_fixed modified files v2

Replace this file in the repository:

```text
src/model_config.py
```

Then run:

```bash
python run_all.py
```

## What changed

- Main model remains the new 4000-round baseline:
  - `n_estimators=4000`
  - `max_depth=5`
  - `learning_rate=0.008`
- `TRAINING_CURVE_ROUNDS` now goes to 4000.
- Hyperparameter ablation compares:
  - `legacy_2000_d4_lr0015`
  - `rounds_3000_d5_lr0008`
  - `main_4000_d5_lr0008`
  - `rounds_5000_d5_lr0008`
  - `rounds_6000_d5_lr0008`
- Feature-weight profiles were reset as requested:
  - kept `core_momentum_max_stress` as the original/main profile
  - removed the lighter `balanced_original`, `core_momentum_heavy`, and `core_momentum_ultra` profiles
  - added four more aggressive profiles:
    - `core_momentum_aggressive_1`
    - `core_momentum_aggressive_2`
    - `core_momentum_aggressive_3`
    - `core_momentum_pure_ranker_stress`
- `MAIN_WEIGHT_PROFILE` is now `core_momentum_max_stress`.
