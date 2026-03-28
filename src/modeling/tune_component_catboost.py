import argparse
import json
import sys
from time import perf_counter

import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

from src.config.paths import OUTPUTS_DIR, ensure_project_directories
from src.modeling.component_catboost import (
    CAT_COLS,
    FEATURE_COLS,
    TARGET_COL,
    build_model,
    load_cases,
    prep_cases,
    split_cases,
)

# -----------------------------------------------------------------------------
# Output names
# -----------------------------------------------------------------------------
TRIALS_NAME = 'component_catboost_optuna_trials.csv'
BEST_NAME = 'component_catboost_optuna_best_params.json'
BEST_METRIC_NAME = 'component_catboost_optuna_best_metrics.csv'


# -----------------------------------------------------------------------------
# Search setup
# -----------------------------------------------------------------------------
DEF_TRIALS = 40
DEF_SEEDS = [42, 43]

FIXED_PARAMS = {
    'bootstrap_type': 'Bernoulli',
    'border_count': 128
}


# -----------------------------------------------------------------------------
# Search helpers
# -----------------------------------------------------------------------------
def parse_seed_text(seed_text):
    if not seed_text:
        return DEF_SEEDS
    return [int(part.strip()) for part in seed_text.split(',') if part.strip()]


def suggest_params(trial):
    return {
        'bootstrap_type': 'Bernoulli',
        'border_count': 128,
        'iterations': trial.suggest_int('iterations', 1200, 2000, step=200),
        'learning_rate': trial.suggest_float('learning_rate', 0.045, 0.10, log=True),
        'depth': trial.suggest_int('depth', 8, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 4.0, 12.0),
        'random_strength': trial.suggest_float('random_strength', 0.05, 0.6, log=True),
        'subsample': trial.suggest_float('subsample', 0.60, 0.75)
    }


def score_trial(y_valid, pred, proba, classes):
    return {
        'top_1_accuracy': accuracy_score(y_valid, pred),
        'macro_f1': f1_score(y_valid, pred, average='macro'),
        'top_3_accuracy': top_k_accuracy_score(y_valid, proba, labels=classes, k=3)
    }


def build_objective(train_df, valid_df, task_type, devices, seed_list, verbose):
    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df[TARGET_COL].copy()
    X_valid = valid_df[FEATURE_COLS].copy()
    y_valid = valid_df[TARGET_COL].copy()

    def objective(trial):
        tuned = suggest_params(trial)
        run_rows = []

        for seed in seed_list:
            model = build_model(
                task_type=task_type,
                devices=devices,
                random_seed=seed,
                verbose=verbose,
                params_override=tuned
            )

            start = perf_counter()
            model.fit(
                X_train,
                y_train,
                cat_features=CAT_COLS,
                eval_set=(X_valid, y_valid),
                use_best_model=True,
                early_stopping_rounds=100
            )
            fit_seconds = perf_counter() - start

            valid_pred = model.predict(X_valid).ravel()
            valid_proba = model.predict_proba(X_valid)
            scores = score_trial(y_valid, valid_pred, valid_proba, model.classes_)
            run_rows.append(
                {
                    'seed': seed,
                    'fit_seconds': round(fit_seconds, 2),
                    'best_iteration': int(model.get_best_iteration()),
                    **scores
                }
            )

        run_df = pd.DataFrame(run_rows)
        trial.set_user_attr('fit_seconds_mean', round(run_df['fit_seconds'].mean(), 2))
        trial.set_user_attr('fit_seconds_std', round(run_df['fit_seconds'].std(ddof=0), 2))
        trial.set_user_attr('best_iteration_mean', round(run_df['best_iteration'].mean(), 2))
        trial.set_user_attr('top_1_accuracy_mean', round(run_df['top_1_accuracy'].mean(), 4))
        trial.set_user_attr('top_1_accuracy_std', round(run_df['top_1_accuracy'].std(ddof=0), 4))
        trial.set_user_attr('macro_f1_mean', round(run_df['macro_f1'].mean(), 4))
        trial.set_user_attr('macro_f1_std', round(run_df['macro_f1'].std(ddof=0), 4))
        trial.set_user_attr('top_3_accuracy_mean', round(run_df['top_3_accuracy'].mean(), 4))
        trial.set_user_attr('top_3_accuracy_std', round(run_df['top_3_accuracy'].std(ddof=0), 4))
        return float(run_df['macro_f1'].mean())

    return objective


def build_trials_df(study):
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        rows.append(
            {
                'trial': trial.number,
                'value_macro_f1': round(trial.value, 4),
                'fit_seconds_mean': trial.user_attrs.get('fit_seconds_mean'),
                'fit_seconds_std': trial.user_attrs.get('fit_seconds_std'),
                'best_iteration_mean': trial.user_attrs.get('best_iteration_mean'),
                'top_1_accuracy_mean': trial.user_attrs.get('top_1_accuracy_mean'),
                'top_1_accuracy_std': trial.user_attrs.get('top_1_accuracy_std'),
                'macro_f1_mean': trial.user_attrs.get('macro_f1_mean'),
                'macro_f1_std': trial.user_attrs.get('macro_f1_std'),
                'top_3_accuracy_mean': trial.user_attrs.get('top_3_accuracy_mean'),
                'top_3_accuracy_std': trial.user_attrs.get('top_3_accuracy_std'),
                **trial.params
            }
        )

    if not rows:
        raise ValueError('No completed Optuna trials were produced')

    return pd.DataFrame(rows).sort_values(
        ['value_macro_f1', 'top_1_accuracy_mean', 'top_3_accuracy_mean'],
        ascending=False
    ).reset_index(drop=True)


def refit_best(train_df, valid_df, best_params, task_type, devices, random_seed, verbose):
    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df[TARGET_COL].copy()
    X_valid = valid_df[FEATURE_COLS].copy()
    y_valid = valid_df[TARGET_COL].copy()

    model = build_model(
        task_type=task_type,
        devices=devices,
        random_seed=random_seed,
        verbose=verbose,
        params_override=best_params
    )

    start = perf_counter()
    model.fit(
        X_train,
        y_train,
        cat_features=CAT_COLS,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=100
    )
    fit_seconds = perf_counter() - start

    valid_pred = model.predict(X_valid).ravel()
    valid_proba = model.predict_proba(X_valid)
    scores = score_trial(y_valid, valid_pred, valid_proba, model.classes_)

    best_metric_df = pd.DataFrame(
        [
            {
                'split': 'valid_2025',
                'rows': int(len(y_valid)),
                'fit_seconds': round(fit_seconds, 2),
                'best_iteration': int(model.get_best_iteration()),
                'top_1_accuracy': round(scores['top_1_accuracy'], 4),
                'macro_f1': round(scores['macro_f1'], 4),
                'top_3_accuracy': round(scores['top_3_accuracy'], 4)
            }
        ]
    )

    return model, best_metric_df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a focused Optuna search for the component CatBoost model around the best benchmark region'
    )
    parser.add_argument(
        '--input-path',
        default=None,
        help='Optional path to the component case parquet or csv file'
    )
    parser.add_argument(
        '--task-type',
        choices=['CPU', 'GPU', 'cpu', 'gpu'],
        default='GPU',
        help='CatBoost processing target. GPU is recommended for Optuna tuning'
    )
    parser.add_argument(
        '--devices',
        default='0',
        help='GPU device string for CatBoost when task_type is GPU'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=DEF_TRIALS,
        help='Number of Optuna trials to run'
    )
    parser.add_argument(
        '--seed-list',
        default='42,43',
        help='Comma-separated seeds used to score each trial for stability'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Sampler seed and best-model refit seed'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=0,
        help='CatBoost logging interval for trial runs'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    seed_list = parse_seed_text(args.seed_list)
    raw_df = load_cases(args.input_path)
    case_df = prep_cases(raw_df)
    train_df, valid_df, holdout_df, split_df = split_cases(case_df)

    sampler = optuna.samplers.TPESampler(seed=args.random_seed)
    study = optuna.create_study(
        study_name='component_catboost_optuna_focus',
        direction='maximize',
        sampler=sampler
    )
    study.enqueue_trial(
        {
            'iterations': 1800,
            'learning_rate': 0.07405467149893648,
            'depth': 9,
            'l2_leaf_reg': 7.572705439311379,
            'random_strength': 0.29374126086853103,
            'subsample': 0.6895168484791427
        }
    )
    study.enqueue_trial(
        {
            'iterations': 1400,
            'learning_rate': 0.09506321489947779,
            'depth': 9,
            'l2_leaf_reg': 9.019802379437923,
            'random_strength': 0.151734280408781,
            'subsample': 0.6282559329229486
        }
    )

    objective = build_objective(
        train_df,
        valid_df,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        seed_list=seed_list,
        verbose=args.verbose
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    trials_df = build_trials_df(study)
    best_params = dict(FIXED_PARAMS)
    best_params.update(study.best_trial.params)
    _, best_metric_df = refit_best(
        train_df,
        valid_df,
        best_params,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed,
        verbose=args.verbose
    )

    best_info = {
        'study_name': study.study_name,
        'n_trials': args.n_trials,
        'seed_list': seed_list,
        'task_type': str(args.task_type).upper(),
        'devices': args.devices if str(args.task_type).upper() == 'GPU' else None,
        'random_seed': args.random_seed,
        'fixed_params': FIXED_PARAMS,
        'best_trial_number': study.best_trial.number,
        'best_value_macro_f1': round(study.best_trial.value, 4),
        'best_params': best_params,
        'best_trial_attrs': study.best_trial.user_attrs,
        'split_rows': {
            'train': int(len(train_df)),
            'valid_2025': int(len(valid_df)),
            'holdout_2026': int(len(holdout_df))
        }
    }

    trials_path = OUTPUTS_DIR / TRIALS_NAME
    best_path = OUTPUTS_DIR / BEST_NAME
    best_metric_path = OUTPUTS_DIR / BEST_METRIC_NAME

    trials_df.to_csv(trials_path, index=False)
    best_metric_df.to_csv(best_metric_path, index=False)
    with best_path.open('w', encoding='utf-8') as handle:
        json.dump(best_info, handle, indent=2)

    print(f'[write] {trials_path}')
    print(f'[write] {best_metric_path}')
    print(f'[write] {best_path}')
    print('')
    print('[done] Focused CatBoost Optuna tuning finished')
    return 0


if __name__ == '__main__':
    sys.exit(main())
