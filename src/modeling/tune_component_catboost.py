import argparse
import sys

import optuna
import pandas as pd

from src.config.paths import OUTPUTS_DIR, ensure_project_directories
from src.modeling.component_common import (
    BENCHMARK_FEATURE_SET_NAMES,
    DEFAULT_SELECTION_SEEDS,
    SINGLE_INPUT_STEM,
    TARGET_COL,
    TRAIN_END,
    VALID_END,
    feature_manifest,
    fit_catboost_with_external_selection,
    get_git_dirty_flag,
    get_git_head,
    json_ready,
    load_frame,
    prep_single_label_cases,
    runtime_manifest,
    sha256_path,
    split_single_label_cases,
    write_json,
)

# -----------------------------------------------------------------------------
# Output names
# -----------------------------------------------------------------------------
FEATURES_NAME = 'component_single_label_selection_feature_sets.csv'
TRIALS_NAME = 'component_single_label_selection_trials.csv'
BEST_METRIC_NAME = 'component_single_label_selection_best_metrics.csv'
MANIFEST_NAME = 'component_single_label_selection_manifest.json'


# -----------------------------------------------------------------------------
# Search setup
# -----------------------------------------------------------------------------
DEF_TRIALS = 40
DEF_SEEDS = DEFAULT_SELECTION_SEEDS
DEF_SELECTION_EVAL_PERIOD = 10
DEFAULT_SEED_TEXT = ','.join(str(seed) for seed in DEF_SEEDS)
QUICK_FEATURE_SET = 'core_structured'
QUICK_TRIALS = 8
QUICK_SEEDS = [42]
QUICK_SELECTION_EVAL_PERIOD = 25
SUMMARY_KEYS = [
    'fit_seconds_mean',
    'fit_seconds_std',
    'selected_iteration_mean',
    'selected_iteration_median',
    'top_1_accuracy_mean',
    'top_1_accuracy_std',
    'macro_f1_mean',
    'macro_f1_std',
    'top_3_accuracy_mean',
    'top_3_accuracy_std'
]

ANCHOR_PARAMS = {
    'bootstrap_type': 'Bernoulli',
    'border_count': 128,
    'iterations': 1800,
    'learning_rate': 0.07405467149893648,
    'depth': 9,
    'l2_leaf_reg': 7.572705439311379,
    'random_strength': 0.29374126086853103,
    'subsample': 0.6895168484791427
}

ALT_ANCHOR_PARAMS = {
    'bootstrap_type': 'Bernoulli',
    'border_count': 128,
    'iterations': 1400,
    'learning_rate': 0.09506321489947779,
    'depth': 9,
    'l2_leaf_reg': 9.019802379437923,
    'random_strength': 0.151734280408781,
    'subsample': 0.6282559329229486
}


# -----------------------------------------------------------------------------
# Search helpers
# -----------------------------------------------------------------------------
def log_line(message=''):
    print(message, flush=True)


def format_seed_list(seed_list):
    return ','.join(str(seed) for seed in seed_list)


def parse_seed_text(seed_text, default_seeds=None):
    fallback = list(DEF_SEEDS if default_seeds is None else default_seeds)
    if not seed_text:
        return fallback

    parsed = [int(part.strip()) for part in seed_text.split(',') if part.strip()]
    if not parsed:
        raise ValueError('Seed list is empty after parsing')
    return parsed


def resolve_run_config(args):
    if int(args.n_trials) < 1:
        raise ValueError('n_trials must be at least 1')
    if int(args.selection_eval_period) < 1:
        raise ValueError('selection_eval_period must be at least 1')

    seed_list = parse_seed_text(args.seed_list)
    selection_seed_list = parse_seed_text(
        args.feature_selection_seed_list,
        default_seeds=seed_list
    )
    manual_feature_set = args.feature_set
    n_trials = int(args.n_trials)
    selection_eval_period = int(args.selection_eval_period)
    quick_notes = []

    if args.quick:
        if manual_feature_set is None:
            manual_feature_set = QUICK_FEATURE_SET
            quick_notes.append(f'Feature sweep skipped with {QUICK_FEATURE_SET}')
        if args.n_trials == DEF_TRIALS:
            n_trials = QUICK_TRIALS
            quick_notes.append(f'n_trials -> {QUICK_TRIALS}')
        if args.seed_list == DEFAULT_SEED_TEXT:
            seed_list = list(QUICK_SEEDS)
            quick_notes.append(f'tuning seeds -> {format_seed_list(seed_list)}')
        if args.feature_selection_seed_list is None:
            selection_seed_list = list(QUICK_SEEDS)
            quick_notes.append(f'feature selection seeds -> {format_seed_list(selection_seed_list)}')
        if int(args.selection_eval_period) == DEF_SELECTION_EVAL_PERIOD:
            selection_eval_period = QUICK_SELECTION_EVAL_PERIOD
            quick_notes.append(f'selection_eval_period -> {QUICK_SELECTION_EVAL_PERIOD}')

    return {
        'task_type': str(args.task_type).upper(),
        'devices': args.devices,
        'n_trials': n_trials,
        'seed_list': seed_list,
        'selection_seed_list': selection_seed_list,
        'selection_eval_period': selection_eval_period,
        'manual_feature_set': manual_feature_set,
        'run_feature_selection': manual_feature_set is None,
        'feature_set_names': [manual_feature_set] if manual_feature_set else list(BENCHMARK_FEATURE_SET_NAMES),
        'quick_notes': quick_notes
    }


def build_fit_plan(run_config):
    feature_selection_fits = (
        len(run_config['feature_set_names']) * len(run_config['selection_seed_list'])
        if run_config['run_feature_selection']
        else 0
    )
    optuna_fits = int(run_config['n_trials']) * len(run_config['seed_list'])
    best_trial_rescore_fits = len(run_config['seed_list'])
    return {
        'feature_selection_fits': feature_selection_fits,
        'optuna_fits': optuna_fits,
        'best_trial_rescore_fits': best_trial_rescore_fits,
        'total_fits': feature_selection_fits + optuna_fits + best_trial_rescore_fits
    }


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


def evaluate_params_across_seeds(train_df, valid_df, feature_info, params, task_type, devices, seed_list, verbose, selection_eval_period=1, progress_label=None):
    run_rows = []
    total_seeds = len(seed_list)
    for seed_idx, seed in enumerate(seed_list, start=1):
        result = fit_catboost_with_external_selection(
            train_df,
            valid_df,
            feature_info,
            params,
            task_type=task_type,
            devices=devices,
            random_seed=seed,
            verbose=verbose,
            selection_eval_period=selection_eval_period,
            include_train_outputs=False,
            include_valid_outputs=False
        )
        row = {
            'seed': seed,
            'fit_seconds': result['fit_seconds'],
            'selected_iteration': result['selected_iteration'],
            'top_1_accuracy': result['valid_metrics']['top_1_accuracy'],
            'macro_f1': result['valid_metrics']['macro_f1'],
            'top_3_accuracy': result['valid_metrics']['top_3_accuracy']
        }
        run_rows.append(row)

        if progress_label:
            log_line(
                f'[{progress_label}] seed {seed_idx}/{total_seeds} ({seed}) '
                f'fit={row["fit_seconds"]:.2f}s iter={row["selected_iteration"]} '
                f'macro_f1={row["macro_f1"]:.4f} top1={row["top_1_accuracy"]:.4f} '
                f'top3={row["top_3_accuracy"]:.4f}'
            )
    return pd.DataFrame(run_rows)


def summarize_seed_metrics(run_df):
    return {
        'fit_seconds_mean': round(float(run_df['fit_seconds'].mean()), 2),
        'fit_seconds_std': round(float(run_df['fit_seconds'].std(ddof=0)), 2),
        'selected_iteration_mean': round(float(run_df['selected_iteration'].mean()), 2),
        'selected_iteration_median': int(run_df['selected_iteration'].median()),
        'top_1_accuracy_mean': round(float(run_df['top_1_accuracy'].mean()), 4),
        'top_1_accuracy_std': round(float(run_df['top_1_accuracy'].std(ddof=0)), 4),
        'macro_f1_mean': round(float(run_df['macro_f1'].mean()), 4),
        'macro_f1_std': round(float(run_df['macro_f1'].std(ddof=0)), 4),
        'top_3_accuracy_mean': round(float(run_df['top_3_accuracy'].mean()), 4),
        'top_3_accuracy_std': round(float(run_df['top_3_accuracy'].std(ddof=0)), 4)
    }


def build_manual_feature_selection(feature_set_name, selection_eval_period):
    feature_info = feature_manifest(feature_set_name)
    summary = dict.fromkeys(SUMMARY_KEYS, pd.NA)
    feature_df = pd.DataFrame(
        [
            {
                'feature_set_name': feature_set_name,
                'feature_count': len(feature_info['feature_cols']),
                'selection_mode': 'manual_override',
                'selection_eval_period': int(selection_eval_period),
                **summary
            }
        ]
    )
    feature_results = {
        feature_set_name: {
            'feature_info': feature_info,
            'seed_metrics': pd.DataFrame(),
            'summary': summary
        }
    }
    return feature_df, feature_results


def evaluate_feature_sets(train_df, valid_df, task_type, devices, seed_list, verbose, selection_eval_period, feature_set_names=None):
    rows = []
    feature_results = {}
    feature_set_names = list(BENCHMARK_FEATURE_SET_NAMES) if feature_set_names is None else list(feature_set_names)
    total_feature_sets = len(feature_set_names)

    for feature_idx, feature_set_name in enumerate(feature_set_names, start=1):
        feature_info = feature_manifest(feature_set_name)
        log_line(
            f'[feature selection] {feature_idx}/{total_feature_sets} '
            f'{feature_set_name} ({len(feature_info["feature_cols"])} features)'
        )
        train_ready = prep_single_label_cases(train_df, feature_info['feature_cols'])
        valid_ready = prep_single_label_cases(valid_df, feature_info['feature_cols'])
        run_df = evaluate_params_across_seeds(
            train_ready,
            valid_ready,
            feature_info,
            ANCHOR_PARAMS,
            task_type=task_type,
            devices=devices,
            seed_list=seed_list,
            verbose=verbose,
            selection_eval_period=selection_eval_period,
            progress_label=f'feature selection | {feature_set_name}'
        )
        summary = summarize_seed_metrics(run_df)
        feature_results[feature_set_name] = {
            'feature_info': feature_info,
            'seed_metrics': run_df,
            'summary': summary
        }
        rows.append(
            {
                'feature_set_name': feature_set_name,
                'feature_count': len(feature_info['feature_cols']),
                'selection_mode': 'anchor_sweep',
                'selection_eval_period': int(selection_eval_period),
                **summary
            }
        )
        log_line(
            f'[feature selection] {feature_set_name} mean macro_f1={summary["macro_f1_mean"]:.4f} '
            f'top1={summary["top_1_accuracy_mean"]:.4f} top3={summary["top_3_accuracy_mean"]:.4f} '
            f'mean_fit={summary["fit_seconds_mean"]:.2f}s'
        )

    feature_df = pd.DataFrame(rows).sort_values(
        ['macro_f1_mean', 'top_1_accuracy_mean', 'top_3_accuracy_mean'],
        ascending=False
    ).reset_index(drop=True)
    return feature_df, feature_results


def build_objective(train_df, valid_df, feature_info, task_type, devices, seed_list, verbose, selection_eval_period):
    train_ready = prep_single_label_cases(train_df, feature_info['feature_cols'])
    valid_ready = prep_single_label_cases(valid_df, feature_info['feature_cols'])

    def objective(trial):
        log_line(f'[optuna] Trial {trial.number} started')
        tuned = suggest_params(trial)
        run_df = evaluate_params_across_seeds(
            train_ready,
            valid_ready,
            feature_info,
            tuned,
            task_type=task_type,
            devices=devices,
            seed_list=seed_list,
            verbose=verbose,
            selection_eval_period=selection_eval_period,
            progress_label=f'optuna trial {trial.number}'
        )
        summary = summarize_seed_metrics(run_df)
        for key, value in summary.items():
            trial.set_user_attr(key, value)
        return float(run_df['macro_f1'].mean())

    return objective


def trial_callback(study, trial):
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return

    macro_f1 = float(trial.value)
    top_1 = trial.user_attrs.get('top_1_accuracy_mean')
    top_3 = trial.user_attrs.get('top_3_accuracy_mean')
    best_number = getattr(study.best_trial, 'number', None)
    best_value = getattr(study.best_trial, 'value', None)
    log_line(
        f'[optuna] Trial {trial.number} complete '
        f'macro_f1={macro_f1:.4f} top1={float(top_1):.4f} top3={float(top_3):.4f} '
        f'best_trial={best_number} best_macro_f1={float(best_value):.4f}'
    )


def build_trials_df(study):
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        rows.append(
            {
                'trial': trial.number,
                'value_macro_f1': round(float(trial.value), 4),
                'fit_seconds_mean': trial.user_attrs.get('fit_seconds_mean'),
                'fit_seconds_std': trial.user_attrs.get('fit_seconds_std'),
                'selected_iteration_mean': trial.user_attrs.get('selected_iteration_mean'),
                'selected_iteration_median': trial.user_attrs.get('selected_iteration_median'),
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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Run feature-set selection and focused CatBoost tuning for the single-label component benchmark'
    )
    parser.add_argument(
        '--input-path',
        default=None,
        help='Optional path to the single-label component case parquet or csv file'
    )
    parser.add_argument(
        '--task-type',
        choices=['CPU', 'GPU', 'cpu', 'gpu'],
        default='CPU',
        help='CatBoost processing target. CPU is the canonical local default'
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
        help='Number of Optuna trials to run after feature-set selection'
    )
    parser.add_argument(
        '--seed-list',
        default=DEFAULT_SEED_TEXT,
        help='Comma-separated seeds used to score each feature set and trial'
    )
    parser.add_argument(
        '--feature-selection-seed-list',
        default=None,
        help='Optional comma-separated seeds used only during feature-set selection'
    )
    parser.add_argument(
        '--feature-set',
        choices=sorted(BENCHMARK_FEATURE_SET_NAMES),
        default=None,
        help='Optional feature set name to use directly and skip the feature sweep'
    )
    parser.add_argument(
        '--selection-eval-period',
        type=int,
        default=DEF_SELECTION_EVAL_PERIOD,
        help='Tree interval used when scanning validation macro F1 during tuning runs'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Sampler seed for the Optuna study'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=0,
        help='CatBoost logging interval for trial runs'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use a lighter-weight local tuning preset with fewer seeds/trials and no feature sweep unless overridden'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    run_config = resolve_run_config(args)
    fit_plan = build_fit_plan(run_config)

    log_line('[setup] Loading single-label component cases')
    raw_df, input_path = load_frame(SINGLE_INPUT_STEM, input_path=args.input_path)
    train_df, valid_df, holdout_df, split_df = split_single_label_cases(
        prep_single_label_cases(raw_df, feature_manifest('core_structured')['feature_cols'])
    )
    log_line(f'[setup] Input path: {input_path}')
    log_line(
        f'[setup] Split rows | train={len(train_df):,} '
        f'valid_2025={len(valid_df):,} holdout_2026={len(holdout_df):,}'
    )
    log_line(
        f'[setup] Task type={run_config["task_type"]} '
        f'devices={run_config["devices"] if run_config["task_type"] == "GPU" else "cpu-only"}'
    )
    log_line(
        f'[setup] Tuning seeds={format_seed_list(run_config["seed_list"])} '
        f'feature selection seeds={format_seed_list(run_config["selection_seed_list"])} '
        f'selection_eval_period={run_config["selection_eval_period"]}'
    )
    if run_config['quick_notes']:
        for note in run_config['quick_notes']:
            log_line(f'[quick] {note}')
    log_line(
        f'[plan] Estimated CatBoost fits | '
        f'feature_selection={fit_plan["feature_selection_fits"]} '
        f'optuna={fit_plan["optuna_fits"]} '
        f'best_trial_rescore={fit_plan["best_trial_rescore_fits"]} '
        f'total={fit_plan["total_fits"]}'
    )
    log_line('[plan] holdout_2026 remains untouched during feature selection and tuning')

    log_line('')
    if run_config['run_feature_selection']:
        log_line('[phase 1/3] Feature-set selection started')
        feature_df, feature_results = evaluate_feature_sets(
            train_df,
            valid_df,
            task_type=run_config['task_type'],
            devices=run_config['devices'],
            seed_list=run_config['selection_seed_list'],
            verbose=args.verbose,
            selection_eval_period=run_config['selection_eval_period'],
            feature_set_names=run_config['feature_set_names']
        )
    else:
        log_line(f'[phase 1/3] Feature-set selection skipped by manual override -> {run_config["manual_feature_set"]}')
        feature_df, feature_results = build_manual_feature_selection(
            run_config['manual_feature_set'],
            selection_eval_period=run_config['selection_eval_period']
        )
    best_feature_name = str(feature_df.iloc[0]['feature_set_name'])
    best_feature_info = feature_results[best_feature_name]['feature_info']
    log_line(f'[phase 1/3] Selected feature set -> {best_feature_name}')

    log_line('')
    log_line(f'[phase 2/3] Starting Optuna study with {run_config["n_trials"]} trials')
    study = optuna.create_study(
        study_name='component_single_label_catboost_selection',
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=args.random_seed)
    )
    study.enqueue_trial({key: value for key, value in ANCHOR_PARAMS.items() if key not in {'bootstrap_type', 'border_count'}})
    study.enqueue_trial({key: value for key, value in ALT_ANCHOR_PARAMS.items() if key not in {'bootstrap_type', 'border_count'}})
    log_line('[phase 2/3] Optuna study created and anchor trials enqueued')

    objective = build_objective(
        train_df,
        valid_df,
        best_feature_info,
        task_type=run_config['task_type'],
        devices=run_config['devices'],
        seed_list=run_config['seed_list'],
        verbose=args.verbose,
        selection_eval_period=run_config['selection_eval_period']
    )
    study.optimize(
        objective,
        n_trials=run_config['n_trials'],
        show_progress_bar=True,
        callbacks=[trial_callback]
    )
    log_line(
        f'[phase 2/3] Best trial -> {study.best_trial.number} '
        f'macro_f1={float(study.best_value):.4f}'
    )

    trials_df = build_trials_df(study)
    best_params = {
        'bootstrap_type': 'Bernoulli',
        'border_count': 128
    }
    best_params.update(study.best_trial.params)

    train_ready = prep_single_label_cases(train_df, best_feature_info['feature_cols'])
    valid_ready = prep_single_label_cases(valid_df, best_feature_info['feature_cols'])
    log_line('')
    log_line('[phase 3/3] Rescoring best trial across tuning seeds')
    best_run_df = evaluate_params_across_seeds(
        train_ready,
        valid_ready,
        best_feature_info,
        best_params,
        task_type=run_config['task_type'],
        devices=run_config['devices'],
        seed_list=run_config['seed_list'],
        verbose=args.verbose,
        selection_eval_period=run_config['selection_eval_period'],
        progress_label='best trial rescore'
    )
    best_summary = summarize_seed_metrics(best_run_df)
    log_line(
        f'[phase 3/3] Best trial summary '
        f'macro_f1={best_summary["macro_f1_mean"]:.4f} '
        f'top1={best_summary["top_1_accuracy_mean"]:.4f} '
        f'top3={best_summary["top_3_accuracy_mean"]:.4f} '
        f'selected_iteration_median={best_summary["selected_iteration_median"]}'
    )
    best_metric_df = best_run_df.copy()
    best_metric_df.insert(0, 'split', 'valid_2025')
    best_metric_df.insert(0, 'stage', 'model_selection')
    best_metric_df.insert(0, 'feature_set_name', best_feature_name)

    manifest = {
        'artifact_role': 'model_selection',
        'target_scope': 'single_label_benchmark',
        'input_stem': SINGLE_INPUT_STEM,
        'input_path': str(input_path),
        'input_sha256': sha256_path(input_path),
        'code_version': {
            'git_head': get_git_head(),
            'git_dirty': get_git_dirty_flag()
        },
        'runtime': runtime_manifest(),
        'split_policy': {
            'train_end': str(TRAIN_END.date()),
            'valid_end': str(VALID_END.date()),
            'holdout_policy': 'holdout_2026 remains untouched during feature selection and tuning'
        },
        'metric_definitions': {
            'primary': 'macro_f1',
            'secondary': ['top_1_accuracy', 'top_3_accuracy'],
            'selected_iteration_rule': 'highest validation macro_f1 from staged CatBoost probabilities, then top_1_accuracy, then top_3_accuracy'
        },
        'selection_policy': {
            'feature_set_choice': (
                'highest mean validation macro_f1 across feature-selection seeds using anchor CatBoost params'
                if run_config['run_feature_selection']
                else f'manual override -> {best_feature_name}'
            ),
            'parameter_search': 'focused structured tuning around the CatBoost benchmark region',
            'seed_list': run_config['seed_list'],
            'feature_selection_seed_list': run_config['selection_seed_list'],
            'selection_eval_period': run_config['selection_eval_period']
        },
        'feature_sets': json_ready(feature_df.to_dict(orient='records')),
        'selected_feature_set': best_feature_name,
        'feature_manifest': best_feature_info,
        'n_trials': run_config['n_trials'],
        'task_type': run_config['task_type'],
        'devices': run_config['devices'] if run_config['task_type'] == 'GPU' else None,
        'study_name': study.study_name,
        'best_trial_number': int(study.best_trial.number),
        'best_params': best_params,
        'best_trial_attrs': study.best_trial.user_attrs,
        'selected_iteration': int(best_summary['selected_iteration_median']),
        'selection_metrics': best_summary,
        'fit_plan': fit_plan,
        'split_rows': {
            'train': int(len(train_df)),
            'valid_2025': int(len(valid_df)),
            'holdout_2026': int(len(holdout_df))
        }
    }

    features_path = OUTPUTS_DIR / FEATURES_NAME
    trials_path = OUTPUTS_DIR / TRIALS_NAME
    best_metric_path = OUTPUTS_DIR / BEST_METRIC_NAME
    manifest_path = OUTPUTS_DIR / MANIFEST_NAME

    feature_df.to_csv(features_path, index=False)
    trials_df.to_csv(trials_path, index=False)
    best_metric_df.to_csv(best_metric_path, index=False)
    write_json(manifest, manifest_path)

    print(f'[write] {features_path}')
    print(f'[write] {trials_path}')
    print(f'[write] {best_metric_path}')
    print(f'[write] {manifest_path}')
    print('')
    print('[done] Single-label component model selection finished')
    return 0


if __name__ == '__main__':
    sys.exit(main())
