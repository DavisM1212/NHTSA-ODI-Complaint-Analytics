import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import settings
from src.config.paths import OUTPUTS_DIR, ensure_project_directories
from src.modeling.component_common import (
    DEFAULT_SELECTION_SEEDS,
    FEATURE_WAVE1_SPLIT_MODE,
    MULTI_INPUT_STEM,
    SINGLE_INPUT_STEM,
    TARGET_COL,
    WAVE1_BUNDLE_DEFS,
    WAVE1_PAIRWISE_FAMILIES,
    WAVE1_PRUNE_PROTECTED,
    WAVE1_PRUNE_QUEUE,
    all_feature_columns,
    build_catboost_model,
    build_multiclass_calibration_df,
    build_multiclass_metric_row,
    compose_feature_manifest,
    fit_catboost_with_external_selection,
    get_git_dirty_flag,
    get_git_head,
    get_split_policy,
    json_ready,
    load_frame,
    parse_pipe_labels,
    prep_catboost_frames,
    prep_multi_label_cases,
    prep_single_label_cases,
    runtime_manifest,
    score_multiclass_from_proba,
    sha256_path,
    split_multi_label_cases_by_mode,
    split_single_label_cases_by_mode,
    subset_case_frame,
    write_json,
)
from src.modeling.component_multilabel import (
    CATBOOST_NAME,
    fit_catboost_holdout_with_fallback,
    fit_catboost_selection_with_fallback,
)
from src.modeling.component_multilabel import (
    build_metric_row as build_multilabel_metric_row,
)
from src.modeling.tune_component_catboost import (
    evaluate_params_across_seeds,
    summarize_seed_metrics,
)

# -----------------------------------------------------------------------------
# Output names
# -----------------------------------------------------------------------------
LOCKED_SINGLE_MANIFEST = OUTPUTS_DIR / 'component_single_label_benchmark_manifest.json'
LOCKED_SINGLE_SELECTION = OUTPUTS_DIR / 'component_single_label_selection_manifest.json'
LOCKED_SINGLE_CALIBRATION = OUTPUTS_DIR / 'component_single_label_holdout_calibration.csv'
LOCKED_MULTI_MANIFEST = OUTPUTS_DIR / 'component_multilabel_manifest.json'

GLOBAL_MANIFEST_NAME = 'component_featurewave1_manifest.json'
SINGLE_SCREEN_NAME = 'component_single_label_featurewave1_screen.csv'
SINGLE_SELECT_NAME = 'component_single_label_featurewave1_select.csv'
SINGLE_HOLDOUT_NAME = 'component_single_label_featurewave1_holdout.csv'
SINGLE_TRIALS_NAME = 'component_single_label_featurewave1_trials.csv'
SINGLE_SELECTION_NAME = 'component_single_label_featurewave1_selection_manifest.json'
SINGLE_CALIB_NAME = 'component_single_label_featurewave1_holdout_calibration.csv'
MULTI_SCREEN_NAME = 'component_multilabel_featurewave1_screen.csv'
MULTI_SELECT_NAME = 'component_multilabel_featurewave1_select.csv'
MULTI_HOLDOUT_NAME = 'component_multilabel_featurewave1_holdout.csv'
MULTI_LABEL_NAME = 'component_multilabel_featurewave1_label_metrics.csv'

# -----------------------------------------------------------------------------
# Wave defaults
# -----------------------------------------------------------------------------
FEATUREWAVE_TASK = 'feature_wave1'
SINGLE_PROMOTE_SELECT_DELTA = 0.010
SINGLE_PROMOTE_HOLDOUT_DELTA = 0.010
SINGLE_TOP3_DROP_LIMIT = 0.005
SINGLE_ECE_WORSE_LIMIT = 0.020
MULTI_PROMOTE_SELECT_DELTA = 0.015
MULTI_PROMOTE_HOLDOUT_MACRO_DELTA = 0.015
MULTI_PROMOTE_HOLDOUT_MICRO_DELTA = 0.010
MULTI_LABEL_COVERAGE_FLOOR = 0.80
MULTI_THRESHOLDS = [0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30]
MULTI_ITERATIONS = 1800
SINGLE_TUNE_TRIALS = 24
SINGLE_TUNE_ITER_MAX = 2200
SINGLE_SCREEN_EVAL_PERIOD = 25


def log_line(message=''):
    print(message, flush=True)


def log_family_start(task_name, stage_name, feature_name, run_idx, run_total):
    log_line(f'[{task_name}] {stage_name} {run_idx}/{run_total} -> {feature_name}')


def log_single_result(stage_name, feature_name, row, result):
    log_line(
        f'[single] {stage_name} {feature_name} done '
        f'fit={float(result["fit_seconds"]):.2f}s '
        f'select={float(result.get("selection_seconds", 0.0)):.2f}s '
        f'iter={int(result["selected_iteration"])} '
        f'macro_f1={float(row["macro_f1"]):.4f} '
        f'top1={float(row["top_1_accuracy"]):.4f} '
        f'top3={float(row["top_3_accuracy"]):.4f}'
    )


def log_multi_result(stage_name, feature_name, row, result):
    log_line(
        f'[multi] {stage_name} {feature_name} done '
        f'fit={float(result["fit_seconds"]):.2f}s '
        f'iter={int(result["selected_iteration"])} '
        f'threshold={float(result["selected_threshold"]):.3f} '
        f'macro_f1={float(row["macro_f1"]):.4f} '
        f'micro_f1={float(row["micro_f1"]):.4f} '
        f'recall@3={float(row["recall_at_3"]):.4f}'
    )


def load_json(path):
    path = Path(path)
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def build_feature_families():
    families = {
        'core_structured': compose_feature_manifest('core_structured')
    }
    for bundle_name, columns in WAVE1_BUNDLE_DEFS.items():
        family_name = f'wave1_{bundle_name}'
        families[family_name] = compose_feature_manifest(
            family_name,
            add_cols=columns
        )

    for family_name, bundle_names in WAVE1_PAIRWISE_FAMILIES.items():
        add_cols = []
        for bundle_name in bundle_names:
            add_cols.extend(WAVE1_BUNDLE_DEFS[bundle_name])
        families[family_name] = compose_feature_manifest(
            family_name,
            add_cols=add_cols
        )
    return families


def compact_list(values):
    values = [str(value) for value in values if str(value)]
    return '|'.join(values) if values else ''


def feature_row_base(task_name, feature_info, input_path, split_mode, hyperparam_policy):
    return {
        'task': task_name,
        'feature_set_name': feature_info['feature_set_name'],
        'feature_count': len(feature_info['feature_cols']),
        'added_cols': compact_list(feature_info.get('added_cols', [])),
        'removed_cols': compact_list(feature_info.get('removed_cols', [])),
        'split_mode': split_mode,
        'input_path': str(input_path),
        'input_sha256': sha256_path(input_path),
        'hyperparam_policy': hyperparam_policy
    }


def read_locked_single_ece():
    if not LOCKED_SINGLE_CALIBRATION.exists():
        return None
    calib_df = pd.read_csv(LOCKED_SINGLE_CALIBRATION)
    overall = calib_df.loc[calib_df['section'].eq('overall')]
    if overall.empty:
        return None
    return float(overall['ece'].iloc[0])


def sort_candidate_rows(rows, keys):
    if not rows:
        return []
    row_df = pd.DataFrame(rows)
    return row_df.sort_values(keys, ascending=False).to_dict(orient='records')


def build_pruned_feature_info(base_feature_info, drop_col):
    if drop_col in WAVE1_PRUNE_PROTECTED:
        return None
    if drop_col not in base_feature_info['feature_cols']:
        return None

    add_cols = list(base_feature_info.get('added_cols', []))
    remove_cols = list(base_feature_info.get('removed_cols', []))
    if drop_col not in remove_cols:
        remove_cols.append(drop_col)
    feature_name = f'{base_feature_info["feature_set_name"]}__drop_{drop_col}'
    return compose_feature_manifest(
        feature_name,
        add_cols=add_cols,
        remove_cols=remove_cols
    )


def select_best_row(row_df, ranking_cols):
    if row_df.empty:
        return None
    ranked = row_df.sort_values(ranking_cols, ascending=False).reset_index(drop=True)
    return ranked.iloc[0].to_dict()


def keep_if_better_or_equal(candidate_row, current_row, metric_cols):
    if current_row is None:
        return True
    candidate = tuple(float(candidate_row[column]) for column in metric_cols)
    current = tuple(float(current_row[column]) for column in metric_cols)
    return candidate >= current


# -----------------------------------------------------------------------------
# Single-label helpers
# -----------------------------------------------------------------------------
def score_single_family(train_df, eval_df, feature_info, params, task_type, devices, random_seed, selection_eval_period, input_path, stage_name):
    train_ready = subset_case_frame(train_df, feature_info['feature_cols'], target_col=TARGET_COL)
    eval_ready = subset_case_frame(eval_df, feature_info['feature_cols'], target_col=TARGET_COL)
    result = fit_catboost_with_external_selection(
        train_ready,
        eval_ready,
        feature_info,
        params,
        task_type=task_type,
        devices=devices,
        random_seed=random_seed,
        verbose=0,
        selection_eval_period=selection_eval_period,
        include_train_outputs=False,
        include_valid_outputs=False
    )
    row = {
        **feature_row_base(
            'single_label',
            feature_info,
            input_path,
            FEATURE_WAVE1_SPLIT_MODE,
            'fixed_locked'
        ),
        'stage': stage_name,
        'rows_train': int(len(train_ready)),
        'rows_eval': int(len(eval_ready)),
        'fit_seconds': result['fit_seconds'],
        'selected_iteration': int(result['selected_iteration']),
        'top_1_accuracy': float(result['valid_metrics']['top_1_accuracy']),
        'macro_f1': float(result['valid_metrics']['macro_f1']),
        'top_3_accuracy': float(result['valid_metrics']['top_3_accuracy'])
    }
    return row, result


def fit_single_holdout(train_df, holdout_df, feature_info, params, selected_iteration, task_type, devices, random_seed):
    train_ready = subset_case_frame(train_df, feature_info['feature_cols'], target_col=TARGET_COL)
    holdout_ready = subset_case_frame(holdout_df, feature_info['feature_cols'], target_col=TARGET_COL)
    X_train, X_holdout = prep_catboost_frames(train_ready, holdout_ready, feature_info)
    y_train = train_ready[TARGET_COL].copy()
    y_holdout = holdout_ready[TARGET_COL].copy()

    fixed_params = dict(params)
    fixed_params['iterations'] = int(selected_iteration)
    model = build_catboost_model(
        fixed_params,
        task_type=task_type,
        devices=devices,
        random_seed=random_seed,
        verbose=0
    )

    start = perf_counter()
    model.fit(
        X_train,
        y_train,
        cat_features=feature_info['cat_cols'],
        use_best_model=False
    )
    fit_seconds = round(perf_counter() - start, 2)

    holdout_proba = model.predict_proba(X_holdout)
    holdout_pred, holdout_metrics = score_multiclass_from_proba(
        y_holdout,
        holdout_proba,
        model.classes_
    )
    calibration_df = build_multiclass_calibration_df(
        y_holdout,
        holdout_proba,
        model.classes_
    )
    overall_ece = float(
        calibration_df.loc[calibration_df['section'].eq('overall'), 'ece'].iloc[0]
    )
    row = build_multiclass_metric_row(
        'CatBoost',
        'final_holdout',
        'holdout_2026',
        y_holdout,
        holdout_proba,
        model.classes_,
        fit_seconds=fit_seconds,
        selected_iteration=selected_iteration
    )
    return {
        'row': row,
        'pred': holdout_pred,
        'metrics': holdout_metrics,
        'calibration_df': calibration_df,
        'ece': overall_ece
    }


def suggest_single_wave1_params(trial, locked_params):
    return {
        'bootstrap_type': locked_params.get('bootstrap_type', 'Bernoulli'),
        'border_count': locked_params.get('border_count', 128),
        'iterations': trial.suggest_int('iterations', 1400, SINGLE_TUNE_ITER_MAX, step=200),
        'learning_rate': trial.suggest_float('learning_rate', 0.045, 0.10, log=True),
        'depth': trial.suggest_int('depth', 8, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 4.0, 12.0),
        'random_strength': trial.suggest_float('random_strength', 0.05, 0.6, log=True),
        'subsample': trial.suggest_float('subsample', 0.60, 0.80)
    }


def retune_single_family(train_df, valid_df, feature_info, locked_params, task_type, devices, seed_list, selection_eval_period, random_seed):
    train_ready = subset_case_frame(train_df, feature_info['feature_cols'], target_col=TARGET_COL)
    valid_ready = subset_case_frame(valid_df, feature_info['feature_cols'], target_col=TARGET_COL)

    def objective(trial):
        params = suggest_single_wave1_params(trial, locked_params)
        run_df = evaluate_params_across_seeds(
            train_ready,
            valid_ready,
            feature_info,
            params,
            task_type=task_type,
            devices=devices,
            seed_list=seed_list,
            verbose=0,
            selection_eval_period=selection_eval_period
        )
        summary = summarize_seed_metrics(run_df)
        for key, value in summary.items():
            trial.set_user_attr(key, value)
        return float(run_df['macro_f1'].mean())

    study = optuna.create_study(
        study_name='component_single_label_featurewave1',
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=random_seed)
    )
    enqueue_trial = {
        key: value
        for key, value in locked_params.items()
        if key not in {'bootstrap_type', 'border_count'}
    }
    if enqueue_trial:
        study.enqueue_trial(enqueue_trial)
    study.optimize(objective, n_trials=SINGLE_TUNE_TRIALS, show_progress_bar=False)

    best_params = {
        'bootstrap_type': locked_params.get('bootstrap_type', 'Bernoulli'),
        'border_count': locked_params.get('border_count', 128),
        **study.best_trial.params
    }
    best_run_df = evaluate_params_across_seeds(
        train_ready,
        valid_ready,
        feature_info,
        best_params,
        task_type=task_type,
        devices=devices,
        seed_list=seed_list,
        verbose=0,
        selection_eval_period=selection_eval_period
    )
    return {
        'study': study,
        'best_params': best_params,
        'selection_metrics': summarize_seed_metrics(best_run_df),
        'seed_metrics': best_run_df
    }


def run_single_wave(args, feature_families, locked_single_manifest, locked_single_selection):
    screen_rows = []
    select_rows = []
    holdout_rows = []

    raw_df, input_path = load_frame(SINGLE_INPUT_STEM, input_path=args.single_input_path)
    case_df = prep_single_label_cases(raw_df, all_feature_columns())
    split_parts = split_single_label_cases_by_mode(case_df, split_mode=FEATURE_WAVE1_SPLIT_MODE)
    policy = get_split_policy(FEATURE_WAVE1_SPLIT_MODE)

    train_core_df = split_parts[policy['train_name']]
    screen_df = split_parts[policy['screen_name']]
    dev_screen_df = split_parts[policy['select_train_name']]
    select_df = split_parts[policy['select_name']]
    dev_select_df = split_parts[policy['dev_name']]
    holdout_df = split_parts[policy['holdout_name']]

    locked_params = locked_single_selection['best_params']
    locked_eval_period = int(
        locked_single_selection.get('selection_policy', {}).get('selection_eval_period', 10)
    )
    screen_eval_period = int(args.single_screen_eval_period) if args.single_screen_eval_period else max(
        SINGLE_SCREEN_EVAL_PERIOD,
        locked_eval_period
    )
    retune_eval_period = int(args.single_retune_eval_period) if args.single_retune_eval_period else locked_eval_period

    log_line(
        f'[single] Split rows | train_core={len(train_core_df):,} screen_2024={len(screen_df):,} '
        f'select_2025={len(select_df):,} holdout_2026={len(holdout_df):,}'
    )
    log_line(
        f'[single] Eval periods | screen/select={screen_eval_period} retune={retune_eval_period}'
    )

    log_family_start('single', 'screen_2024', 'core_structured', 1, 5)
    core_family = feature_families['core_structured']
    core_screen_row, _ = score_single_family(
        train_core_df,
        screen_df,
        core_family,
        locked_params,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed,
        selection_eval_period=screen_eval_period,
        input_path=input_path,
        stage_name='screen_2024'
    )
    screen_rows.append(core_screen_row)
    log_line(
        f'[single] screen_2024 core_structured done '
        f'macro_f1={float(core_screen_row["macro_f1"]):.4f} '
        f'top1={float(core_screen_row["top_1_accuracy"]):.4f} '
        f'top3={float(core_screen_row["top_3_accuracy"]):.4f}'
    )

    bundle_rows = []
    screen_family_names = [
        'wave1_incident_bundle',
        'wave1_date_quality_bundle',
        'wave1_geo_time_bundle',
        'wave1_cohort_history_bundle'
    ]
    for family_idx, family_name in enumerate(screen_family_names, start=2):
        log_family_start('single', 'screen_2024', family_name, family_idx, 5)
        row, _ = score_single_family(
            train_core_df,
            screen_df,
            feature_families[family_name],
            locked_params,
            task_type=str(args.task_type).upper(),
            devices=args.devices,
            random_seed=args.random_seed,
            selection_eval_period=screen_eval_period,
            input_path=input_path,
            stage_name='screen_2024'
        )
        bundle_rows.append(row)
        screen_rows.append(row)
        log_line(
            f'[single] screen_2024 {family_name} done '
            f'macro_f1={float(row["macro_f1"]):.4f} '
            f'top1={float(row["top_1_accuracy"]):.4f} '
            f'top3={float(row["top_3_accuracy"]):.4f}'
        )

    bundle_ranked = sort_candidate_rows(
        bundle_rows,
        ['macro_f1', 'top_1_accuracy', 'top_3_accuracy']
    )
    top_bundle_names = [row['feature_set_name'] for row in bundle_ranked[:3]]

    log_family_start('single', 'select_2025', 'core_structured', 1, 7)
    core_select_row, core_select_result = score_single_family(
        dev_screen_df,
        select_df,
        core_family,
        locked_params,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed,
        selection_eval_period=screen_eval_period,
        input_path=input_path,
        stage_name='select_2025'
    )
    select_rows.append(core_select_row)
    log_single_result('select_2025', 'core_structured', core_select_row, core_select_result)

    candidate_rows = []
    candidate_info_lookup = {}
    for family_idx, family_name in enumerate(top_bundle_names, start=2):
        log_family_start('single', 'select_2025', family_name, family_idx, 7)
        row, result = score_single_family(
            dev_screen_df,
            select_df,
            feature_families[family_name],
            locked_params,
            task_type=str(args.task_type).upper(),
            devices=args.devices,
            random_seed=args.random_seed,
            selection_eval_period=screen_eval_period,
            input_path=input_path,
            stage_name='select_2025'
        )
        select_rows.append(row)
        candidate_rows.append(row)
        candidate_info_lookup[family_name] = feature_families[family_name]
        log_single_result('select_2025', family_name, row, result)

    for family_idx, family_name in enumerate(WAVE1_PAIRWISE_FAMILIES, start=5):
        log_family_start('single', 'select_2025', family_name, family_idx, 7)
        row, result = score_single_family(
            dev_screen_df,
            select_df,
            feature_families[family_name],
            locked_params,
            task_type=str(args.task_type).upper(),
            devices=args.devices,
            random_seed=args.random_seed,
            selection_eval_period=screen_eval_period,
            input_path=input_path,
            stage_name='select_2025'
        )
        select_rows.append(row)
        candidate_rows.append(row)
        candidate_info_lookup[family_name] = feature_families[family_name]
        log_single_result('select_2025', family_name, row, result)

    best_select_row = select_best_row(
        pd.DataFrame(candidate_rows),
        ['macro_f1', 'top_1_accuracy', 'top_3_accuracy']
    )
    best_feature_info = candidate_info_lookup[best_select_row['feature_set_name']]
    current_best_row = best_select_row

    for prune_col in WAVE1_PRUNE_QUEUE:
        pruned_info = build_pruned_feature_info(best_feature_info, prune_col)
        if pruned_info is None:
            continue
        log_line(f'[single] select_prune -> drop {prune_col}')
        row, result = score_single_family(
            dev_screen_df,
            select_df,
            pruned_info,
            locked_params,
            task_type=str(args.task_type).upper(),
            devices=args.devices,
            random_seed=args.random_seed,
            selection_eval_period=screen_eval_period,
            input_path=input_path,
            stage_name='select_prune'
        )
        select_rows.append(row)
        log_single_result('select_prune', pruned_info['feature_set_name'], row, result)
        if keep_if_better_or_equal(row, current_best_row, ['macro_f1', 'top_1_accuracy', 'top_3_accuracy']):
            current_best_row = row
            best_feature_info = pruned_info
            candidate_info_lookup[pruned_info['feature_set_name']] = pruned_info
            continue
        break

    select_improvement = float(current_best_row['macro_f1'] - core_select_row['macro_f1'])
    select_top3_drop = float(core_select_row['top_3_accuracy'] - current_best_row['top_3_accuracy'])
    select_gate_pass = (
        select_improvement >= SINGLE_PROMOTE_SELECT_DELTA
        and select_top3_drop <= SINGLE_TOP3_DROP_LIMIT
    )

    single_trials_df = pd.DataFrame()
    single_calibration_df = pd.DataFrame()
    holdout_result = None
    promotion_status = 'rejected_select'
    tuned_manifest = None

    if select_gate_pass:
        log_line(f'[single] Retuning promoted select family -> {best_feature_info["feature_set_name"]}')
        retune = retune_single_family(
            dev_screen_df,
            select_df,
            best_feature_info,
            locked_params,
            task_type=str(args.task_type).upper(),
            devices=args.devices,
            seed_list=DEFAULT_SELECTION_SEEDS,
            selection_eval_period=retune_eval_period,
            random_seed=args.random_seed
        )
        single_trials_df = pd.DataFrame(
            [
                {
                    'trial': trial.number,
                    'macro_f1_mean': round(float(trial.value), 4),
                    **trial.params,
                    **trial.user_attrs
                }
                for trial in retune['study'].trials
                if trial.state == optuna.trial.TrialState.COMPLETE
            ]
        ).sort_values(['macro_f1_mean'], ascending=False).reset_index(drop=True)
        selected_iteration = int(retune['selection_metrics']['selected_iteration_median'])
        tuned_manifest = {
            'artifact_role': 'model_selection',
            'target_scope': 'feature_wave1_single_label',
            'feature_manifest': best_feature_info,
            'best_params': retune['best_params'],
            'selected_iteration': selected_iteration,
            'selection_metrics': retune['selection_metrics'],
            'selection_policy': {
                'split_mode': FEATURE_WAVE1_SPLIT_MODE,
                'n_trials': SINGLE_TUNE_TRIALS,
                'seed_list': DEFAULT_SELECTION_SEEDS,
                'iteration_cap': SINGLE_TUNE_ITER_MAX,
                'selection_eval_period': retune_eval_period
            }
        }
        holdout_result = fit_single_holdout(
            dev_select_df,
            holdout_df,
            best_feature_info,
            retune['best_params'],
            selected_iteration,
            task_type=str(args.task_type).upper(),
            devices=args.devices,
            random_seed=args.random_seed
        )
        single_calibration_df = holdout_result['calibration_df']
        holdout_row = {
            **feature_row_base(
                'single_label',
                best_feature_info,
                input_path,
                FEATURE_WAVE1_SPLIT_MODE,
                'retuned_final'
            ),
            **holdout_result['row']
        }
        holdout_rows.append(holdout_row)
        locked_ece = read_locked_single_ece()
        holdout_macro_gain = float(
            holdout_row['macro_f1'] - locked_single_manifest['official_holdout_metrics']['macro_f1']
        )
        ece_ok = True if locked_ece is None else holdout_result['ece'] <= locked_ece + SINGLE_ECE_WORSE_LIMIT
        top3_ok = holdout_row['top_3_accuracy'] >= (
            locked_single_manifest['official_holdout_metrics']['top_3_accuracy'] - SINGLE_TOP3_DROP_LIMIT
        )
        promotion_status = 'promoted' if (
            holdout_macro_gain >= SINGLE_PROMOTE_HOLDOUT_DELTA and top3_ok and ece_ok
        ) else 'rejected_holdout'

    return {
        'input_path': str(input_path),
        'split_df': split_parts['split_df'],
        'screen_df': pd.DataFrame(screen_rows),
        'select_df': pd.DataFrame(select_rows),
        'holdout_df': pd.DataFrame(holdout_rows),
        'trials_df': single_trials_df,
        'selection_manifest': tuned_manifest,
        'calibration_df': single_calibration_df,
        'selected_feature': current_best_row['feature_set_name'],
        'select_metrics': current_best_row,
        'select_baseline': core_select_row,
        'promotion_status': promotion_status,
        'select_gate_pass': select_gate_pass
    }


# -----------------------------------------------------------------------------
# Multi-label helpers
# -----------------------------------------------------------------------------
def build_multilabel_targets(frame):
    return parse_pipe_labels(frame['component_groups'])


def build_multilabel_encoded(train_labels, eval_labels):
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_labels)
    y_eval = mlb.transform(eval_labels)
    return mlb, y_train, y_eval


def check_unseen_multilabel_labels(train_labels, eval_labels, split_name):
    train_set = {label for labels in train_labels for label in labels}
    unseen = sorted({label for labels in eval_labels for label in labels if label not in train_set})
    if unseen:
        unseen_text = ', '.join(unseen)
        raise ValueError(f'{split_name} has unseen target labels: {unseen_text}')


def score_multi_family(train_df, eval_df, feature_info, task_type, devices, random_seed, input_path, stage_name):
    train_labels = build_multilabel_targets(train_df)
    eval_labels = build_multilabel_targets(eval_df)
    check_unseen_multilabel_labels(train_labels, eval_labels, stage_name)
    _, y_train, y_eval = build_multilabel_encoded(train_labels, eval_labels)
    result = fit_catboost_selection_with_fallback(
        train_df,
        eval_df,
        y_train,
        y_eval,
        feature_info,
        task_type=task_type,
        devices=devices,
        random_seed=random_seed,
        verbose=0,
        iterations=MULTI_ITERATIONS,
        eval_period=25,
        thresholds=MULTI_THRESHOLDS,
        min_positive_labels=1
    )
    row = {
        **feature_row_base(
            'multi_label',
            feature_info,
            input_path,
            FEATURE_WAVE1_SPLIT_MODE,
            'fixed_locked'
        ),
        **build_multilabel_metric_row(
            CATBOOST_NAME,
            stage_name,
            stage_name,
            y_eval,
            result['valid_pred'],
            result['valid_proba'],
            threshold=result['selected_threshold'],
            fit_seconds=result['fit_seconds'],
            selected_iteration=result['selected_iteration']
        ),
        'actual_task_type': result['actual_task_type']
    }
    return row, result


def fit_multi_holdout(dev_df, holdout_df, feature_info, selection_result, random_seed, devices):
    dev_labels = build_multilabel_targets(dev_df)
    holdout_labels = build_multilabel_targets(holdout_df)
    check_unseen_multilabel_labels(dev_labels, holdout_labels, 'Holdout split')
    mlb, y_dev, y_holdout = build_multilabel_encoded(dev_labels, holdout_labels)
    holdout = fit_catboost_holdout_with_fallback(
        dev_df,
        holdout_df,
        y_dev,
        feature_info,
        task_type=selection_result['actual_task_type'],
        devices=devices,
        random_seed=random_seed,
        verbose=0,
        selected_iteration=selection_result['selected_iteration'],
        selected_threshold=selection_result['selected_threshold'],
        min_positive_labels=1
    )
    row = build_multilabel_metric_row(
        CATBOOST_NAME,
        'final_holdout',
        'holdout_2026',
        y_holdout,
        holdout['holdout_pred'],
        holdout['holdout_proba'],
        threshold=holdout['selected_threshold'],
        fit_seconds=holdout['fit_seconds'],
        selected_iteration=holdout['selected_iteration']
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_holdout,
        holdout['holdout_pred'],
        average=None,
        zero_division=0
    )
    label_df = pd.DataFrame(
        {
            'component_group': mlb.classes_,
            'support': support,
            'precision': np.round(precision, 4),
            'recall': np.round(recall, 4),
            'f1': np.round(f1, 4)
        }
    ).sort_values(['support', 'f1'], ascending=[False, False]).reset_index(drop=True)
    return {
        'row': row,
        'label_df': label_df
    }


def run_multi_wave(args, feature_families, locked_multi_manifest):
    screen_rows = []
    select_rows = []
    holdout_rows = []

    raw_df, input_path = load_frame(MULTI_INPUT_STEM, input_path=args.multi_input_path)
    case_df = prep_multi_label_cases(raw_df, all_feature_columns())
    split_parts = split_multi_label_cases_by_mode(case_df, split_mode=FEATURE_WAVE1_SPLIT_MODE)
    policy = get_split_policy(FEATURE_WAVE1_SPLIT_MODE)

    train_core_df = split_parts[policy['train_name']]
    screen_df = split_parts[policy['screen_name']]
    dev_screen_df = split_parts[policy['select_train_name']]
    select_df = split_parts[policy['select_name']]
    dev_select_df = split_parts[policy['dev_name']]
    holdout_df = split_parts[policy['holdout_name']]

    log_line(
        f'[multi] Split rows | train_core={len(train_core_df):,} screen_2024={len(screen_df):,} '
        f'select_2025={len(select_df):,} holdout_2026={len(holdout_df):,}'
    )

    core_family = feature_families['core_structured']
    log_family_start('multi', 'screen_2024', 'core_structured', 1, 5)
    core_screen_row, _ = score_multi_family(
        train_core_df,
        screen_df,
        core_family,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed,
        input_path=input_path,
        stage_name='screen_2024'
    )
    screen_rows.append(core_screen_row)
    log_line(
        f'[multi] screen_2024 core_structured done '
        f'macro_f1={float(core_screen_row["macro_f1"]):.4f} '
        f'micro_f1={float(core_screen_row["micro_f1"]):.4f} '
        f'recall@3={float(core_screen_row["recall_at_3"]):.4f}'
    )

    bundle_rows = []
    screen_family_names = [
        'wave1_incident_bundle',
        'wave1_date_quality_bundle',
        'wave1_geo_time_bundle',
        'wave1_cohort_history_bundle'
    ]
    for family_idx, family_name in enumerate(screen_family_names, start=2):
        log_family_start('multi', 'screen_2024', family_name, family_idx, 5)
        row, _ = score_multi_family(
            train_core_df,
            screen_df,
            feature_families[family_name],
            task_type=str(args.task_type).upper(),
            devices=args.devices,
            random_seed=args.random_seed,
            input_path=input_path,
            stage_name='screen_2024'
        )
        bundle_rows.append(row)
        screen_rows.append(row)
        log_line(
            f'[multi] screen_2024 {family_name} done '
            f'macro_f1={float(row["macro_f1"]):.4f} '
            f'micro_f1={float(row["micro_f1"]):.4f} '
            f'recall@3={float(row["recall_at_3"]):.4f}'
        )

    bundle_ranked = sort_candidate_rows(
        bundle_rows,
        ['macro_f1', 'micro_f1', 'recall_at_3', 'precision_at_3']
    )
    top_bundle_names = [row['feature_set_name'] for row in bundle_ranked[:3]]

    log_family_start('multi', 'select_2025', 'core_structured', 1, 7)
    core_select_row, core_select_result = score_multi_family(
        dev_screen_df,
        select_df,
        core_family,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed,
        input_path=input_path,
        stage_name='select_2025'
    )
    select_rows.append(core_select_row)
    log_multi_result('select_2025', 'core_structured', core_select_row, core_select_result)

    candidate_rows = []
    candidate_info_lookup = {}
    candidate_selection_lookup = {}
    for family_idx, family_name in enumerate(top_bundle_names, start=2):
        log_family_start('multi', 'select_2025', family_name, family_idx, 7)
        row, result = score_multi_family(
            dev_screen_df,
            select_df,
            feature_families[family_name],
            task_type=str(args.task_type).upper(),
            devices=args.devices,
            random_seed=args.random_seed,
            input_path=input_path,
            stage_name='select_2025'
        )
        select_rows.append(row)
        candidate_rows.append(row)
        candidate_info_lookup[family_name] = feature_families[family_name]
        candidate_selection_lookup[family_name] = result
        log_multi_result('select_2025', family_name, row, result)

    for family_idx, family_name in enumerate(WAVE1_PAIRWISE_FAMILIES, start=5):
        log_family_start('multi', 'select_2025', family_name, family_idx, 7)
        row, result = score_multi_family(
            dev_screen_df,
            select_df,
            feature_families[family_name],
            task_type=str(args.task_type).upper(),
            devices=args.devices,
            random_seed=args.random_seed,
            input_path=input_path,
            stage_name='select_2025'
        )
        select_rows.append(row)
        candidate_rows.append(row)
        candidate_info_lookup[family_name] = feature_families[family_name]
        candidate_selection_lookup[family_name] = result
        log_multi_result('select_2025', family_name, row, result)

    best_select_row = select_best_row(
        pd.DataFrame(candidate_rows),
        ['macro_f1', 'micro_f1', 'recall_at_3', 'precision_at_3']
    )
    best_feature_info = candidate_info_lookup[best_select_row['feature_set_name']]
    best_selection_result = candidate_selection_lookup[best_select_row['feature_set_name']]
    current_best_row = best_select_row

    for prune_col in WAVE1_PRUNE_QUEUE:
        pruned_info = build_pruned_feature_info(best_feature_info, prune_col)
        if pruned_info is None:
            continue
        log_line(f'[multi] select_prune -> drop {prune_col}')
        row, result = score_multi_family(
            dev_screen_df,
            select_df,
            pruned_info,
            task_type=str(args.task_type).upper(),
            devices=args.devices,
            random_seed=args.random_seed,
            input_path=input_path,
            stage_name='select_prune'
        )
        select_rows.append(row)
        log_multi_result('select_prune', pruned_info['feature_set_name'], row, result)
        if keep_if_better_or_equal(row, current_best_row, ['macro_f1', 'micro_f1', 'recall_at_3', 'precision_at_3']):
            current_best_row = row
            best_feature_info = pruned_info
            best_selection_result = result
            continue
        break

    select_improvement = float(current_best_row['macro_f1'] - core_select_row['macro_f1'])
    select_gate_pass = select_improvement >= MULTI_PROMOTE_SELECT_DELTA

    multi_label_df = pd.DataFrame()
    promotion_status = 'rejected_select'
    if select_gate_pass:
        holdout_result = fit_multi_holdout(
            dev_select_df,
            holdout_df,
            best_feature_info,
            best_selection_result,
            random_seed=args.random_seed,
            devices=args.devices
        )
        holdout_row = {
            **feature_row_base(
                'multi_label',
                best_feature_info,
                input_path,
                FEATURE_WAVE1_SPLIT_MODE,
                'fixed_locked'
            ),
            **holdout_result['row']
        }
        holdout_rows.append(holdout_row)
        multi_label_df = holdout_result['label_df']

        locked_holdout = locked_multi_manifest['official_holdout_metrics']
        macro_gain = float(holdout_row['macro_f1'] - locked_holdout['macro_f1'])
        micro_gain = float(holdout_row['micro_f1'] - locked_holdout['micro_f1'])
        promotion_status = 'promoted' if (
            (macro_gain >= MULTI_PROMOTE_HOLDOUT_MACRO_DELTA or micro_gain >= MULTI_PROMOTE_HOLDOUT_MICRO_DELTA)
            and holdout_row['recall_at_3'] >= locked_holdout['recall_at_3']
            and holdout_row['label_coverage'] >= MULTI_LABEL_COVERAGE_FLOOR
        ) else 'rejected_holdout'

    return {
        'input_path': str(input_path),
        'split_df': split_parts['split_df'],
        'screen_df': pd.DataFrame(screen_rows),
        'select_df': pd.DataFrame(select_rows),
        'holdout_df': pd.DataFrame(holdout_rows),
        'label_df': multi_label_df,
        'selected_feature': current_best_row['feature_set_name'],
        'select_metrics': current_best_row,
        'select_baseline': core_select_row,
        'promotion_status': promotion_status,
        'select_gate_pass': select_gate_pass
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Run structured feature wave 1 screening and promotion for the component benchmarks'
    )
    parser.add_argument(
        '--task-type',
        choices=['CPU', 'GPU', 'cpu', 'gpu'],
        default='CPU',
        help='CatBoost processing target'
    )
    parser.add_argument(
        '--devices',
        default='0',
        help='GPU device string for CatBoost when task_type is GPU'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=settings.RANDOM_SEED
    )
    parser.add_argument(
        '--single-input-path',
        default=None
    )
    parser.add_argument(
        '--multi-input-path',
        default=None
    )
    parser.add_argument(
        '--skip-single',
        action='store_true'
    )
    parser.add_argument(
        '--skip-multi',
        action='store_true'
    )
    parser.add_argument(
        '--single-screen-eval-period',
        type=int,
        default=None,
        help='External CatBoost selection scan interval for single-label screen/select scoring'
    )
    parser.add_argument(
        '--single-retune-eval-period',
        type=int,
        default=None,
        help='External CatBoost selection scan interval for single-label retuning'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    if args.skip_single and args.skip_multi:
        raise ValueError('Nothing to do: both single and multi tasks were skipped')

    feature_families = build_feature_families()
    locked_single_manifest = load_json(LOCKED_SINGLE_MANIFEST)
    locked_single_selection = load_json(LOCKED_SINGLE_SELECTION)
    locked_multi_manifest = load_json(LOCKED_MULTI_MANIFEST)

    global_manifest = {
        'artifact_role': FEATUREWAVE_TASK,
        'feature_wave': 1,
        'split_mode': FEATURE_WAVE1_SPLIT_MODE,
        'runtime': runtime_manifest(),
        'code_version': {
            'git_head': get_git_head(),
            'git_dirty': get_git_dirty_flag()
        },
        'feature_families': json_ready(
            {
                name: {
                    'added_cols': info.get('added_cols', []),
                    'removed_cols': info.get('removed_cols', []),
                    'feature_count': len(info['feature_cols'])
                }
                for name, info in feature_families.items()
            }
        ),
        'tasks': {}
    }

    if not args.skip_single:
        log_line('[run] Single-label structured feature wave 1')
        single_result = run_single_wave(
            args,
            feature_families,
            locked_single_manifest,
            locked_single_selection
        )
        single_result['screen_df'].to_csv(OUTPUTS_DIR / SINGLE_SCREEN_NAME, index=False)
        single_result['select_df'].to_csv(OUTPUTS_DIR / SINGLE_SELECT_NAME, index=False)
        single_result['holdout_df'].to_csv(OUTPUTS_DIR / SINGLE_HOLDOUT_NAME, index=False)
        if not single_result['trials_df'].empty:
            single_result['trials_df'].to_csv(OUTPUTS_DIR / SINGLE_TRIALS_NAME, index=False)
        if single_result['selection_manifest'] is not None:
            write_json(single_result['selection_manifest'], OUTPUTS_DIR / SINGLE_SELECTION_NAME)
        if not single_result['calibration_df'].empty:
            single_result['calibration_df'].to_csv(OUTPUTS_DIR / SINGLE_CALIB_NAME, index=False)

        global_manifest['tasks']['single_label'] = {
            'input_path': single_result['input_path'],
            'selected_feature': single_result['selected_feature'],
            'select_metrics': single_result['select_metrics'],
            'select_baseline': single_result['select_baseline'],
            'promotion_status': single_result['promotion_status'],
            'select_gate_pass': single_result['select_gate_pass'],
            'artifacts': {
                'screen': str(OUTPUTS_DIR / SINGLE_SCREEN_NAME),
                'select': str(OUTPUTS_DIR / SINGLE_SELECT_NAME),
                'holdout': str(OUTPUTS_DIR / SINGLE_HOLDOUT_NAME),
                'trials': str(OUTPUTS_DIR / SINGLE_TRIALS_NAME),
                'selection_manifest': str(OUTPUTS_DIR / SINGLE_SELECTION_NAME),
                'holdout_calibration': str(OUTPUTS_DIR / SINGLE_CALIB_NAME)
            }
        }

    if not args.skip_multi:
        log_line('[run] Multi-label structured feature wave 1')
        multi_result = run_multi_wave(
            args,
            feature_families,
            locked_multi_manifest
        )
        multi_result['screen_df'].to_csv(OUTPUTS_DIR / MULTI_SCREEN_NAME, index=False)
        multi_result['select_df'].to_csv(OUTPUTS_DIR / MULTI_SELECT_NAME, index=False)
        multi_result['holdout_df'].to_csv(OUTPUTS_DIR / MULTI_HOLDOUT_NAME, index=False)
        if not multi_result['label_df'].empty:
            multi_result['label_df'].to_csv(OUTPUTS_DIR / MULTI_LABEL_NAME, index=False)

        global_manifest['tasks']['multi_label'] = {
            'input_path': multi_result['input_path'],
            'selected_feature': multi_result['selected_feature'],
            'select_metrics': multi_result['select_metrics'],
            'select_baseline': multi_result['select_baseline'],
            'promotion_status': multi_result['promotion_status'],
            'select_gate_pass': multi_result['select_gate_pass'],
            'artifacts': {
                'screen': str(OUTPUTS_DIR / MULTI_SCREEN_NAME),
                'select': str(OUTPUTS_DIR / MULTI_SELECT_NAME),
                'holdout': str(OUTPUTS_DIR / MULTI_HOLDOUT_NAME),
                'label_metrics': str(OUTPUTS_DIR / MULTI_LABEL_NAME)
            }
        }

    write_json(global_manifest, OUTPUTS_DIR / GLOBAL_MANIFEST_NAME)
    print(f'[write] {OUTPUTS_DIR / GLOBAL_MANIFEST_NAME}')
    print('[done] Structured feature wave 1 finished')
    return 0


if __name__ == '__main__':
    sys.exit(main())
