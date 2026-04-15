from time import perf_counter

import numpy as np
from catboost import CatBoostClassifier, CatBoostError

from src.modeling.component_common import (
    MAX_TOP_K,
    apply_multilabel_threshold,
    prep_catboost_frames,
    score_multilabel_predictions,
    select_multilabel_threshold,
)

# Shared multilabel CatBoost helpers reused by entrypoints
# Read src/modeling/README.md or component_multilabel.py first for workflow order


CATBOOST_NAME = 'CatBoost MultiLabel'
DEF_THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
DEF_CATBOOST_ITERS = 1200
DEF_CATBOOST_EVAL_PERIOD = 25
DEF_MIN_POSITIVE_LABELS = 1
CATBOOST_PARAMS = {
    'bootstrap_type': 'Bernoulli',
    'border_count': 128,
    'depth': 8,
    'iterations': DEF_CATBOOST_ITERS,
    'l2_leaf_reg': 6.0,
    'learning_rate': 0.08,
    'subsample': 0.8
}


def log_line(message=''):
    print(message, flush=True)


def build_metric_row(model_name, stage_name, split_name, y_true, y_pred, proba, threshold=np.nan, fit_seconds=np.nan, selected_iteration=np.nan):
    return {
        'model': model_name,
        'stage': stage_name,
        'split': split_name,
        'rows': int(len(y_true)),
        'fit_seconds': fit_seconds,
        'selected_iteration': selected_iteration,
        'threshold': threshold,
        **score_multilabel_predictions(y_true, y_pred, proba, top_k=MAX_TOP_K)
    }


def build_catboost_model(task_type, devices, random_seed, verbose, iterations):
    task_type = str(task_type).upper().strip()
    if task_type not in {'CPU', 'GPU'}:
        raise ValueError("task_type must be either 'CPU' or 'GPU'")

    params = {
        'loss_function': 'MultiLogloss',
        'eval_metric': 'MultiLogloss',
        'has_time': True,
        'grow_policy': 'SymmetricTree',
        'allow_writing_files': False,
        'random_seed': int(random_seed),
        'task_type': task_type,
        'verbose': int(verbose),
        **CATBOOST_PARAMS
    }
    params['iterations'] = int(iterations)

    if task_type == 'GPU':
        params['devices'] = str(devices)

    return CatBoostClassifier(**params)


def is_retryable_catboost_gpu_error(exc):
    text = str(exc)
    retry_markers = [
        'catboost/cuda/',
        'gpu_metrics.cpp',
        'approx[dim].size() == target.size()'
    ]
    return any(marker in text for marker in retry_markers)


def select_best_multilabel_threshold(y_true, proba, thresholds, min_positive_labels):
    return select_multilabel_threshold(
        y_true,
        proba,
        thresholds=list(thresholds),
        min_positive_labels=min_positive_labels
    )


def select_catboost_iteration(model, X_valid, y_valid, eval_period, thresholds, min_positive_labels):
    best = None
    total_trees = int(model.tree_count_)
    eval_period = max(int(eval_period), 1)

    for step_idx, proba in enumerate(model.staged_predict_proba(X_valid, eval_period=eval_period), start=1):
        current_iteration = min(step_idx * eval_period, total_trees)
        threshold_choice = select_best_multilabel_threshold(
            y_valid,
            proba,
            thresholds=thresholds,
            min_positive_labels=min_positive_labels
        )
        candidate = {
            'selected_iteration': current_iteration,
            **threshold_choice
        }
        ranking = (
            candidate['macro_f1'],
            candidate['micro_f1'],
            candidate['recall_at_3'],
            candidate['precision_at_3'],
            -candidate['threshold'],
            -candidate['selected_iteration']
        )

        if best is None:
            best = candidate
            continue

        best_ranking = (
            best['macro_f1'],
            best['micro_f1'],
            best['recall_at_3'],
            best['precision_at_3'],
            -best['threshold'],
            -best['selected_iteration']
        )
        if ranking > best_ranking:
            best = candidate

    if best is None:
        raise ValueError('Unable to select a CatBoost multi-label iteration')
    return best


def fit_catboost_selection_stage(train_df, valid_df, y_train, y_valid, feature_info, task_type, devices, random_seed, verbose, iterations, eval_period, thresholds, min_positive_labels):
    X_train, X_valid = prep_catboost_frames(train_df, valid_df, feature_info)

    model = build_catboost_model(
        task_type=task_type,
        devices=devices,
        random_seed=random_seed,
        verbose=verbose,
        iterations=iterations
    )
    start = perf_counter()
    model.fit(
        X_train,
        y_train,
        cat_features=feature_info['cat_cols'],
        use_best_model=False
    )
    fit_seconds = round(perf_counter() - start, 2)

    best = select_catboost_iteration(
        model,
        X_valid,
        y_valid,
        eval_period=eval_period,
        thresholds=thresholds,
        min_positive_labels=min_positive_labels
    )
    selected_iteration = int(best['selected_iteration'])
    selected_threshold = float(best['threshold'])
    train_proba = model.predict_proba(X_train, ntree_end=selected_iteration)
    valid_proba = model.predict_proba(X_valid, ntree_end=selected_iteration)
    train_pred = apply_multilabel_threshold(
        train_proba,
        selected_threshold,
        min_positive_labels=min_positive_labels
    )
    valid_pred = apply_multilabel_threshold(
        valid_proba,
        selected_threshold,
        min_positive_labels=min_positive_labels
    )

    return {
        'model': model,
        'fit_seconds': fit_seconds,
        'selected_iteration': selected_iteration,
        'selected_threshold': selected_threshold,
        'selection_metrics': best,
        'train_pred': train_pred,
        'train_proba': train_proba,
        'valid_pred': valid_pred,
        'valid_proba': valid_proba
    }


def fit_catboost_holdout_stage(dev_df, holdout_df, y_dev, feature_info, task_type, devices, random_seed, verbose, selected_iteration, selected_threshold, min_positive_labels):
    X_dev, X_holdout = prep_catboost_frames(dev_df, holdout_df, feature_info)

    model = build_catboost_model(
        task_type=task_type,
        devices=devices,
        random_seed=random_seed,
        verbose=verbose,
        iterations=selected_iteration
    )
    start = perf_counter()
    model.fit(
        X_dev,
        y_dev,
        cat_features=feature_info['cat_cols'],
        use_best_model=False
    )
    fit_seconds = round(perf_counter() - start, 2)

    holdout_proba = model.predict_proba(X_holdout)
    holdout_pred = apply_multilabel_threshold(
        holdout_proba,
        selected_threshold,
        min_positive_labels=min_positive_labels
    )

    return {
        'model': model,
        'fit_seconds': fit_seconds,
        'selected_iteration': int(selected_iteration),
        'selected_threshold': float(selected_threshold),
        'holdout_pred': holdout_pred,
        'holdout_proba': holdout_proba
    }


def fit_catboost_selection_with_fallback(train_df, valid_df, y_train, y_valid, feature_info, task_type, devices, random_seed, verbose, iterations, eval_period, thresholds, min_positive_labels):
    requested_task_type = str(task_type).upper()
    try:
        result = fit_catboost_selection_stage(
            train_df,
            valid_df,
            y_train,
            y_valid,
            feature_info,
            task_type=requested_task_type,
            devices=devices,
            random_seed=random_seed,
            verbose=verbose,
            iterations=iterations,
            eval_period=eval_period,
            thresholds=thresholds,
            min_positive_labels=min_positive_labels
        )
        result['requested_task_type'] = requested_task_type
        result['actual_task_type'] = requested_task_type
        result['fallback_reason'] = None
        return result
    except CatBoostError as exc:
        if requested_task_type != 'GPU' or not is_retryable_catboost_gpu_error(exc):
            raise

        log_line(f'[warn] CatBoost multi-label GPU selection fit failed: {exc}')
        log_line('[warn] Retrying CatBoost multi-label selection on CPU')
        result = fit_catboost_selection_stage(
            train_df,
            valid_df,
            y_train,
            y_valid,
            feature_info,
            task_type='CPU',
            devices=devices,
            random_seed=random_seed,
            verbose=verbose,
            iterations=iterations,
            eval_period=eval_period,
            thresholds=thresholds,
            min_positive_labels=min_positive_labels
        )
        result['requested_task_type'] = requested_task_type
        result['actual_task_type'] = 'CPU'
        result['fallback_reason'] = str(exc)
        return result


def fit_catboost_holdout_with_fallback(dev_df, holdout_df, y_dev, feature_info, task_type, devices, random_seed, verbose, selected_iteration, selected_threshold, min_positive_labels):
    requested_task_type = str(task_type).upper()
    try:
        result = fit_catboost_holdout_stage(
            dev_df,
            holdout_df,
            y_dev,
            feature_info,
            task_type=requested_task_type,
            devices=devices,
            random_seed=random_seed,
            verbose=verbose,
            selected_iteration=selected_iteration,
            selected_threshold=selected_threshold,
            min_positive_labels=min_positive_labels
        )
        result['requested_task_type'] = requested_task_type
        result['actual_task_type'] = requested_task_type
        result['fallback_reason'] = None
        return result
    except CatBoostError as exc:
        if requested_task_type != 'GPU' or not is_retryable_catboost_gpu_error(exc):
            raise

        log_line(f'[warn] CatBoost multi-label GPU holdout refit failed: {exc}')
        log_line('[warn] Retrying CatBoost multi-label holdout refit on CPU')
        result = fit_catboost_holdout_stage(
            dev_df,
            holdout_df,
            y_dev,
            feature_info,
            task_type='CPU',
            devices=devices,
            random_seed=random_seed,
            verbose=verbose,
            selected_iteration=selected_iteration,
            selected_threshold=selected_threshold,
            min_positive_labels=min_positive_labels
        )
        result['requested_task_type'] = requested_task_type
        result['actual_task_type'] = 'CPU'
        result['fallback_reason'] = str(exc)
        return result
