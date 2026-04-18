import argparse
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler

from src.config import settings
from src.config.contracts import (
    COMPONENT_MULTI_OFFICIAL_LABELS,
    COMPONENT_MULTI_OFFICIAL_MANIFEST,
    COMPONENT_MULTI_OFFICIAL_METRICS,
    COMPONENT_MULTI_OFFICIAL_SPLIT,
    TRAIN_END,
    VALID_END,
)
from src.config.paths import OUTPUTS_DIR, ensure_project_directories
from src.data.io_utils import load_frame, write_json
from src.modeling.common.helpers import (
    CATBOOST_NAME,
    DEF_CATBOOST_EVAL_PERIOD,
    DEF_CATBOOST_ITERS,
    DEF_MIN_POSITIVE_LABELS,
    DEF_THRESHOLDS,
    MAX_TOP_K,
    MULTI_INPUT_STEM,
    MULTI_TARGET_COL,
    apply_multilabel_threshold,
    build_metric_row,
    feature_manifest,
    fit_catboost_holdout_with_fallback,
    fit_catboost_selection_with_fallback,
    log_line,
    parse_pipe_labels,
    prep_multi_label_cases,
    select_multilabel_threshold,
    split_multi_label_cases,
)

# -----------------------------------------------------------------------------
# Output names
# -----------------------------------------------------------------------------
SPLIT_NAME = COMPONENT_MULTI_OFFICIAL_SPLIT
METRIC_NAME = COMPONENT_MULTI_OFFICIAL_METRICS
LABEL_NAME = COMPONENT_MULTI_OFFICIAL_LABELS
MANIFEST_NAME = COMPONENT_MULTI_OFFICIAL_MANIFEST


# -----------------------------------------------------------------------------
# Benchmark defaults
# -----------------------------------------------------------------------------
NAIVE_NAME = 'Most Frequent Labels'
LOGIT_NAME = 'OneVsRest Logistic'
DEF_ONEHOT_MIN_FREQ = 50
DEF_OVR_N_JOBS = 1


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def parse_threshold_text(threshold_text):
    if not threshold_text:
        return list(DEF_THRESHOLDS)

    thresholds = [float(part.strip()) for part in threshold_text.split(',') if part.strip()]
    if not thresholds:
        raise ValueError('Threshold grid is empty after parsing')
    return thresholds


def build_naive_predictions(y_train, y_eval):
    label_prevalence = y_train.mean(axis=0)
    top_idx = np.argsort(label_prevalence)[-MAX_TOP_K:][::-1]
    proba = np.tile(label_prevalence, (len(y_eval), 1))
    pred = np.zeros_like(proba, dtype=int)
    pred[:, top_idx] = 1
    return pred, proba


def check_unseen_labels(train_labels, eval_labels, split_name):
    train_set = sorted({label for labels in train_labels for label in labels})
    train_lookup = set(train_set)
    unseen = sorted({label for labels in eval_labels for label in labels if label not in train_lookup})
    if unseen:
        unseen_text = ', '.join(unseen)
        raise ValueError(f'{split_name} has unseen target labels: {unseen_text}')


def build_logistic_pipeline(feature_info, random_seed, ovr_n_jobs, onehot_min_frequency):
    cat_cols = list(feature_info['cat_cols'])
    num_cols = list(feature_info['num_cols'] + feature_info['flag_cols'])

    cat_pipe = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            (
                'encoder',
                OneHotEncoder(
                    handle_unknown='infrequent_if_exist',
                    min_frequency=int(onehot_min_frequency)
                )
            )
        ]
    )
    num_pipe = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler(with_mean=False))
        ]
    )
    prep = ColumnTransformer(
        [
            ('cat', cat_pipe, cat_cols),
            ('num', num_pipe, num_cols)
        ]
    )

    return Pipeline(
        [
            ('prep', prep),
            (
                'model',
                OneVsRestClassifier(
                    LogisticRegression(
                        solver='saga',
                        max_iter=2000,
                        tol=1e-3,
                        random_state=random_seed
                    ),
                    n_jobs=int(ovr_n_jobs)
                )
            )
        ]
    )


def fit_logistic_selection_stage(train_df, valid_df, y_train, y_valid, feature_info, random_seed, thresholds, min_positive_labels, ovr_n_jobs, onehot_min_frequency):
    X_train = train_df[feature_info['feature_cols']].copy()
    X_valid = valid_df[feature_info['feature_cols']].copy()

    model = build_logistic_pipeline(
        feature_info,
        random_seed=random_seed,
        ovr_n_jobs=ovr_n_jobs,
        onehot_min_frequency=onehot_min_frequency
    )
    start = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = round(perf_counter() - start, 2)

    train_proba = model.predict_proba(X_train)
    valid_proba = model.predict_proba(X_valid)
    threshold_choice = select_multilabel_threshold(
        y_valid,
        valid_proba,
        thresholds=thresholds,
        min_positive_labels=min_positive_labels
    )
    selected_threshold = float(threshold_choice['threshold'])
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
        'selected_threshold': selected_threshold,
        'selection_metrics': threshold_choice,
        'train_pred': train_pred,
        'train_proba': train_proba,
        'valid_pred': valid_pred,
        'valid_proba': valid_proba
    }


def fit_logistic_holdout_stage(dev_df, holdout_df, y_dev, feature_info, random_seed, selected_threshold, min_positive_labels, ovr_n_jobs, onehot_min_frequency):
    X_dev = dev_df[feature_info['feature_cols']].copy()
    X_holdout = holdout_df[feature_info['feature_cols']].copy()

    model = build_logistic_pipeline(
        feature_info,
        random_seed=random_seed,
        ovr_n_jobs=ovr_n_jobs,
        onehot_min_frequency=onehot_min_frequency
    )
    start = perf_counter()
    model.fit(X_dev, y_dev)
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
        'selected_threshold': float(selected_threshold),
        'holdout_pred': holdout_pred,
        'holdout_proba': holdout_proba
    }


def select_official_model(metric_df):
    valid_rows = metric_df.loc[
        metric_df['stage'].eq('selection_train_valid')
        & metric_df['split'].eq('valid_2025')
    ].copy()
    if valid_rows.empty:
        raise ValueError('No validation rows available for model selection')

    valid_rows = valid_rows.sort_values(
        ['macro_f1', 'micro_f1', 'recall_at_3', 'precision_at_3'],
        ascending=False
    ).reset_index(drop=True)
    return valid_rows.iloc[0]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the multi-label structured component benchmark on the full kept-case routing problem'
    )
    parser.add_argument(
        '--input-path',
        default=None,
        help='Optional path to the multi-label component case parquet or csv file'
    )
    parser.add_argument(
        '--task-type',
        choices=['CPU', 'GPU', 'cpu', 'gpu'],
        default='CPU',
        help='CatBoost processing target for the multi-label candidate'
    )
    parser.add_argument(
        '--devices',
        default='0',
        help='GPU device string for CatBoost when task_type is GPU'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=settings.RANDOM_SEED,
        help='Random seed used for benchmark candidates'
    )
    parser.add_argument(
        '--catboost-iterations',
        type=int,
        default=DEF_CATBOOST_ITERS,
        help='Maximum CatBoost trees considered before validation-based selection'
    )
    parser.add_argument(
        '--catboost-eval-period',
        type=int,
        default=DEF_CATBOOST_EVAL_PERIOD,
        help='Tree interval used when scanning CatBoost validation probabilities'
    )
    parser.add_argument(
        '--threshold-grid',
        default=','.join(str(value) for value in DEF_THRESHOLDS),
        help='Comma-separated probability thresholds used for multi-label decoding'
    )
    parser.add_argument(
        '--min-positive-labels',
        type=int,
        default=DEF_MIN_POSITIVE_LABELS,
        help='Minimum predicted labels per complaint after thresholding'
    )
    parser.add_argument(
        '--onehot-min-frequency',
        type=int,
        default=DEF_ONEHOT_MIN_FREQ,
        help='Minimum category frequency before one-hot levels are grouped as infrequent'
    )
    parser.add_argument(
        '--ovr-n-jobs',
        type=int,
        default=DEF_OVR_N_JOBS,
        help='Parallel jobs for the logistic OneVsRest baseline'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=0,
        help='CatBoost logging interval'
    )
    parser.add_argument(
        '--publish-status',
        default='official',
        help='Release status recorded in the official manifest'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    threshold_grid = parse_threshold_text(args.threshold_grid)
    feature_info = feature_manifest('core_structured')

    log_line('[setup] Loading multi-label component cases')
    raw_df, input_path = load_frame(MULTI_INPUT_STEM, input_path=args.input_path)
    case_df = prep_multi_label_cases(raw_df, feature_info['feature_cols'])
    train_df, valid_df, holdout_df, split_df = split_multi_label_cases(case_df)
    dev_df = pd.concat([train_df, valid_df], ignore_index=True).sort_values(['ldate', 'odino']).reset_index(drop=True)

    train_labels = parse_pipe_labels(train_df[MULTI_TARGET_COL])
    valid_labels = parse_pipe_labels(valid_df[MULTI_TARGET_COL])
    holdout_labels = parse_pipe_labels(holdout_df[MULTI_TARGET_COL])
    check_unseen_labels(train_labels, valid_labels, 'Validation split')
    check_unseen_labels(train_labels + valid_labels, holdout_labels, 'Holdout split')

    train_mlb = MultiLabelBinarizer()
    y_train = train_mlb.fit_transform(train_labels)
    y_valid = train_mlb.transform(valid_labels)
    dev_mlb = MultiLabelBinarizer()
    y_dev = dev_mlb.fit_transform(train_labels + valid_labels)
    y_holdout = dev_mlb.transform(holdout_labels)

    label_count_series = case_df[MULTI_TARGET_COL].fillna('').astype(str)
    label_count_series = label_count_series.str.count(r'\|') + label_count_series.ne('')
    log_line(f'[setup] Input path: {input_path}')
    log_line(
        f'[setup] Split rows | train={len(train_df):,} valid_2025={len(valid_df):,} '
        f'holdout_2026={len(holdout_df):,}'
    )
    log_line(
        f'[setup] Labels | train={y_train.shape[1]} '
        f'mean_per_case={float(label_count_series.mean()):.4f} '
        f'pct_multi={float((label_count_series > 1).mean()):.4f}'
    )
    log_line(
        f'[setup] Task type={str(args.task_type).upper()} '
        f'devices={args.devices if str(args.task_type).upper() == "GPU" else "cpu-only"} '
        f'thresholds={threshold_grid}'
    )
    log_line(
        f'[setup] Logistic OVR jobs={args.ovr_n_jobs} '
        f'onehot_min_frequency={args.onehot_min_frequency} '
        f'min_positive_labels={args.min_positive_labels}'
    )

    metric_rows = []
    holdout_results = {}

    log_line('')
    log_line('[phase 1/4] Naive baseline')
    naive_train_pred, naive_train_proba = build_naive_predictions(y_train, y_train)
    naive_valid_pred, naive_valid_proba = build_naive_predictions(y_train, y_valid)
    metric_rows.append(
        build_metric_row(
            NAIVE_NAME,
            'selection_train_valid',
            'train',
            y_train,
            naive_train_pred,
            naive_train_proba
        )
    )
    metric_rows.append(
        build_metric_row(
            NAIVE_NAME,
            'selection_train_valid',
            'valid_2025',
            y_valid,
            naive_valid_pred,
            naive_valid_proba
        )
    )
    naive_holdout_pred, naive_holdout_proba = build_naive_predictions(y_dev, y_holdout)
    metric_rows.append(
        build_metric_row(
            NAIVE_NAME,
            'final_holdout',
            'holdout_2026',
            y_holdout,
            naive_holdout_pred,
            naive_holdout_proba
        )
    )
    holdout_results[NAIVE_NAME] = {
        'pred': naive_holdout_pred,
        'proba': naive_holdout_proba
    }
    log_line(
        f'[phase 1/4] {NAIVE_NAME} valid macro_f1={metric_rows[-2]["macro_f1"]:.4f} '
        f'micro_f1={metric_rows[-2]["micro_f1"]:.4f} recall@3={metric_rows[-2]["recall_at_3"]:.4f}'
    )

    log_line('')
    log_line('[phase 2/4] Logistic baseline')
    logistic_selection = fit_logistic_selection_stage(
        train_df,
        valid_df,
        y_train,
        y_valid,
        feature_info,
        random_seed=args.random_seed,
        thresholds=threshold_grid,
        min_positive_labels=args.min_positive_labels,
        ovr_n_jobs=args.ovr_n_jobs,
        onehot_min_frequency=args.onehot_min_frequency
    )
    metric_rows.append(
        build_metric_row(
            LOGIT_NAME,
            'selection_train_valid',
            'train',
            y_train,
            logistic_selection['train_pred'],
            logistic_selection['train_proba'],
            threshold=logistic_selection['selected_threshold'],
            fit_seconds=logistic_selection['fit_seconds']
        )
    )
    metric_rows.append(
        build_metric_row(
            LOGIT_NAME,
            'selection_train_valid',
            'valid_2025',
            y_valid,
            logistic_selection['valid_pred'],
            logistic_selection['valid_proba'],
            threshold=logistic_selection['selected_threshold'],
            fit_seconds=logistic_selection['fit_seconds']
        )
    )
    log_line(
        f'[phase 2/4] {LOGIT_NAME} fit={logistic_selection["fit_seconds"]:.2f}s '
        f'threshold={logistic_selection["selected_threshold"]:.2f} '
        f'valid macro_f1={metric_rows[-1]["macro_f1"]:.4f} '
        f'micro_f1={metric_rows[-1]["micro_f1"]:.4f} '
        f'recall@3={metric_rows[-1]["recall_at_3"]:.4f}'
    )

    log_line('')
    log_line('[phase 3/4] CatBoost multi-label candidate')
    catboost_selection = fit_catboost_selection_with_fallback(
        train_df,
        valid_df,
        y_train,
        y_valid,
        feature_info,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed,
        verbose=args.verbose,
        iterations=args.catboost_iterations,
        eval_period=args.catboost_eval_period,
        thresholds=threshold_grid,
        min_positive_labels=args.min_positive_labels
    )
    if catboost_selection['actual_task_type'] != str(args.task_type).upper():
        log_line(
            f'[phase 3/4] {CATBOOST_NAME} using {catboost_selection["actual_task_type"]} '
            f'instead of requested {str(args.task_type).upper()}'
        )
    metric_rows.append(
        build_metric_row(
            CATBOOST_NAME,
            'selection_train_valid',
            'train',
            y_train,
            catboost_selection['train_pred'],
            catboost_selection['train_proba'],
            threshold=catboost_selection['selected_threshold'],
            fit_seconds=catboost_selection['fit_seconds'],
            selected_iteration=catboost_selection['selected_iteration']
        )
    )
    metric_rows.append(
        build_metric_row(
            CATBOOST_NAME,
            'selection_train_valid',
            'valid_2025',
            y_valid,
            catboost_selection['valid_pred'],
            catboost_selection['valid_proba'],
            threshold=catboost_selection['selected_threshold'],
            fit_seconds=catboost_selection['fit_seconds'],
            selected_iteration=catboost_selection['selected_iteration']
        )
    )
    log_line(
        f'[phase 3/4] {CATBOOST_NAME} fit={catboost_selection["fit_seconds"]:.2f}s '
        f'iter={catboost_selection["selected_iteration"]} '
        f'threshold={catboost_selection["selected_threshold"]:.2f} '
        f'valid macro_f1={metric_rows[-1]["macro_f1"]:.4f} '
        f'micro_f1={metric_rows[-1]["micro_f1"]:.4f} '
        f'recall@3={metric_rows[-1]["recall_at_3"]:.4f}'
    )

    log_line('')
    log_line('[phase 4/4] Holdout evaluation')
    logistic_holdout = fit_logistic_holdout_stage(
        dev_df,
        holdout_df,
        y_dev,
        feature_info,
        random_seed=args.random_seed,
        selected_threshold=logistic_selection['selected_threshold'],
        min_positive_labels=args.min_positive_labels,
        ovr_n_jobs=args.ovr_n_jobs,
        onehot_min_frequency=args.onehot_min_frequency
    )
    metric_rows.append(
        build_metric_row(
            LOGIT_NAME,
            'final_holdout',
            'holdout_2026',
            y_holdout,
            logistic_holdout['holdout_pred'],
            logistic_holdout['holdout_proba'],
            threshold=logistic_holdout['selected_threshold'],
            fit_seconds=logistic_holdout['fit_seconds']
        )
    )
    holdout_results[LOGIT_NAME] = {
        'pred': logistic_holdout['holdout_pred'],
        'proba': logistic_holdout['holdout_proba']
    }
    log_line(
        f'[phase 4/4] {LOGIT_NAME} holdout macro_f1={metric_rows[-1]["macro_f1"]:.4f} '
        f'micro_f1={metric_rows[-1]["micro_f1"]:.4f}'
    )

    catboost_holdout = fit_catboost_holdout_with_fallback(
        dev_df,
        holdout_df,
        y_dev,
        feature_info,
        task_type=catboost_selection['actual_task_type'],
        devices=args.devices,
        random_seed=args.random_seed,
        verbose=args.verbose,
        selected_iteration=catboost_selection['selected_iteration'],
        selected_threshold=catboost_selection['selected_threshold'],
        min_positive_labels=args.min_positive_labels
    )
    if catboost_holdout['actual_task_type'] != catboost_selection['actual_task_type']:
        log_line(
            f'[phase 4/4] {CATBOOST_NAME} holdout refit switched to {catboost_holdout["actual_task_type"]}'
        )
    metric_rows.append(
        build_metric_row(
            CATBOOST_NAME,
            'final_holdout',
            'holdout_2026',
            y_holdout,
            catboost_holdout['holdout_pred'],
            catboost_holdout['holdout_proba'],
            threshold=catboost_holdout['selected_threshold'],
            fit_seconds=catboost_holdout['fit_seconds'],
            selected_iteration=catboost_holdout['selected_iteration']
        )
    )
    holdout_results[CATBOOST_NAME] = {
        'pred': catboost_holdout['holdout_pred'],
        'proba': catboost_holdout['holdout_proba']
    }
    log_line(
        f'[phase 4/4] {CATBOOST_NAME} holdout macro_f1={metric_rows[-1]["macro_f1"]:.4f} '
        f'micro_f1={metric_rows[-1]["micro_f1"]:.4f}'
    )

    metric_df = pd.DataFrame(metric_rows)
    selected_model_name = CATBOOST_NAME
    selected_model_row = metric_df.loc[
        metric_df['model'].eq(selected_model_name)
        & metric_df['stage'].eq('selection_train_valid')
        & metric_df['split'].eq('valid_2025')
    ].iloc[0]
    official_holdout_row = metric_df.loc[
        metric_df['model'].eq(selected_model_name)
        & metric_df['stage'].eq('final_holdout')
        & metric_df['split'].eq('holdout_2026')
    ].iloc[0]
    official_holdout = holdout_results[selected_model_name]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_holdout,
        official_holdout['pred'],
        average=None,
        zero_division=0
    )
    label_df = pd.DataFrame(
        {
            'model': selected_model_name,
            'component_group': dev_mlb.classes_,
            'support': support,
            'precision': np.round(precision, 4),
            'recall': np.round(recall, 4),
            'f1': np.round(f1, 4)
        }
    ).sort_values(['support', 'f1'], ascending=[False, False]).reset_index(drop=True)

    split_path = OUTPUTS_DIR / SPLIT_NAME
    metric_path = OUTPUTS_DIR / METRIC_NAME
    label_path = OUTPUTS_DIR / LABEL_NAME
    manifest_path = OUTPUTS_DIR / MANIFEST_NAME
    split_df.to_csv(split_path, index=False)
    metric_df.to_csv(metric_path, index=False)
    label_df.to_csv(label_path, index=False)

    manifest = {
        'artifact_role': 'component_multilabel_official',
        'target_scope': 'multi_label_component_routing',
        'task': 'multi_label_component_routing',
        'input_stem': MULTI_INPUT_STEM,
        'input_path': str(input_path),
        'official_model': selected_model_name,
        'selected_model': selected_model_name,
        'selected_feature_set': feature_info['feature_set_name'],
        'feature_manifest': feature_info,
        'selected_threshold': (
            float(selected_model_row['threshold'])
            if not pd.isna(selected_model_row['threshold'])
            else None
        ),
        'selected_iteration': (
            int(selected_model_row['selected_iteration'])
            if not pd.isna(selected_model_row['selected_iteration'])
            else None
        ),
        'official_metric': 'macro_f1 then micro_f1 on valid_2025, reported on holdout_2026',
        'official_holdout_metrics': official_holdout_row.to_dict(),
        'candidate_models': [NAIVE_NAME, LOGIT_NAME, CATBOOST_NAME],
        'promotion_status': args.publish_status,
        'reporting_ready': True,
        'catboost_runtime': {
            'requested_task_type': str(args.task_type).upper(),
            'selection_task_type': catboost_selection['actual_task_type'],
            'holdout_task_type': catboost_holdout['actual_task_type'],
            'selection_fallback_reason': catboost_selection['fallback_reason'],
            'holdout_fallback_reason': catboost_holdout['fallback_reason']
        },
        'selection_policy': {
            'primary': 'macro_f1 on valid_2025',
            'secondary': ['micro_f1', 'recall_at_3', 'precision_at_3'],
            'threshold_grid': threshold_grid,
            'min_positive_labels': int(args.min_positive_labels)
        },
        'split_policy': {
            'train_end': str(TRAIN_END.date()),
            'valid_end': str(VALID_END.date()),
            'holdout_policy': '2026 holdout untouched during threshold and model choice'
        },
        'split_rows': {
            'train': int(len(train_df)),
            'valid_2025': int(len(valid_df)),
            'holdout_2026': int(len(holdout_df)),
            'dev_2020_2025': int(len(dev_df))
        },
        'artifacts': {
            'split_summary': str(split_path),
            'metrics': str(metric_path),
            'label_metrics': str(label_path)
        }
    }
    write_json(manifest, manifest_path)

    print(f'[write] {split_path}')
    print(f'[write] {metric_path}')
    print(f'[write] {label_path}')
    print(f'[write] {manifest_path}')
    print('')
    print(f'[done] Multi-label component benchmark finished | selected model: {selected_model_name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
