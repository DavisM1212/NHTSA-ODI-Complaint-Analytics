import argparse
import inspect
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.config import settings
from src.config.contracts import TRAIN_END, VALID_END
from src.config.paths import OUTPUTS_DIR, ensure_project_directories
from src.data.io_utils import load_frame, write_dataframe, write_json
from src.modeling.common.helpers import (
    DATE_COL,
    ID_COL,
    MAX_TOP_K,
    SINGLE_INPUT_STEM,
    TARGET_COL,
    build_catboost_model,
    build_multiclass_calibration_df,
    build_multiclass_class_df,
    build_multiclass_confusion_df,
    build_multiclass_metric_row,
    feature_manifest,
    prep_catboost_frames,
    prep_single_label_cases,
    score_multiclass_from_proba,
    split_single_label_cases,
)

# -----------------------------------------------------------------------------
# Output names
# -----------------------------------------------------------------------------
SELECTION_MANIFEST_NAME = 'component_single_label_selection_manifest.json'
SPLIT_NAME = 'component_single_label_split_summary.csv'
METRIC_NAME = 'component_single_label_benchmark_metrics.csv'
CLASS_NAME = 'component_single_label_holdout_class_metrics.csv'
IMPORTANCE_NAME = 'component_single_label_feature_importance.csv'
CALIB_NAME = 'component_single_label_holdout_calibration.csv'
CONFUSION_NAME = 'component_single_label_holdout_confusion_major.csv'
MANIFEST_NAME = 'component_single_label_benchmark_manifest.json'
HOLDOUT_PRED_STEM = 'component_single_label_holdout_preds'
MODEL_NAME = 'component_single_label_model.cbm'


# -----------------------------------------------------------------------------
# Shared prep for sklearn baselines
# -----------------------------------------------------------------------------
def resolve_selection_manifest(selection_manifest_path=None):
    if selection_manifest_path is not None:
        path = Path(selection_manifest_path)
        if not path.exists():
            raise FileNotFoundError(f'Selection manifest not found: {path}')
        return path

    default_path = OUTPUTS_DIR / SELECTION_MANIFEST_NAME
    if not default_path.exists():
        raise FileNotFoundError(
            f'No selection manifest found at {default_path}. Run tune_component_catboost first'
        )
    return default_path


def load_selection_manifest(selection_manifest_path=None):
    path = resolve_selection_manifest(selection_manifest_path)
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle), path


def build_logistic_pipeline(feature_info, random_seed):
    cat_cols = list(feature_info['cat_cols'])
    num_cols = list(feature_info['num_cols'] + feature_info['flag_cols'])

    cat_pipe = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
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
    logit_kwargs = {
        'solver': 'saga',
        'class_weight': 'balanced',
        'max_iter': 4000,
        'tol': 1e-3,
        'random_state': random_seed
    }

    # sklearn 1.8 removed multi_class, but older versions still need it here
    if 'multi_class' in inspect.signature(LogisticRegression).parameters:
        logit_kwargs['multi_class'] = 'multinomial'

    return Pipeline(
        [
            ('prep', prep),
            (
                'model',
                LogisticRegression(**logit_kwargs)
            )
        ]
    )


def build_histgb_model(random_seed):
    return HistGradientBoostingClassifier(
        loss='log_loss',
        learning_rate=0.05,
        max_iter=300,
        max_depth=10,
        min_samples_leaf=30,
        class_weight='balanced',
        random_state=random_seed
    )


def build_histgb_pipeline(feature_info, random_seed):
    cat_cols = list(feature_info['cat_cols'])
    num_cols = list(feature_info['num_cols'] + feature_info['flag_cols'])

    cat_pipe = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            (
                'encoder',
                OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                    encoded_missing_value=-1
                )
            )
        ]
    )
    num_pipe = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='median'))
        ]
    )
    prep = ColumnTransformer(
        [
            ('cat', cat_pipe, cat_cols),
            ('num', num_pipe, num_cols)
        ],
        sparse_threshold=0
    )

    return Pipeline(
        [
            ('prep', prep),
            ('model', build_histgb_model(random_seed=random_seed))
        ]
    )


# -----------------------------------------------------------------------------
# Baseline and benchmark runners
# -----------------------------------------------------------------------------
def build_naive_rows(train_target, eval_target, stage_name, split_name):
    train_counts = train_target.value_counts()
    majority_label = str(train_counts.idxmax())
    top_labels = train_counts.head(MAX_TOP_K).index.tolist()
    pred = pd.Series(majority_label, index=eval_target.index)
    metrics = {
        'top_1_accuracy': round(float((eval_target == majority_label).mean()), 4),
        'macro_f1': round(float(f1_score(eval_target, pred, average='macro')), 4),
        'top_3_accuracy': round(float(eval_target.isin(top_labels).mean()), 4)
    }
    return {
        'model': 'Most Frequent Label',
        'stage': stage_name,
        'split': split_name,
        'rows': int(len(eval_target)),
        'fit_seconds': np.nan,
        'selected_iteration': np.nan,
        **metrics
    }


def fit_logistic_stage(train_df, eval_df, feature_info, stage_name, split_name, random_seed):
    X_train = train_df[feature_info['feature_cols']].copy()
    y_train = train_df[TARGET_COL].copy()
    X_eval = eval_df[feature_info['feature_cols']].copy()
    y_eval = eval_df[TARGET_COL].copy()

    model = build_logistic_pipeline(feature_info, random_seed=random_seed)
    start = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = round(perf_counter() - start, 2)

    train_proba = model.predict_proba(X_train)
    eval_proba = model.predict_proba(X_eval)
    classes = model.named_steps['model'].classes_

    return [
        build_multiclass_metric_row(
            'Logistic Regression',
            stage_name,
            'train' if split_name == 'valid_2025' else split_name,
            y_train if split_name == 'valid_2025' else y_eval,
            train_proba if split_name == 'valid_2025' else eval_proba,
            classes,
            fit_seconds=fit_seconds
        ),
        build_multiclass_metric_row(
            'Logistic Regression',
            stage_name,
            split_name,
            y_eval,
            eval_proba,
            classes,
            fit_seconds=fit_seconds
        )
    ] if split_name == 'valid_2025' else [
        build_multiclass_metric_row(
            'Logistic Regression',
            stage_name,
            split_name,
            y_eval,
            eval_proba,
            classes,
            fit_seconds=fit_seconds
        )
    ]


def fit_histgb_stage(train_df, eval_df, feature_info, stage_name, split_name, random_seed):
    X_train = train_df[feature_info['feature_cols']].copy()
    y_train = train_df[TARGET_COL].copy()
    X_eval = eval_df[feature_info['feature_cols']].copy()
    y_eval = eval_df[TARGET_COL].copy()

    model = build_histgb_pipeline(feature_info, random_seed=random_seed)
    start = perf_counter()
    # Keep this secondary baseline single-threaded so sklearn behaves on Windows and in sandboxed runs
    # with threadpool_limits(limits=1):
    #     model.fit(X_train, y_train)
    model.fit(X_train, y_train)
    fit_seconds = round(perf_counter() - start, 2)

    train_proba = model.predict_proba(X_train)
    eval_proba = model.predict_proba(X_eval)
    classes = model.named_steps['model'].classes_

    return [
        build_multiclass_metric_row(
            'HistGradientBoosting',
            stage_name,
            'train' if split_name == 'valid_2025' else split_name,
            y_train if split_name == 'valid_2025' else y_eval,
            train_proba if split_name == 'valid_2025' else eval_proba,
            classes,
            fit_seconds=fit_seconds
        ),
        build_multiclass_metric_row(
            'HistGradientBoosting',
            stage_name,
            split_name,
            y_eval,
            eval_proba,
            classes,
            fit_seconds=fit_seconds
        )
    ] if split_name == 'valid_2025' else [
        build_multiclass_metric_row(
            'HistGradientBoosting',
            stage_name,
            split_name,
            y_eval,
            eval_proba,
            classes,
            fit_seconds=fit_seconds
        )
    ]


def fit_fixed_catboost_stage(train_df, eval_df, feature_info, params, selected_iteration, task_type, devices, random_seed, verbose, stage_name, split_name):
    X_train, X_eval = prep_catboost_frames(train_df, eval_df, feature_info)
    y_train = train_df[TARGET_COL].copy()
    y_eval = eval_df[TARGET_COL].copy()

    fixed_params = dict(params)
    fixed_params['iterations'] = int(selected_iteration)
    model = build_catboost_model(
        fixed_params,
        task_type=task_type,
        devices=devices,
        random_seed=random_seed,
        verbose=verbose
    )

    start = perf_counter()
    model.fit(
        X_train,
        y_train,
        cat_features=feature_info['cat_cols'],
        use_best_model=False
    )
    fit_seconds = round(perf_counter() - start, 2)

    train_proba = model.predict_proba(X_train)
    eval_proba = model.predict_proba(X_eval)
    classes = model.classes_

    metric_rows = [
        build_multiclass_metric_row(
            'CatBoost',
            stage_name,
            'train' if split_name == 'valid_2025' else split_name,
            y_train if split_name == 'valid_2025' else y_eval,
            train_proba if split_name == 'valid_2025' else eval_proba,
            classes,
            fit_seconds=fit_seconds,
            selected_iteration=selected_iteration
        ),
        build_multiclass_metric_row(
            'CatBoost',
            stage_name,
            split_name,
            y_eval,
            eval_proba,
            classes,
            fit_seconds=fit_seconds,
            selected_iteration=selected_iteration
        )
    ] if split_name == 'valid_2025' else [
        build_multiclass_metric_row(
            'CatBoost',
            stage_name,
            split_name,
            y_eval,
            eval_proba,
            classes,
            fit_seconds=fit_seconds,
            selected_iteration=selected_iteration
        )
    ]

    return {
        'model': model,
        'metric_rows': metric_rows,
        'classes': classes,
        'train_proba': train_proba,
        'eval_proba': eval_proba,
        'train_target': y_train,
        'eval_target': y_eval,
        'eval_frame': eval_df
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the single-label structured component benchmark with baselines and an untouched 2026 holdout'
    )
    parser.add_argument(
        '--input-path',
        default=None,
        help='Optional path to the single-label component case parquet or csv file'
    )
    parser.add_argument(
        '--selection-manifest',
        default=None,
        help='Optional path to the component selection manifest json file'
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
        '--random-seed',
        type=int,
        default=settings.RANDOM_SEED,
        help='Random seed used for benchmark baselines and the fixed CatBoost refit'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=0,
        help='CatBoost logging interval'
    )
    parser.add_argument(
        '--output-format',
        choices=['parquet', 'csv'],
        default=settings.OUTPUT_FORMAT if settings.OUTPUT_FORMAT in {'parquet', 'csv'} else 'parquet',
        help='Preferred output format for holdout predictions'
    )
    parser.add_argument(
        '--skip-readme-update',
        action='store_true',
        help='Skip updating the README benchmark section from the final artifact'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    selection_manifest, selection_manifest_path = load_selection_manifest(args.selection_manifest)
    feature_info = selection_manifest['feature_manifest']
    feature_info = feature_manifest(feature_info['feature_set_name'])
    selected_iteration = int(selection_manifest['selected_iteration'])
    best_params = selection_manifest['best_params']

    raw_df, input_path = load_frame(SINGLE_INPUT_STEM, input_path=args.input_path)
    case_df = prep_single_label_cases(raw_df, feature_info['feature_cols'])
    train_df, valid_df, holdout_df, split_df = split_single_label_cases(case_df)
    dev_df = pd.concat([train_df, valid_df], ignore_index=True).sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    metric_rows = []
    metric_rows.extend(
        [
            build_naive_rows(train_df[TARGET_COL], train_df[TARGET_COL], 'selection_train_valid', 'train'),
            build_naive_rows(train_df[TARGET_COL], valid_df[TARGET_COL], 'selection_train_valid', 'valid_2025')
        ]
    )
    metric_rows.extend(
        fit_logistic_stage(
            train_df,
            valid_df,
            feature_info,
            stage_name='selection_train_valid',
            split_name='valid_2025',
            random_seed=args.random_seed
        )
    )
    metric_rows.extend(
        fit_histgb_stage(
            train_df,
            valid_df,
            feature_info,
            stage_name='selection_train_valid',
            split_name='valid_2025',
            random_seed=args.random_seed
        )
    )
    selection_catboost = fit_fixed_catboost_stage(
        train_df,
        valid_df,
        feature_info,
        best_params,
        selected_iteration,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed,
        verbose=args.verbose,
        stage_name='selection_train_valid',
        split_name='valid_2025'
    )
    metric_rows.extend(selection_catboost['metric_rows'])

    metric_rows.append(
        build_naive_rows(dev_df[TARGET_COL], holdout_df[TARGET_COL], 'final_holdout', 'holdout_2026')
    )
    metric_rows.extend(
        fit_logistic_stage(
            dev_df,
            holdout_df,
            feature_info,
            stage_name='final_holdout',
            split_name='holdout_2026',
            random_seed=args.random_seed
        )
    )
    metric_rows.extend(
        fit_histgb_stage(
            dev_df,
            holdout_df,
            feature_info,
            stage_name='final_holdout',
            split_name='holdout_2026',
            random_seed=args.random_seed
        )
    )
    holdout_catboost = fit_fixed_catboost_stage(
        dev_df,
        holdout_df,
        feature_info,
        best_params,
        selected_iteration,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed,
        verbose=args.verbose,
        stage_name='final_holdout',
        split_name='holdout_2026'
    )
    metric_rows.extend(holdout_catboost['metric_rows'])
    metric_df = pd.DataFrame(metric_rows)

    holdout_pred, holdout_metrics = score_multiclass_from_proba(
        holdout_catboost['eval_target'],
        holdout_catboost['eval_proba'],
        holdout_catboost['classes']
    )
    class_df = build_multiclass_class_df(
        holdout_catboost['eval_target'],
        holdout_pred,
        holdout_catboost['classes']
    )
    importance_df = pd.DataFrame(
        {
            'feature': feature_info['feature_cols'],
            'importance': np.round(holdout_catboost['model'].get_feature_importance(), 4)
        }
    ).sort_values('importance', ascending=False).reset_index(drop=True)
    calibration_df = build_multiclass_calibration_df(
        holdout_catboost['eval_target'],
        holdout_catboost['eval_proba'],
        holdout_catboost['classes']
    )

    focus_groups = (
        dev_df[TARGET_COL]
        .value_counts()
        .head(12)
        .index
        .tolist()
    )
    confusion_df = build_multiclass_confusion_df(
        holdout_catboost['eval_target'],
        holdout_pred,
        focus_groups
    )

    top_idx = np.argsort(holdout_catboost['eval_proba'], axis=1)[:, -MAX_TOP_K:][:, ::-1]
    top_labels = np.asarray(holdout_catboost['classes'])[top_idx]
    top_scores = np.take_along_axis(holdout_catboost['eval_proba'], top_idx, axis=1)
    holdout_pred_df = holdout_df[[ID_COL, DATE_COL, TARGET_COL]].copy().reset_index(drop=True)
    holdout_pred_df = holdout_pred_df.rename(columns={TARGET_COL: 'component_true'})
    holdout_pred_df['component_pred'] = holdout_pred
    holdout_pred_df['component_correct_top1'] = holdout_pred_df['component_true'].eq(holdout_pred_df['component_pred'])
    holdout_pred_df['component_correct_top3'] = [
        truth in labels
        for truth, labels in zip(holdout_pred_df['component_true'], top_labels)
    ]
    holdout_pred_df['pred_confidence_top1'] = np.round(top_scores[:, 0], 6)
    for rank in [1, 2, 3]:
        holdout_pred_df[f'pred_top{rank}'] = top_labels[:, rank - 1]
        holdout_pred_df[f'pred_top{rank}_proba'] = np.round(top_scores[:, rank - 1], 6)

    split_path = OUTPUTS_DIR / SPLIT_NAME
    metric_path = OUTPUTS_DIR / METRIC_NAME
    class_path = OUTPUTS_DIR / CLASS_NAME
    importance_path = OUTPUTS_DIR / IMPORTANCE_NAME
    calib_path = OUTPUTS_DIR / CALIB_NAME
    confusion_path = OUTPUTS_DIR / CONFUSION_NAME
    manifest_path = OUTPUTS_DIR / MANIFEST_NAME
    model_path = OUTPUTS_DIR / MODEL_NAME
    pred_path = write_dataframe(
        holdout_pred_df,
        OUTPUTS_DIR / HOLDOUT_PRED_STEM,
        prefer_parquet=args.output_format == 'parquet'
    )

    split_df.to_csv(split_path, index=False)
    metric_df.to_csv(metric_path, index=False)
    class_df.to_csv(class_path, index=False)
    importance_df.to_csv(importance_path, index=False)
    calibration_df.to_csv(calib_path, index=False)
    calibration_df.to_csv(calib_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)
    holdout_catboost['model'].save_model(model_path.as_posix())

    holdout_metric_row = metric_df.loc[
        metric_df['model'].eq('CatBoost')
        & metric_df['stage'].eq('final_holdout')
        & metric_df['split'].eq('holdout_2026')
    ].iloc[0]
    manifest = {
        'artifact_role': 'final_benchmark',
        'target_scope': 'single_label_benchmark',
        'input_stem': SINGLE_INPUT_STEM,
        'input_path': str(input_path),
        'selection_manifest_path': str(selection_manifest_path),
        'selected_feature_set': feature_info['feature_set_name'],
        'feature_manifest': feature_info,
        'best_params': best_params,
        'selected_iteration': selected_iteration,
        'official_metric': 'macro_f1 on holdout_2026',
        'official_holdout_metrics': holdout_metric_row.to_dict(),
        'candidate_models': ['Most Frequent Label', 'Logistic Regression', 'HistGradientBoosting', 'CatBoost'],
        'split_policy': {
            'train_end': str(TRAIN_END.date()),
            'valid_end': str(VALID_END.date()),
            'holdout_policy': '2026 holdout untouched during feature selection and tuning'
        },
        'selection_summary': selection_manifest.get('selection_metrics'),
        'split_rows': {
            'train': int(len(train_df)),
            'valid_2025': int(len(valid_df)),
            'holdout_2026': int(len(holdout_df)),
            'dev_2020_2025': int(len(dev_df))
        }
    }
    write_json(manifest, manifest_path)

    if not args.skip_readme_update:
        try:
            from src.reporting.update_component_readme import update_component_readme

            update_component_readme()
        except Exception as exc:
            print(f'[warn] README benchmark update skipped: {exc}')

    print(f'[write] {split_path}')
    print(f'[write] {metric_path}')
    print(f'[write] {class_path}')
    print(f'[write] {importance_path}')
    print(f'[write] {calib_path}')
    print(f'[write] {confusion_path}')
    print(f'[write] {pred_path}')
    print(f'[write] {manifest_path}')
    print(f'[write] {model_path}')
    print('')
    print('[done] Single-label component benchmark finished')
    return 0


if __name__ == '__main__':
    sys.exit(main())
