import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

from src.config import settings
from src.config.paths import OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_project_directories
from src.data.io_utils import write_dataframe

# -----------------------------------------------------------------------------
# Output names
# -----------------------------------------------------------------------------
INPUT_STEM = 'odi_component_model_cases'
MODEL_STEM = 'component_catboost_bench'
SPLIT_NAME = 'component_catboost_split_summary.csv'
METRIC_NAME = 'component_catboost_metrics.csv'
CLASS_NAME = 'component_catboost_class_metrics.csv'
IMPORTANCE_NAME = 'component_catboost_feature_importance.csv'
PARAMS_NAME = 'component_catboost_params.json'
VALID_PRED_STEM = 'component_catboost_valid_preds'
MODEL_NAME = 'component_catboost_model.cbm'


# -----------------------------------------------------------------------------
# Benchmark setup
# -----------------------------------------------------------------------------
ID_COL = 'odino'
TARGET_COL = 'component_group'
DATE_COL = 'ldate'
TRAIN_END = pd.Timestamp('2024-12-31')
VALID_END = pd.Timestamp('2025-12-31')

CAT_COLS = [
    'mfr_name',
    'maketxt',
    'modeltxt',
    'state',
    'cmpl_type',
    'drive_train',
    'fuel_sys',
    'fuel_type',
    'trans_type',
    'fire',
    'crash',
    'medical_attn',
    'vehicles_towed_yn',
    'police_rpt_yn',
    'orig_owner_yn',
    'anti_brakes_yn',
    'cruise_cont_yn'
]

NUM_COLS = [
    'yeartxt',
    'miles',
    'veh_speed',
    'injured',
    'deaths',
    'num_cyls',
    'lag_days_safe'
]

FLAG_COLS = [
    'miles_missing_flag',
    'veh_speed_missing_flag',
    'faildate_untrusted_flag',
    'flag_year_unknown',
    'flag_speed_high',
    'flag_miles_high'
]

FEATURE_COLS = CAT_COLS + NUM_COLS + FLAG_COLS

BENCH_PARAMS = {
    'bootstrap_type': 'Bernoulli',
    'iterations': 1800,
    'learning_rate': 0.07405467149893648,
    'depth': 9,
    'l2_leaf_reg': 7.572705439311379,
    'random_strength': 0.29374126086853103,
    'border_count': 128,
    'subsample': 0.6895168484791427
}


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
def resolve_input_path(input_path=None):
    if input_path is not None:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f'Input file not found: {path}')
        return path

    parquet_path = PROCESSED_DATA_DIR / f'{INPUT_STEM}.parquet'
    if parquet_path.exists():
        return parquet_path

    csv_path = PROCESSED_DATA_DIR / f'{INPUT_STEM}.csv'
    if csv_path.exists():
        return csv_path

    raise FileNotFoundError(
        'No component case file found under data/processed. Run the component collapse first'
    )


def load_cases(input_path=None):
    path = resolve_input_path(input_path)
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def require_cols(df):
    required = FEATURE_COLS + [ID_COL, TARGET_COL, DATE_COL]
    missing = [column for column in required if column not in df.columns]
    if missing:
        missing_text = ', '.join(missing)
        raise ValueError(f'Missing required modeling columns: {missing_text}')


# -----------------------------------------------------------------------------
# Prep and split
# -----------------------------------------------------------------------------
def prep_cases(df):
    require_cols(df)
    work = df.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors='coerce')
    if work[DATE_COL].isna().any():
        missing_dates = int(work[DATE_COL].isna().sum())
        raise ValueError(f'Found {missing_dates} rows with invalid {DATE_COL} values')

    for column in CAT_COLS:
        work[column] = work[column].astype('string').fillna('__MISSING__').astype(str)

    for column in NUM_COLS + FLAG_COLS:
        work[column] = pd.to_numeric(work[column], errors='coerce')

    return work.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)


def split_cases(df):
    train_df = df.loc[df[DATE_COL] <= TRAIN_END].copy()
    valid_df = df.loc[(df[DATE_COL] > TRAIN_END) & (df[DATE_COL] <= VALID_END)].copy()
    holdout_df = df.loc[df[DATE_COL] > VALID_END].copy()

    if train_df.empty:
        raise ValueError('Training split is empty')
    if valid_df.empty:
        raise ValueError('Validation split is empty')

    unseen_valid = sorted(set(valid_df[TARGET_COL]) - set(train_df[TARGET_COL]))
    if unseen_valid:
        unseen_text = ', '.join(unseen_valid)
        raise ValueError(f'Validation split has unseen target labels: {unseen_text}')

    split_df = pd.DataFrame(
        [
            {
                'split': 'train',
                'rows': int(len(train_df)),
                'cases': int(train_df[ID_COL].nunique()),
                'date_min': train_df[DATE_COL].min(),
                'date_max': train_df[DATE_COL].max(),
                'target_groups': int(train_df[TARGET_COL].nunique())
            },
            {
                'split': 'valid_2025',
                'rows': int(len(valid_df)),
                'cases': int(valid_df[ID_COL].nunique()),
                'date_min': valid_df[DATE_COL].min(),
                'date_max': valid_df[DATE_COL].max(),
                'target_groups': int(valid_df[TARGET_COL].nunique())
            },
            {
                'split': 'holdout_2026',
                'rows': int(len(holdout_df)),
                'cases': int(holdout_df[ID_COL].nunique()),
                'date_min': holdout_df[DATE_COL].min(),
                'date_max': holdout_df[DATE_COL].max(),
                'target_groups': int(holdout_df[TARGET_COL].nunique())
            }
        ]
    )
    return train_df, valid_df, holdout_df, split_df


# -----------------------------------------------------------------------------
# Benchmark model
# -----------------------------------------------------------------------------
def build_model(task_type='CPU', devices='0', random_seed=None, verbose=100, params_override=None):
    task_type = str(task_type).upper().strip()
    if task_type not in {'CPU', 'GPU'}:
        raise ValueError("task_type must be either 'CPU' or 'GPU'")

    params = {
        'loss_function': 'MultiClass',
        'eval_metric': 'TotalF1:average=Macro',
        'custom_metric': ['Accuracy', 'MultiClass'],
        'auto_class_weights': 'Balanced',
        'has_time': True,
        'grow_policy': 'SymmetricTree',
        'random_seed': settings.RANDOM_SEED if random_seed is None else int(random_seed),
        'allow_writing_files': False,
        'task_type': task_type,
        'verbose': int(verbose)
    }
    params.update(BENCH_PARAMS)

    if params_override:
        params.update(params_override)

    if task_type == 'GPU':
        params['devices'] = str(devices)

    return CatBoostClassifier(**params)


def score_split(model_name, split_name, y_true, pred, proba, classes, fit_seconds=pd.NA, best_iteration=pd.NA):
    return {
        'model': model_name,
        'split': split_name,
        'rows': int(len(y_true)),
        'fit_seconds': fit_seconds,
        'best_iteration': best_iteration,
        'top_1_accuracy': round(accuracy_score(y_true, pred), 4),
        'macro_f1': round(f1_score(y_true, pred, average='macro'), 4),
        'top_3_accuracy': round(top_k_accuracy_score(y_true, proba, labels=classes, k=3), 4)
    }


def fit_component_model(train_df, valid_df, task_type='CPU', devices='0', random_seed=None, verbose=100, params_override=None):
    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df[TARGET_COL].copy()
    X_valid = valid_df[FEATURE_COLS].copy()
    y_valid = valid_df[TARGET_COL].copy()

    unseen_valid = sorted(set(y_valid) - set(y_train))
    if unseen_valid:
        unseen_text = ', '.join(unseen_valid)
        raise ValueError(f'Validation frame has unseen target labels: {unseen_text}')

    train_counts = y_train.value_counts()
    majority_label = train_counts.idxmax()
    top3_labels = train_counts.head(3).index.tolist()
    naive_pred = pd.Series(majority_label, index=y_valid.index)

    model = build_model(
        task_type=task_type,
        devices=devices,
        random_seed=random_seed,
        verbose=verbose,
        params_override=params_override
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
    fit_seconds = round(perf_counter() - start, 2)
    best_iteration = int(model.get_best_iteration())

    train_pred = pd.Series(model.predict(X_train).ravel(), index=y_train.index)
    train_proba = model.predict_proba(X_train)
    valid_pred = pd.Series(model.predict(X_valid).ravel(), index=y_valid.index)
    valid_proba = model.predict_proba(X_valid)
    classes = model.classes_

    metric_rows = [
        {
            'model': 'Most Frequent Label',
            'split': 'valid_2025',
            'rows': int(len(y_valid)),
            'fit_seconds': np.nan,
            'best_iteration': np.nan,
            'top_1_accuracy': round(accuracy_score(y_valid, naive_pred), 4),
            'macro_f1': round(f1_score(y_valid, naive_pred, average='macro'), 4),
            'top_3_accuracy': round(y_valid.isin(top3_labels).mean(), 4)
        },
        score_split(
            'CatBoost',
            'train',
            y_train,
            train_pred,
            train_proba,
            classes,
            fit_seconds=fit_seconds,
            best_iteration=best_iteration
        ),
        score_split(
            'CatBoost',
            'valid_2025',
            y_valid,
            valid_pred,
            valid_proba,
            classes,
            fit_seconds=fit_seconds,
            best_iteration=best_iteration
        )
    ]
    metric_df = pd.DataFrame(metric_rows)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_valid,
        valid_pred,
        labels=classes,
        zero_division=0
    )
    class_df = pd.DataFrame(
        {
            'component_group': classes,
            'support': support,
            'precision': np.round(precision, 4),
            'recall': np.round(recall, 4),
            'f1': np.round(f1, 4)
        }
    ).sort_values(['support', 'f1'], ascending=[False, False]).reset_index(drop=True)

    importance_df = pd.DataFrame(
        {
            'feature': FEATURE_COLS,
            'importance': np.round(model.get_feature_importance(), 4)
        }
    ).sort_values('importance', ascending=False).reset_index(drop=True)

    top_idx = np.argsort(valid_proba, axis=1)[:, -3:][:, ::-1]
    top_labels = np.asarray(classes)[top_idx]
    top_scores = np.take_along_axis(valid_proba, top_idx, axis=1)

    pred_df = valid_df[[ID_COL, DATE_COL, TARGET_COL]].copy().reset_index(drop=True)
    pred_df = pred_df.rename(columns={TARGET_COL: 'component_true'})
    pred_df['component_pred'] = valid_pred.reset_index(drop=True)
    pred_df['component_correct_top1'] = pred_df['component_true'].eq(pred_df['component_pred'])
    pred_df['component_correct_top3'] = [
        truth in labels
        for truth, labels in zip(pred_df['component_true'], top_labels)
    ]

    for rank in [1, 2, 3]:
        pred_df[f'pred_top{rank}'] = top_labels[:, rank - 1]
        pred_df[f'pred_top{rank}_proba'] = np.round(top_scores[:, rank - 1], 6)

    run_info = {
        'benchmark_label': 'optuna_focus_trial_35_default',
        'input_stem': INPUT_STEM,
        'task_type': str(task_type).upper().strip(),
        'devices': str(devices) if str(task_type).upper().strip() == 'GPU' else None,
        'random_seed': settings.RANDOM_SEED if random_seed is None else int(random_seed),
        'train_end': str(TRAIN_END.date()),
        'valid_end': str(VALID_END.date()),
        'fit_seconds': fit_seconds,
        'best_iteration': best_iteration,
        'best_score': model.get_best_score(),
        'feature_cols': FEATURE_COLS,
        'cat_cols': CAT_COLS,
        'num_cols': NUM_COLS,
        'flag_cols': FLAG_COLS,
        'benchmark_params': BENCH_PARAMS,
        'model_params': model.get_all_params()
    }

    return model, metric_df, class_df, importance_df, pred_df, run_info


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the benchmark CatBoost component model on the single-label component case table'
    )
    parser.add_argument(
        '--input-path',
        default=None,
        help='Optional path to the component case parquet or csv file'
    )
    parser.add_argument(
        '--task-type',
        choices=['CPU', 'GPU', 'cpu', 'gpu'],
        default='CPU',
        help='CatBoost processing target. Use GPU in Colab or another GPU runtime for benchmark-speed training'
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
        help='Random seed for the CatBoost benchmark run'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=100,
        help='CatBoost logging interval'
    )
    parser.add_argument(
        '--output-format',
        choices=['parquet', 'csv'],
        default=settings.OUTPUT_FORMAT if settings.OUTPUT_FORMAT in {'parquet', 'csv'} else 'parquet',
        help='Preferred output format for validation predictions'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    raw_df = load_cases(args.input_path)
    case_df = prep_cases(raw_df)
    train_df, valid_df, holdout_df, split_df = split_cases(case_df)
    model, metric_df, class_df, importance_df, pred_df, run_info = fit_component_model(
        train_df,
        valid_df,
        task_type=args.task_type,
        devices=args.devices,
        random_seed=args.random_seed,
        verbose=args.verbose
    )

    split_path = OUTPUTS_DIR / SPLIT_NAME
    metric_path = OUTPUTS_DIR / METRIC_NAME
    class_path = OUTPUTS_DIR / CLASS_NAME
    importance_path = OUTPUTS_DIR / IMPORTANCE_NAME
    params_path = OUTPUTS_DIR / PARAMS_NAME
    pred_path = write_dataframe(
        pred_df,
        OUTPUTS_DIR / VALID_PRED_STEM,
        prefer_parquet=args.output_format == 'parquet'
    )
    model_path = OUTPUTS_DIR / MODEL_NAME

    split_df.to_csv(split_path, index=False)
    metric_df.to_csv(metric_path, index=False)
    class_df.to_csv(class_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    run_info['split_rows'] = {
        'train': int(len(train_df)),
        'valid_2025': int(len(valid_df)),
        'holdout_2026': int(len(holdout_df))
    }
    with params_path.open('w', encoding='utf-8') as handle:
        json.dump(run_info, handle, indent=2)

    model.save_model(model_path.as_posix())

    print(f'[write] {split_path}')
    print(f'[write] {metric_path}')
    print(f'[write] {class_path}')
    print(f'[write] {importance_path}')
    print(f'[write] {params_path}')
    print(f'[write] {pred_path}')
    print(f'[write] {model_path}')
    print('')
    print('[done] Component CatBoost benchmark finished')
    return 0


if __name__ == '__main__':
    sys.exit(main())
