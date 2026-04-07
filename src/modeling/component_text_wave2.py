import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MultiLabelBinarizer,
    OneHotEncoder,
    StandardScaler,
    normalize,
)

from src.config import settings
from src.config.paths import OUTPUTS_DIR, ensure_project_directories
from src.features.component_text_sidecar import SIDECAR_STEM
from src.modeling.component_common import (
    DATE_COL,
    FEATURE_WAVE1_SPLIT_MODE,
    ID_COL,
    MAX_TOP_K,
    MULTI_INPUT_STEM,
    MULTI_TARGET_COL,
    SINGLE_INPUT_STEM,
    TARGET_COL,
    apply_multilabel_threshold,
    build_catboost_model,
    build_multiclass_calibration_df,
    build_multiclass_class_df,
    build_multiclass_confusion_df,
    build_multiclass_metric_row,
    feature_manifest,
    fit_catboost_with_external_selection,
    get_git_dirty_flag,
    get_git_head,
    load_frame,
    parse_pipe_labels,
    prep_catboost_frames,
    prep_multi_label_cases,
    prep_single_label_cases,
    runtime_manifest,
    score_multiclass_from_proba,
    select_multilabel_threshold,
    sha256_path,
    split_multi_label_cases_by_mode,
    split_single_label_cases_by_mode,
    subset_case_frame,
    write_json,
)
from src.modeling.component_multilabel import (
    CATBOOST_NAME,
    DEF_CATBOOST_EVAL_PERIOD,
    DEF_CATBOOST_ITERS,
    fit_catboost_holdout_with_fallback,
    fit_catboost_selection_with_fallback,
)
from src.modeling.component_multilabel import (
    build_metric_row as build_multilabel_metric_row,
)

# -----------------------------------------------------------------------------
# Locked benchmark references
# -----------------------------------------------------------------------------
LOCKED_SINGLE_METRICS = OUTPUTS_DIR / 'component_single_label_benchmark_metrics.csv'
LOCKED_SINGLE_MANIFEST = OUTPUTS_DIR / 'component_single_label_benchmark_manifest.json'
LOCKED_SINGLE_SELECTION = OUTPUTS_DIR / 'component_single_label_selection_manifest.json'
LOCKED_SINGLE_CALIBRATION = OUTPUTS_DIR / 'component_single_label_holdout_calibration.csv'
LOCKED_MULTI_METRICS = OUTPUTS_DIR / 'component_multilabel_metrics.csv'
LOCKED_MULTI_MANIFEST = OUTPUTS_DIR / 'component_multilabel_manifest.json'


# -----------------------------------------------------------------------------
# Output names
# -----------------------------------------------------------------------------
GLOBAL_MANIFEST_NAME = 'component_textwave2_manifest.json'
SINGLE_SCREEN_NAME = 'component_single_label_textwave2_screen.csv'
SINGLE_SELECT_NAME = 'component_single_label_textwave2_select.csv'
SINGLE_HOLDOUT_NAME = 'component_single_label_textwave2_holdout.csv'
SINGLE_CLASS_NAME = 'component_single_label_textwave2_class_metrics.csv'
SINGLE_CONFUSION_NAME = 'component_single_label_textwave2_confusion_major.csv'
SINGLE_CALIB_NAME = 'component_single_label_textwave2_calibration.csv'
MULTI_SCREEN_NAME = 'component_multilabel_textwave2_screen.csv'
MULTI_SELECT_NAME = 'component_multilabel_textwave2_select.csv'
MULTI_HOLDOUT_NAME = 'component_multilabel_textwave2_holdout.csv'
MULTI_LABEL_NAME = 'component_multilabel_textwave2_label_metrics.csv'


# -----------------------------------------------------------------------------
# Wave 2 defaults
# -----------------------------------------------------------------------------
FEATUREWAVE_TASK = 'text_wave2'
STRUCTURED_FEATURE_SET = 'wave1_incident_cohort_history'
STRUCTURED_FAMILY = 'structured_carry_forward'
TEXT_ONLY_FAMILY = 'text_only_linear'
TEXT_PLUS_STRUCTURED_FAMILY = 'text_plus_structured_linear'
LATE_FUSION_FAMILY = 'text_structured_late_fusion'
ALL_FAMILIES = [
    STRUCTURED_FAMILY,
    TEXT_ONLY_FAMILY,
    TEXT_PLUS_STRUCTURED_FAMILY,
    LATE_FUSION_FAMILY
]
TEXT_FAMILIES = [
    TEXT_ONLY_FAMILY,
    TEXT_PLUS_STRUCTURED_FAMILY,
    LATE_FUSION_FAMILY
]
FUSION_TEXT_WEIGHTS = [0.25, 0.50, 0.75]
MULTI_THRESHOLDS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.30]
MIN_POSITIVE_LABELS = 1
STRUCTURED_SELECTION_EVAL_PERIOD = 10
STRUCTURED_MULTI_ITERATIONS = DEF_CATBOOST_ITERS
STRUCTURED_MULTI_EVAL_PERIOD = DEF_CATBOOST_EVAL_PERIOD
STRUCTURED_ONEHOT_MIN_FREQ = 50
OVR_N_JOBS = -1
LOG_SCALED_NUMERIC_COLS = {
    'miles',
    'veh_speed',
    'injured',
    'lag_days_safe',
    'prior_cmpl_mfr_all',
    'prior_cmpl_make_model_all',
    'prior_cmpl_make_model_year_all',
}
SINGLE_PROMOTE_SELECT_DELTA = 0.010
SINGLE_PROMOTE_HOLDOUT_DELTA = 0.010
SINGLE_TOP3_DROP_LIMIT = 0.005
SINGLE_ECE_WORSE_LIMIT = 0.020
MULTI_PROMOTE_SELECT_DELTA = 0.015
MULTI_PROMOTE_HOLDOUT_MACRO_DELTA = 0.015
MULTI_PROMOTE_HOLDOUT_MICRO_DELTA = 0.010
MULTI_LABEL_COVERAGE_FLOOR = 0.80
FINAL_LINEAR_MODEL_DEFAULT = 'sgd'
FINAL_LINEAR_MODEL_CHOICES = ['sgd']
FINAL_SGD_ALPHA = 1e-6
FINAL_SGD_MAX_ITER = 4000
FINAL_SGD_TOL = 1e-4
FINAL_SGD_VALIDATION_FRACTION = 0.05
FINAL_SGD_N_ITER_NO_CHANGE = 5

TEXT_SIDECAR_COLS = [
    ID_COL,
    'cdescr',
    'cdescr_model_text',
    'cdescr_missing_flag',
    'cdescr_placeholder_flag',
    'cdescr_char_len',
    'cdescr_word_count',
    'source_era',
    DATE_COL
]

TEXT_CONFIG = {
    'word_tfidf': {
        'analyzer': 'word',
        'ngram_range': [1, 2],
        'min_df': 5,
        'max_df': 0.995,
        'sublinear_tf': True,
        'lowercase': True,
        'strip_accents': 'unicode',
        'dtype': 'float32'
    },
    'char_tfidf': {
        'analyzer': 'char_wb',
        'ngram_range': [3, 5],
        'min_df': 5,
        'sublinear_tf': True,
        'lowercase': True,
        'strip_accents': 'unicode',
        'dtype': 'float32'
    },
    'fusion_text_weights': FUSION_TEXT_WEIGHTS,
    'multi_threshold_grid': MULTI_THRESHOLDS,
    'structured_feature_set': STRUCTURED_FEATURE_SET,
    'final_linear_model_default': FINAL_LINEAR_MODEL_DEFAULT
}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def log_line(message=''):
    print(message, flush=True)


def load_json(path):
    path = Path(path)
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def family_kind(family_name):
    if family_name == STRUCTURED_FAMILY:
        return 'structured'
    if family_name == LATE_FUSION_FAMILY:
        return 'fusion'
    return 'text_linear'


def base_row(task_name, family_name, input_path, text_sidecar_path):
    return {
        'task': task_name,
        'family_name': family_name,
        'family_kind': family_kind(family_name),
        'structured_feature_set': STRUCTURED_FEATURE_SET,
        'input_path': str(input_path),
        'input_sha256': sha256_path(input_path),
        'text_sidecar_path': str(text_sidecar_path),
        'text_sidecar_sha256': sha256_path(text_sidecar_path),
        'split_mode': FEATURE_WAVE1_SPLIT_MODE,
        'text_enabled': family_name != STRUCTURED_FAMILY
    }


def sort_rows(row_df, ranking_cols):
    if row_df.empty:
        return row_df
    return row_df.sort_values(ranking_cols, ascending=False).reset_index(drop=True)


def select_best_row(row_df, ranking_cols):
    ranked = sort_rows(row_df, ranking_cols)
    if ranked.empty:
        return None
    return ranked.iloc[0].to_dict()


def read_locked_single_select_baseline():
    metric_df = pd.read_csv(LOCKED_SINGLE_METRICS)
    row = metric_df.loc[
        metric_df['model'].eq('CatBoost')
        & metric_df['stage'].eq('selection_train_valid')
        & metric_df['split'].eq('valid_2025')
    ]
    if row.empty:
        raise ValueError('Locked single-label validation baseline row is missing')
    return row.iloc[0].to_dict()


def read_locked_multi_select_baseline():
    metric_df = pd.read_csv(LOCKED_MULTI_METRICS)
    row = metric_df.loc[
        metric_df['model'].eq(CATBOOST_NAME)
        & metric_df['stage'].eq('selection_train_valid')
        & metric_df['split'].eq('valid_2025')
    ]
    if row.empty:
        raise ValueError('Locked multi-label validation baseline row is missing')
    return row.iloc[0].to_dict()


def read_locked_single_ece():
    calib_df = pd.read_csv(LOCKED_SINGLE_CALIBRATION)
    overall = calib_df.loc[calib_df['section'].eq('overall')]
    if overall.empty:
        raise ValueError('Locked single-label calibration file is missing its overall row')
    return float(overall['ece'].iloc[0])


def empty_single_holdout_df():
    return pd.DataFrame(
        columns=[
            'task',
            'family_name',
            'family_kind',
            'structured_feature_set',
            'input_path',
            'input_sha256',
            'text_sidecar_path',
            'text_sidecar_sha256',
            'split_mode',
            'text_enabled',
            'model',
            'stage',
            'split',
            'rows',
            'fit_seconds',
            'selected_iteration',
            'top_1_accuracy',
            'macro_f1',
            'top_3_accuracy',
            'selected_text_weight',
            'prior_text_overlap'
        ]
    )


def empty_single_calibration_df():
    return pd.DataFrame(
        columns=[
            'section',
            'bin',
            'count',
            'share',
            'accuracy',
            'avg_confidence',
            'gap',
            'ece',
            'multiclass_brier'
        ]
    )


def empty_single_confusion_df():
    return pd.DataFrame(
        columns=[
            'true_group',
            'pred_group',
            'count',
            'row_share'
        ]
    )


def empty_multi_holdout_df():
    return pd.DataFrame(
        columns=[
            'task',
            'family_name',
            'family_kind',
            'structured_feature_set',
            'input_path',
            'input_sha256',
            'text_sidecar_path',
            'text_sidecar_sha256',
            'split_mode',
            'text_enabled',
            'model',
            'stage',
            'split',
            'rows',
            'fit_seconds',
            'selected_iteration',
            'threshold',
            'micro_f1',
            'macro_f1',
            'recall_at_3',
            'precision_at_3',
            'label_coverage',
            'selected_text_weight',
            'prior_text_overlap'
        ]
    )


def empty_multi_label_df():
    return pd.DataFrame(
        columns=[
            'component_group',
            'support',
            'precision',
            'recall',
            'f1'
        ]
    )


def write_single_outputs(result):
    result.get('screen_df', pd.DataFrame()).to_csv(OUTPUTS_DIR / SINGLE_SCREEN_NAME, index=False)
    result.get('select_df', pd.DataFrame()).to_csv(OUTPUTS_DIR / SINGLE_SELECT_NAME, index=False)
    result.get('holdout_df', empty_single_holdout_df()).to_csv(OUTPUTS_DIR / SINGLE_HOLDOUT_NAME, index=False)
    result.get('class_df', pd.DataFrame()).to_csv(OUTPUTS_DIR / SINGLE_CLASS_NAME, index=False)
    result.get('confusion_df', empty_single_confusion_df()).to_csv(OUTPUTS_DIR / SINGLE_CONFUSION_NAME, index=False)
    result.get('calibration_df', empty_single_calibration_df()).to_csv(OUTPUTS_DIR / SINGLE_CALIB_NAME, index=False)


def write_multi_outputs(result):
    result.get('screen_df', pd.DataFrame()).to_csv(OUTPUTS_DIR / MULTI_SCREEN_NAME, index=False)
    result.get('select_df', pd.DataFrame()).to_csv(OUTPUTS_DIR / MULTI_SELECT_NAME, index=False)
    result.get('holdout_df', empty_multi_holdout_df()).to_csv(OUTPUTS_DIR / MULTI_HOLDOUT_NAME, index=False)
    result.get('label_df', empty_multi_label_df()).to_csv(OUTPUTS_DIR / MULTI_LABEL_NAME, index=False)


def build_single_manifest_entry(result, locked_select_baseline, locked_holdout_baseline, locked_holdout_ece):
    return {
        'input_path': result.get('input_path'),
        'text_sidecar_path': result.get('text_sidecar_path'),
        'selected_family': result.get('selected_family'),
        'final_linear_model': result.get('final_linear_model'),
        'screen_fusion_weight': result.get('screen_fusion_weight'),
        'select_metrics': result.get('select_metrics'),
        'locked_select_baseline': locked_select_baseline,
        'locked_holdout_baseline': locked_holdout_baseline,
        'locked_holdout_ece': locked_holdout_ece,
        'select_gate_pass': result.get('select_gate_pass'),
        'promotion_status': result.get('promotion_status'),
        'holdout_overlap_slices': result.get('overlap_metrics', []),
        'checkpoint_stage': result.get('checkpoint_stage'),
        'completed_stages': result.get('completed_stages', []),
        'artifacts': {
            'screen': str(OUTPUTS_DIR / SINGLE_SCREEN_NAME),
            'select': str(OUTPUTS_DIR / SINGLE_SELECT_NAME),
            'holdout': str(OUTPUTS_DIR / SINGLE_HOLDOUT_NAME),
            'class_metrics': str(OUTPUTS_DIR / SINGLE_CLASS_NAME),
            'confusion_major': str(OUTPUTS_DIR / SINGLE_CONFUSION_NAME),
            'calibration': str(OUTPUTS_DIR / SINGLE_CALIB_NAME)
        }
    }


def build_multi_manifest_entry(result, locked_select_baseline, locked_holdout_baseline):
    return {
        'input_path': result.get('input_path'),
        'text_sidecar_path': result.get('text_sidecar_path'),
        'selected_family': result.get('selected_family'),
        'final_linear_model': result.get('final_linear_model'),
        'screen_fusion_weight': result.get('screen_fusion_weight'),
        'select_metrics': result.get('select_metrics'),
        'locked_select_baseline': locked_select_baseline,
        'locked_holdout_baseline': locked_holdout_baseline,
        'select_gate_pass': result.get('select_gate_pass'),
        'promotion_status': result.get('promotion_status'),
        'holdout_overlap_slices': result.get('overlap_metrics', []),
        'checkpoint_stage': result.get('checkpoint_stage'),
        'completed_stages': result.get('completed_stages', []),
        'artifacts': {
            'screen': str(OUTPUTS_DIR / MULTI_SCREEN_NAME),
            'select': str(OUTPUTS_DIR / MULTI_SELECT_NAME),
            'holdout': str(OUTPUTS_DIR / MULTI_HOLDOUT_NAME),
            'label_metrics': str(OUTPUTS_DIR / MULTI_LABEL_NAME)
        }
    }


def prepare_text_sidecar(sidecar_df):
    missing = [column for column in TEXT_SIDECAR_COLS if column not in sidecar_df.columns]
    if missing:
        missing_text = ', '.join(missing)
        raise ValueError(f'Text sidecar is missing required columns: {missing_text}')

    work = sidecar_df[TEXT_SIDECAR_COLS].copy()
    work[ID_COL] = work[ID_COL].astype('string').astype(str)
    work['cdescr'] = work['cdescr'].astype('string')
    work['cdescr_model_text'] = work['cdescr_model_text'].fillna('').astype(str)
    work['cdescr_missing_flag'] = work['cdescr_missing_flag'].fillna(False).astype(bool)
    work['cdescr_placeholder_flag'] = work['cdescr_placeholder_flag'].fillna(False).astype(bool)
    work['cdescr_char_len'] = pd.to_numeric(work['cdescr_char_len'], errors='coerce').fillna(0).astype('Int64')
    work['cdescr_word_count'] = pd.to_numeric(work['cdescr_word_count'], errors='coerce').fillna(0).astype('Int64')
    work['source_era'] = work['source_era'].astype('string')
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors='coerce')

    if work[ID_COL].duplicated().any():
        dupes = work.loc[work[ID_COL].duplicated(), ID_COL].head(5).tolist()
        raise ValueError(f'Text sidecar has duplicate odino rows: {dupes}')

    return work.sort_values(ID_COL).reset_index(drop=True)


def merge_text_sidecar(case_df, sidecar_df):
    work = case_df.copy()
    work[ID_COL] = work[ID_COL].astype('string').astype(str)
    text_df = prepare_text_sidecar(sidecar_df)
    keep_cols = [
        ID_COL,
        'cdescr',
        'cdescr_model_text',
        'cdescr_missing_flag',
        'cdescr_placeholder_flag',
        'cdescr_char_len',
        'cdescr_word_count',
        'source_era'
    ]
    merge_df = text_df[keep_cols].copy()
    if 'source_era' in work.columns:
        merge_df = merge_df.rename(columns={'source_era': 'source_era_text'})
    merged = work.merge(merge_df, on=ID_COL, how='left', validate='one_to_one')
    merged['cdescr'] = merged['cdescr'].astype('string')
    merged['cdescr_model_text'] = merged['cdescr_model_text'].fillna('').astype(str)
    merged['cdescr_missing_flag'] = merged['cdescr_missing_flag'].fillna(True).astype(bool)
    merged['cdescr_placeholder_flag'] = merged['cdescr_placeholder_flag'].fillna(False).astype(bool)
    merged['cdescr_char_len'] = pd.to_numeric(merged['cdescr_char_len'], errors='coerce').fillna(0).astype('Int64')
    merged['cdescr_word_count'] = pd.to_numeric(merged['cdescr_word_count'], errors='coerce').fillna(0).astype('Int64')
    if 'source_era_text' in merged.columns:
        merged['source_era'] = merged['source_era'].where(
            merged['source_era'].notna(),
            merged['source_era_text']
        )
        merged = merged.drop(columns=['source_era_text'])
    elif 'source_era' not in merged.columns:
        merged['source_era'] = pd.NA
    merged['source_era'] = merged['source_era'].astype('string')
    return merged


def build_overlap_mask(prior_text, later_text):
    prior = {
        text
        for text in pd.Series(prior_text).fillna('').astype(str).tolist()
        if text
    }
    later_series = pd.Series(later_text).fillna('').astype(str)
    return later_series.ne('').to_numpy() & later_series.isin(prior).to_numpy()


def build_word_vectorizer():
    return TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.995,
        sublinear_tf=True,
        lowercase=True,
        strip_accents='unicode',
        dtype=np.float32
    )


def build_char_vectorizer():
    return TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        min_df=5,
        sublinear_tf=True,
        lowercase=True,
        strip_accents='unicode',
        dtype=np.float32
    )


def fit_text_vectorizers(text_series):
    text_values = pd.Series(text_series).fillna('').astype(str)
    word_vec = build_word_vectorizer()
    char_vec = build_char_vectorizer()
    word_vec.fit(text_values)
    char_vec.fit(text_values)
    return {
        'word': word_vec,
        'char': char_vec
    }


def transform_text_matrix(vectorizers, text_series):
    text_values = pd.Series(text_series).fillna('').astype(str)
    word_matrix = vectorizers['word'].transform(text_values).astype(np.float32)
    char_matrix = vectorizers['char'].transform(text_values).astype(np.float32)
    return sparse.hstack(
        [
            word_matrix.multiply(0.5),
            char_matrix.multiply(0.5)
        ],
        format='csr'
    ).astype(np.float32)


def log1p_clip_nonnegative(values):
    array = np.asarray(values, dtype=np.float64)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.log1p(np.clip(array, a_min=0.0, a_max=None))


def build_structured_preprocessor(feature_info):
    cat_cols = list(feature_info['cat_cols'])
    num_cols = list(feature_info['num_cols'] + feature_info['flag_cols'])
    log_num_cols = [col for col in num_cols if col in LOG_SCALED_NUMERIC_COLS]
    linear_num_cols = [col for col in num_cols if col not in LOG_SCALED_NUMERIC_COLS]

    cat_pipe = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            (
                'encoder',
                OneHotEncoder(
                    handle_unknown='infrequent_if_exist',
                    min_frequency=STRUCTURED_ONEHOT_MIN_FREQ
                )
            )
        ]
    )
    transformers = [('cat', cat_pipe, cat_cols)]

    if linear_num_cols:
        linear_num_pipe = Pipeline(
            [
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False))
            ]
        )
        transformers.append(('num', linear_num_pipe, linear_num_cols))

    if log_num_cols:
        log_num_pipe = Pipeline(
            [
                ('imputer', SimpleImputer(strategy='median')),
                (
                    'log1p',
                    FunctionTransformer(
                        log1p_clip_nonnegative,
                        feature_names_out='one-to-one'
                    )
                ),
                ('scaler', StandardScaler(with_mean=False))
            ]
        )
        transformers.append(('num_log', log_num_pipe, log_num_cols))

    return ColumnTransformer(transformers, sparse_threshold=1.0)


def combine_matrices(text_matrix, structured_matrix, row_normalize=False):
    combined = sparse.hstack(
        [
            text_matrix,
            structured_matrix
        ],
        format='csr'
    ).astype(np.float32)
    if row_normalize:
        combined = normalize(combined, norm='l2', copy=False)
    return combined


def build_single_screen_model():
    return SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=1e-5,
        max_iter=2000,
        tol=1e-3,
        random_state=settings.RANDOM_SEED
    )


def build_single_final_model(model_kind=FINAL_LINEAR_MODEL_DEFAULT):
    if model_kind == 'sgd':
        return SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=FINAL_SGD_ALPHA,
            max_iter=FINAL_SGD_MAX_ITER,
            tol=FINAL_SGD_TOL,
            class_weight='balanced',
            average=True,
            early_stopping=True,
            validation_fraction=FINAL_SGD_VALIDATION_FRACTION,
            n_iter_no_change=FINAL_SGD_N_ITER_NO_CHANGE,
            random_state=settings.RANDOM_SEED
        )
    raise ValueError(f'Unsupported final linear model: {model_kind}')


def build_multi_screen_model():
    return OneVsRestClassifier(
        SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=1e-5,
            max_iter=2000,
            tol=1e-3,
            random_state=settings.RANDOM_SEED
        ),
        n_jobs=OVR_N_JOBS
    )


def build_multi_final_model(model_kind=FINAL_LINEAR_MODEL_DEFAULT):
    if model_kind == 'sgd':
        estimator = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=FINAL_SGD_ALPHA,
            max_iter=FINAL_SGD_MAX_ITER,
            tol=FINAL_SGD_TOL,
            class_weight='balanced',
            average=True,
            early_stopping=True,
            validation_fraction=FINAL_SGD_VALIDATION_FRACTION,
            n_iter_no_change=FINAL_SGD_N_ITER_NO_CHANGE,
            random_state=settings.RANDOM_SEED
        )
    else:
        raise ValueError(f'Unsupported final linear model: {model_kind}')

    return OneVsRestClassifier(estimator, n_jobs=OVR_N_JOBS)


def fit_single_linear(train_matrix, y_train, eval_matrix, final_model=False, final_model_kind=FINAL_LINEAR_MODEL_DEFAULT):
    model = build_single_final_model(final_model_kind) if final_model else build_single_screen_model()
    start = perf_counter()
    model.fit(train_matrix, y_train)
    fit_seconds = round(perf_counter() - start, 2)
    eval_proba = safe_single_predict_proba(model, eval_matrix)
    return {
        'model': model,
        'fit_seconds': fit_seconds,
        'classes': model.classes_,
        'eval_proba': eval_proba
    }


def fit_multi_linear(train_matrix, y_train, eval_matrix, final_model=False, final_model_kind=FINAL_LINEAR_MODEL_DEFAULT):
    model = build_multi_final_model(final_model_kind) if final_model else build_multi_screen_model()
    start = perf_counter()
    model.fit(train_matrix, y_train)
    fit_seconds = round(perf_counter() - start, 2)
    eval_proba = safe_multi_predict_proba(model, eval_matrix)
    return {
        'model': model,
        'fit_seconds': fit_seconds,
        'eval_proba': eval_proba
    }


def stable_softmax(decision):
    decision = np.asarray(decision, dtype=np.float64)
    decision = np.nan_to_num(decision, nan=0.0, posinf=1e6, neginf=-1e6)
    decision = decision - decision.max(axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(decision, -700, 700))
    row_sums = exp_scores.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze(axis=1) <= 0
    if np.any(zero_rows):
        exp_scores[zero_rows] = 1.0
        row_sums = exp_scores.sum(axis=1, keepdims=True)
    return exp_scores / row_sums


def stable_sigmoid(decision):
    decision = np.asarray(decision, dtype=np.float64)
    decision = np.nan_to_num(decision, nan=0.0, posinf=1e6, neginf=-1e6)
    return 1.0 / (1.0 + np.exp(-np.clip(decision, -700, 700)))


def ensure_finite_matrix(name, matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f'{name} contains non-finite values after stabilization')
    return matrix


def prefer_decision_function_path(model):
    if isinstance(model, SGDClassifier):
        return True
    return isinstance(model, OneVsRestClassifier) and isinstance(getattr(model, 'estimator', None), SGDClassifier)


def safe_single_predict_proba(model, matrix):
    if not prefer_decision_function_path(model):
        try:
            proba = model.predict_proba(matrix)
            if np.isfinite(proba).all():
                return np.asarray(proba, dtype=np.float64)
        except Exception:
            pass

    decision = model.decision_function(matrix)
    decision = np.asarray(decision)
    if decision.ndim == 1:
        positive = stable_sigmoid(decision)
        proba = np.column_stack([1.0 - positive, positive])
    else:
        proba = stable_softmax(decision)
    return ensure_finite_matrix('single-label probabilities', proba)


def safe_multi_predict_proba(model, matrix):
    if not prefer_decision_function_path(model):
        try:
            proba = model.predict_proba(matrix)
            if np.isfinite(proba).all():
                return np.asarray(proba, dtype=np.float64)
        except Exception:
            pass

    decision = model.decision_function(matrix)
    proba = stable_sigmoid(decision)
    return ensure_finite_matrix('multi-label probabilities', proba)


def align_single_proba(proba, source_classes, target_classes):
    source_index = {label: idx for idx, label in enumerate(source_classes)}
    aligned = np.zeros((proba.shape[0], len(target_classes)), dtype=np.float64)
    for target_idx, label in enumerate(target_classes):
        aligned[:, target_idx] = proba[:, source_index[label]]
    return aligned


def select_single_fusion_weight(y_true, text_proba, text_classes, structured_proba, structured_classes):
    target_classes = np.asarray(structured_classes)
    text_aligned = align_single_proba(text_proba, text_classes, target_classes)
    structured_aligned = align_single_proba(structured_proba, structured_classes, target_classes)
    best = None

    for weight in FUSION_TEXT_WEIGHTS:
        blended = float(weight) * text_aligned + (1.0 - float(weight)) * structured_aligned
        pred, metrics = score_multiclass_from_proba(y_true, blended, target_classes)
        candidate = {
            'selected_text_weight': float(weight),
            'pred': pred,
            'proba': blended,
            'classes': target_classes,
            **metrics
        }
        ranking = (
            candidate['macro_f1'],
            candidate['top_1_accuracy'],
            candidate['top_3_accuracy'],
            candidate['selected_text_weight']
        )
        if best is None:
            best = candidate
            continue

        best_ranking = (
            best['macro_f1'],
            best['top_1_accuracy'],
            best['top_3_accuracy'],
            best['selected_text_weight']
        )
        if ranking > best_ranking:
            best = candidate

    return best


def apply_single_fusion_weight(y_true, text_proba, text_classes, structured_proba, structured_classes, text_weight):
    target_classes = np.asarray(structured_classes)
    text_aligned = align_single_proba(text_proba, text_classes, target_classes)
    structured_aligned = align_single_proba(structured_proba, structured_classes, target_classes)
    blended = float(text_weight) * text_aligned + (1.0 - float(text_weight)) * structured_aligned
    pred, metrics = score_multiclass_from_proba(y_true, blended, target_classes)
    return {
        'selected_text_weight': float(text_weight),
        'pred': pred,
        'proba': blended,
        'classes': target_classes,
        **metrics
    }


def select_multi_fusion_weight(y_true, text_proba, structured_proba):
    best = None
    for weight in FUSION_TEXT_WEIGHTS:
        blended = float(weight) * text_proba + (1.0 - float(weight)) * structured_proba
        threshold_choice = select_multilabel_threshold(
            y_true,
            blended,
            thresholds=MULTI_THRESHOLDS,
            min_positive_labels=MIN_POSITIVE_LABELS
        )
        pred = apply_multilabel_threshold(
            blended,
            threshold_choice['threshold'],
            min_positive_labels=MIN_POSITIVE_LABELS
        )
        candidate = {
            'selected_text_weight': float(weight),
            'selected_threshold': float(threshold_choice['threshold']),
            'pred': pred,
            'proba': blended,
            **threshold_choice
        }
        ranking = (
            candidate['macro_f1'],
            candidate['micro_f1'],
            candidate['recall_at_3'],
            candidate['precision_at_3'],
            -candidate['selected_threshold'],
            candidate['selected_text_weight']
        )
        if best is None:
            best = candidate
            continue

        best_ranking = (
            best['macro_f1'],
            best['micro_f1'],
            best['recall_at_3'],
            best['precision_at_3'],
            -best['selected_threshold'],
            best['selected_text_weight']
        )
        if ranking > best_ranking:
            best = candidate

    return best


def apply_multi_fusion_weight(y_true, text_proba, structured_proba, text_weight):
    blended = float(text_weight) * text_proba + (1.0 - float(text_weight)) * structured_proba
    threshold_choice = select_multilabel_threshold(
        y_true,
        blended,
        thresholds=MULTI_THRESHOLDS,
        min_positive_labels=MIN_POSITIVE_LABELS
    )
    pred = apply_multilabel_threshold(
        blended,
        threshold_choice['threshold'],
        min_positive_labels=MIN_POSITIVE_LABELS
    )
    return {
        'selected_text_weight': float(text_weight),
        'selected_threshold': float(threshold_choice['threshold']),
        'pred': pred,
        'proba': blended,
        **threshold_choice
    }


def build_single_row(task_name, family_name, input_path, text_sidecar_path, stage_name, split_name, y_true, proba, classes, fit_seconds=np.nan, selected_iteration=np.nan, selected_text_weight=np.nan, prior_text_overlap='all'):
    return {
        **base_row(task_name, family_name, input_path, text_sidecar_path),
        **build_multiclass_metric_row(
            family_name,
            stage_name,
            split_name,
            y_true,
            proba,
            classes,
            fit_seconds=fit_seconds,
            selected_iteration=selected_iteration
        ),
        'selected_text_weight': selected_text_weight,
        'prior_text_overlap': prior_text_overlap
    }


def build_multi_row(task_name, family_name, input_path, text_sidecar_path, stage_name, split_name, y_true, pred, proba, threshold=np.nan, fit_seconds=np.nan, selected_iteration=np.nan, selected_text_weight=np.nan, prior_text_overlap='all'):
    return {
        **base_row(task_name, family_name, input_path, text_sidecar_path),
        **build_multilabel_metric_row(
            family_name,
            stage_name,
            split_name,
            y_true,
            pred,
            proba,
            threshold=threshold,
            fit_seconds=fit_seconds,
            selected_iteration=selected_iteration
        ),
        'selected_text_weight': selected_text_weight,
        'prior_text_overlap': prior_text_overlap
    }


def fit_single_text_family(train_df, eval_df, structured_feature_info, family_name, final_model=False, final_model_kind=FINAL_LINEAR_MODEL_DEFAULT):
    text_vectorizers = fit_text_vectorizers(train_df['cdescr_model_text'])
    X_train_text = transform_text_matrix(text_vectorizers, train_df['cdescr_model_text'])
    X_eval_text = transform_text_matrix(text_vectorizers, eval_df['cdescr_model_text'])

    if family_name == TEXT_ONLY_FAMILY:
        X_train = X_train_text
        X_eval = X_eval_text
    elif family_name == TEXT_PLUS_STRUCTURED_FAMILY:
        preprocessor = build_structured_preprocessor(structured_feature_info)
        X_train_struct = preprocessor.fit_transform(train_df[structured_feature_info['feature_cols']])
        X_eval_struct = preprocessor.transform(eval_df[structured_feature_info['feature_cols']])
        X_train = combine_matrices(X_train_text, X_train_struct, row_normalize=True)
        X_eval = combine_matrices(X_eval_text, X_eval_struct, row_normalize=True)
    else:
        raise ValueError(f'Unsupported single text family: {family_name}')

    result = fit_single_linear(
        X_train,
        train_df[TARGET_COL].astype(str),
        X_eval,
        final_model=final_model,
        final_model_kind=final_model_kind
    )
    pred, metrics = score_multiclass_from_proba(
        eval_df[TARGET_COL].astype(str),
        result['eval_proba'],
        result['classes']
    )
    return {
        **result,
        'pred': pred,
        'metrics': metrics
    }


def fit_single_structured_family(train_df, eval_df, structured_feature_info, params, task_type, devices, random_seed):
    train_ready = subset_case_frame(train_df, structured_feature_info['feature_cols'], target_col=TARGET_COL)
    eval_ready = subset_case_frame(eval_df, structured_feature_info['feature_cols'], target_col=TARGET_COL)
    return fit_catboost_with_external_selection(
        train_ready,
        eval_ready,
        structured_feature_info,
        params,
        task_type=task_type,
        devices=devices,
        random_seed=random_seed,
        verbose=0,
        selection_eval_period=STRUCTURED_SELECTION_EVAL_PERIOD,
        include_train_outputs=False,
        include_valid_outputs=True
    )


def fit_single_structured_holdout(dev_df, holdout_df, structured_feature_info, params, selected_iteration, task_type, devices, random_seed):
    dev_ready = subset_case_frame(dev_df, structured_feature_info['feature_cols'], target_col=TARGET_COL)
    holdout_ready = subset_case_frame(holdout_df, structured_feature_info['feature_cols'], target_col=TARGET_COL)
    X_dev, X_holdout = prep_catboost_frames(dev_ready, holdout_ready, structured_feature_info)
    y_dev = dev_ready[TARGET_COL].astype(str)

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
        X_dev,
        y_dev,
        cat_features=structured_feature_info['cat_cols'],
        use_best_model=False
    )
    fit_seconds = round(perf_counter() - start, 2)
    holdout_proba = model.predict_proba(X_holdout)
    holdout_pred, holdout_metrics = score_multiclass_from_proba(
        holdout_df[TARGET_COL].astype(str),
        holdout_proba,
        model.classes_
    )
    return {
        'fit_seconds': fit_seconds,
        'proba': holdout_proba,
        'pred': holdout_pred,
        'classes': model.classes_,
        'metrics': holdout_metrics
    }


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


def fit_multi_text_family(train_df, eval_df, structured_feature_info, family_name, final_model=False, final_model_kind=FINAL_LINEAR_MODEL_DEFAULT):
    train_labels = parse_pipe_labels(train_df[MULTI_TARGET_COL])
    eval_labels = parse_pipe_labels(eval_df[MULTI_TARGET_COL])
    check_unseen_multilabel_labels(train_labels, eval_labels, family_name)
    mlb, y_train, y_eval = build_multilabel_encoded(train_labels, eval_labels)

    text_vectorizers = fit_text_vectorizers(train_df['cdescr_model_text'])
    X_train_text = transform_text_matrix(text_vectorizers, train_df['cdescr_model_text'])
    X_eval_text = transform_text_matrix(text_vectorizers, eval_df['cdescr_model_text'])

    if family_name == TEXT_ONLY_FAMILY:
        X_train = X_train_text
        X_eval = X_eval_text
    elif family_name == TEXT_PLUS_STRUCTURED_FAMILY:
        preprocessor = build_structured_preprocessor(structured_feature_info)
        X_train_struct = preprocessor.fit_transform(train_df[structured_feature_info['feature_cols']])
        X_eval_struct = preprocessor.transform(eval_df[structured_feature_info['feature_cols']])
        X_train = combine_matrices(X_train_text, X_train_struct, row_normalize=True)
        X_eval = combine_matrices(X_eval_text, X_eval_struct, row_normalize=True)
    else:
        raise ValueError(f'Unsupported multi text family: {family_name}')

    result = fit_multi_linear(
        X_train,
        y_train,
        X_eval,
        final_model=final_model,
        final_model_kind=final_model_kind
    )
    threshold_choice = select_multilabel_threshold(
        y_eval,
        result['eval_proba'],
        thresholds=MULTI_THRESHOLDS,
        min_positive_labels=MIN_POSITIVE_LABELS
    )
    pred = apply_multilabel_threshold(
        result['eval_proba'],
        threshold_choice['threshold'],
        min_positive_labels=MIN_POSITIVE_LABELS
    )
    return {
        **result,
        'mlb': mlb,
        'y_train': y_train,
        'y_eval': y_eval,
        'pred': pred,
        'threshold_choice': threshold_choice
    }


def fit_multi_structured_family(train_df, eval_df, structured_feature_info, task_type, devices, random_seed):
    train_labels = parse_pipe_labels(train_df[MULTI_TARGET_COL])
    eval_labels = parse_pipe_labels(eval_df[MULTI_TARGET_COL])
    check_unseen_multilabel_labels(train_labels, eval_labels, 'structured_carry_forward')
    _, y_train, y_eval = build_multilabel_encoded(train_labels, eval_labels)
    result = fit_catboost_selection_with_fallback(
        train_df,
        eval_df,
        y_train,
        y_eval,
        structured_feature_info,
        task_type=task_type,
        devices=devices,
        random_seed=random_seed,
        verbose=0,
        iterations=STRUCTURED_MULTI_ITERATIONS,
        eval_period=STRUCTURED_MULTI_EVAL_PERIOD,
        thresholds=MULTI_THRESHOLDS,
        min_positive_labels=MIN_POSITIVE_LABELS
    )
    return {
        **result,
        'y_eval': y_eval
    }


def fit_multi_structured_holdout(dev_df, holdout_df, structured_feature_info, selection_result, devices, random_seed):
    dev_labels = parse_pipe_labels(dev_df[MULTI_TARGET_COL])
    holdout_labels = parse_pipe_labels(holdout_df[MULTI_TARGET_COL])
    check_unseen_multilabel_labels(dev_labels, holdout_labels, 'holdout_2026')
    mlb, y_dev, y_holdout = build_multilabel_encoded(dev_labels, holdout_labels)
    result = fit_catboost_holdout_with_fallback(
        dev_df,
        holdout_df,
        y_dev,
        structured_feature_info,
        task_type=selection_result['actual_task_type'],
        devices=devices,
        random_seed=random_seed,
        verbose=0,
        selected_iteration=selection_result['selected_iteration'],
        selected_threshold=selection_result['selected_threshold'],
        min_positive_labels=MIN_POSITIVE_LABELS
    )
    return {
        **result,
        'mlb': mlb,
        'y_holdout': y_holdout
    }


def build_single_overlap_rows(base_row_dict, y_true, proba, classes, overlap_mask):
    rows = []
    for overlap_value, overlap_label in [(True, 'true'), (False, 'false')]:
        mask = overlap_mask == overlap_value
        if not mask.any():
            continue
        rows.append(
            build_single_row(
                base_row_dict['task'],
                base_row_dict['family_name'],
                base_row_dict['input_path'],
                base_row_dict['text_sidecar_path'],
                base_row_dict['stage'],
                base_row_dict['split'],
                np.asarray(y_true)[mask],
                proba[mask],
                classes,
                fit_seconds=base_row_dict['fit_seconds'],
                selected_iteration=base_row_dict.get('selected_iteration', np.nan),
                selected_text_weight=base_row_dict.get('selected_text_weight', np.nan),
                prior_text_overlap=overlap_label
            )
        )
    return rows


def build_multi_overlap_rows(base_row_dict, y_true, pred, proba, overlap_mask):
    rows = []
    for overlap_value, overlap_label in [(True, 'true'), (False, 'false')]:
        mask = overlap_mask == overlap_value
        if not mask.any():
            continue
        rows.append(
            build_multi_row(
                base_row_dict['task'],
                base_row_dict['family_name'],
                base_row_dict['input_path'],
                base_row_dict['text_sidecar_path'],
                base_row_dict['stage'],
                base_row_dict['split'],
                np.asarray(y_true)[mask],
                pred[mask],
                proba[mask],
                threshold=base_row_dict.get('threshold', np.nan),
                fit_seconds=base_row_dict['fit_seconds'],
                selected_iteration=base_row_dict.get('selected_iteration', np.nan),
                selected_text_weight=base_row_dict.get('selected_text_weight', np.nan),
                prior_text_overlap=overlap_label
            )
        )
    return rows


def log_single_family(stage_name, family_name, row):
    log_line(
        f'[single] {stage_name} {family_name} '
        f'macro_f1={float(row["macro_f1"]):.4f} '
        f'top1={float(row["top_1_accuracy"]):.4f} '
        f'top3={float(row["top_3_accuracy"]):.4f}'
    )


def log_multi_family(stage_name, family_name, row):
    log_line(
        f'[multi] {stage_name} {family_name} '
        f'macro_f1={float(row["macro_f1"]):.4f} '
        f'micro_f1={float(row["micro_f1"]):.4f} '
        f'recall@3={float(row["recall_at_3"]):.4f}'
    )


# -----------------------------------------------------------------------------
# Single-label wave
# -----------------------------------------------------------------------------
def run_single_wave(args, structured_feature_info, locked_single_select_row, locked_single_manifest, locked_single_ece, locked_single_selection, checkpoint_fn=None):
    screen_rows = []
    select_rows = []
    holdout_rows = []
    class_df = pd.DataFrame()
    confusion_df = empty_single_confusion_df()
    calibration_df = empty_single_calibration_df()
    overlap_metrics = []
    completed_stages = []

    raw_df, input_path = load_frame(SINGLE_INPUT_STEM, input_path=args.single_input_path)
    sidecar_df, text_sidecar_path = load_frame(SIDECAR_STEM, input_path=args.text_sidecar_path)
    case_df = prep_single_label_cases(raw_df, structured_feature_info['feature_cols'])
    case_df = merge_text_sidecar(case_df, sidecar_df)
    split_parts = split_single_label_cases_by_mode(case_df, split_mode=FEATURE_WAVE1_SPLIT_MODE)

    train_core_df = split_parts['train_core']
    screen_df = split_parts['screen_2024']
    dev_screen_df = split_parts['dev_2020_2024']
    select_df = split_parts['select_2025']
    dev_select_df = split_parts['dev_2020_2025']
    holdout_df = split_parts['holdout_2026']
    locked_params = locked_single_selection['best_params']

    log_line(
        f'[single] Split rows | train_core={len(train_core_df):,} screen_2024={len(screen_df):,} '
        f'select_2025={len(select_df):,} holdout_2026={len(holdout_df):,} '
        f'final_linear_model={args.final_linear_model}'
    )

    def emit_checkpoint(checkpoint_stage, selected_family=None, select_metrics=None, select_gate_pass=None, promotion_status='running'):
        if checkpoint_fn is None:
            return
        checkpoint_fn(
            checkpoint_stage,
            {
                'input_path': str(input_path),
                'text_sidecar_path': str(text_sidecar_path),
                'split_df': split_parts['split_df'],
                'screen_df': pd.DataFrame(screen_rows),
                'select_df': pd.DataFrame(select_rows),
                'holdout_df': pd.DataFrame(holdout_rows) if holdout_rows else empty_single_holdout_df(),
                'class_df': class_df,
                'confusion_df': confusion_df,
                'calibration_df': calibration_df,
                'selected_family': selected_family,
                'final_linear_model': args.final_linear_model,
                'screen_fusion_weight': late_screen['selected_text_weight'] if 'late_screen' in locals() else None,
                'select_metrics': select_metrics,
                'select_gate_pass': select_gate_pass,
                'promotion_status': promotion_status,
                'overlap_metrics': overlap_metrics,
                'checkpoint_stage': checkpoint_stage,
                'completed_stages': list(completed_stages)
            }
        )

    structured_screen = fit_single_structured_family(
        train_core_df,
        screen_df,
        structured_feature_info,
        locked_params,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed
    )
    structured_screen_row = build_single_row(
        'single_label',
        STRUCTURED_FAMILY,
        input_path,
        text_sidecar_path,
        'screen_2024',
        'screen_2024',
        screen_df[TARGET_COL].astype(str),
        structured_screen['valid_proba'],
        structured_screen['classes'],
        fit_seconds=structured_screen['fit_seconds'],
        selected_iteration=structured_screen['selected_iteration']
    )
    screen_rows.append(structured_screen_row)
    log_single_family('screen_2024', STRUCTURED_FAMILY, structured_screen_row)

    text_only_screen = fit_single_text_family(
        train_core_df,
        screen_df,
        structured_feature_info,
        TEXT_ONLY_FAMILY,
        final_model=False
    )
    text_only_screen_row = build_single_row(
        'single_label',
        TEXT_ONLY_FAMILY,
        input_path,
        text_sidecar_path,
        'screen_2024',
        'screen_2024',
        screen_df[TARGET_COL].astype(str),
        text_only_screen['eval_proba'],
        text_only_screen['classes'],
        fit_seconds=text_only_screen['fit_seconds']
    )
    screen_rows.append(text_only_screen_row)
    log_single_family('screen_2024', TEXT_ONLY_FAMILY, text_only_screen_row)

    if args.skip_text_plus:
        log_line('[single] screen_2024 text_plus_structured_linear skipped by flag')
    else:
        text_plus_screen = fit_single_text_family(
            train_core_df,
            screen_df,
            structured_feature_info,
            TEXT_PLUS_STRUCTURED_FAMILY,
            final_model=False
        )
        text_plus_screen_row = build_single_row(
            'single_label',
            TEXT_PLUS_STRUCTURED_FAMILY,
            input_path,
            text_sidecar_path,
            'screen_2024',
            'screen_2024',
            screen_df[TARGET_COL].astype(str),
            text_plus_screen['eval_proba'],
            text_plus_screen['classes'],
            fit_seconds=text_plus_screen['fit_seconds']
        )
        screen_rows.append(text_plus_screen_row)
        log_single_family('screen_2024', TEXT_PLUS_STRUCTURED_FAMILY, text_plus_screen_row)

    late_screen = select_single_fusion_weight(
        screen_df[TARGET_COL].astype(str),
        text_only_screen['eval_proba'],
        text_only_screen['classes'],
        structured_screen['valid_proba'],
        structured_screen['classes']
    )
    late_screen_row = build_single_row(
        'single_label',
        LATE_FUSION_FAMILY,
        input_path,
        text_sidecar_path,
        'screen_2024',
        'screen_2024',
        screen_df[TARGET_COL].astype(str),
        late_screen['proba'],
        late_screen['classes'],
        fit_seconds=round(float(text_only_screen['fit_seconds']) + float(structured_screen['fit_seconds']), 2),
        selected_iteration=structured_screen['selected_iteration'],
        selected_text_weight=late_screen['selected_text_weight']
    )
    screen_rows.append(late_screen_row)
    log_single_family('screen_2024', LATE_FUSION_FAMILY, late_screen_row)
    completed_stages.append('screen_2024')
    emit_checkpoint('screen_2024_complete')

    structured_select = fit_single_structured_family(
        dev_screen_df,
        select_df,
        structured_feature_info,
        locked_params,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed
    )
    structured_select_row = build_single_row(
        'single_label',
        STRUCTURED_FAMILY,
        input_path,
        text_sidecar_path,
        'select_2025',
        'select_2025',
        select_df[TARGET_COL].astype(str),
        structured_select['valid_proba'],
        structured_select['classes'],
        fit_seconds=structured_select['fit_seconds'],
        selected_iteration=structured_select['selected_iteration']
    )
    select_rows.append(structured_select_row)
    log_single_family('select_2025', STRUCTURED_FAMILY, structured_select_row)

    text_only_select = fit_single_text_family(
        dev_screen_df,
        select_df,
        structured_feature_info,
        TEXT_ONLY_FAMILY,
        final_model=False
    )
    text_only_select_row = build_single_row(
        'single_label',
        TEXT_ONLY_FAMILY,
        input_path,
        text_sidecar_path,
        'select_2025',
        'select_2025',
        select_df[TARGET_COL].astype(str),
        text_only_select['eval_proba'],
        text_only_select['classes'],
        fit_seconds=text_only_select['fit_seconds']
    )
    select_rows.append(text_only_select_row)
    log_single_family('select_2025', TEXT_ONLY_FAMILY, text_only_select_row)

    if args.skip_text_plus:
        log_line('[single] select_2025 text_plus_structured_linear skipped by flag')
    else:
        text_plus_select = fit_single_text_family(
            dev_screen_df,
            select_df,
            structured_feature_info,
            TEXT_PLUS_STRUCTURED_FAMILY,
            final_model=False
        )
        text_plus_select_row = build_single_row(
            'single_label',
            TEXT_PLUS_STRUCTURED_FAMILY,
            input_path,
            text_sidecar_path,
            'select_2025',
            'select_2025',
            select_df[TARGET_COL].astype(str),
            text_plus_select['eval_proba'],
            text_plus_select['classes'],
            fit_seconds=text_plus_select['fit_seconds']
        )
        select_rows.append(text_plus_select_row)
        log_single_family('select_2025', TEXT_PLUS_STRUCTURED_FAMILY, text_plus_select_row)

    late_select = apply_single_fusion_weight(
        select_df[TARGET_COL].astype(str),
        text_only_select['eval_proba'],
        text_only_select['classes'],
        structured_select['valid_proba'],
        structured_select['classes'],
        text_weight=late_screen['selected_text_weight']
    )
    late_select_row = build_single_row(
        'single_label',
        LATE_FUSION_FAMILY,
        input_path,
        text_sidecar_path,
        'select_2025',
        'select_2025',
        select_df[TARGET_COL].astype(str),
        late_select['proba'],
        late_select['classes'],
        fit_seconds=round(float(text_only_select['fit_seconds']) + float(structured_select['fit_seconds']), 2),
        selected_iteration=structured_select['selected_iteration'],
        selected_text_weight=late_screen['selected_text_weight']
    )
    select_rows.append(late_select_row)
    log_single_family('select_2025', LATE_FUSION_FAMILY, late_select_row)

    select_df_all = pd.DataFrame(select_rows)
    current_best_row = select_best_row(
        select_df_all,
        ['macro_f1', 'top_1_accuracy', 'top_3_accuracy']
    )
    selected_family = current_best_row['family_name']
    select_improvement = float(current_best_row['macro_f1'] - locked_single_select_row['macro_f1'])
    select_gate_pass = select_improvement >= SINGLE_PROMOTE_SELECT_DELTA
    promotion_status = 'rejected_select'
    log_line(
        f'[single] select_2025 best={selected_family} '
        f'macro_f1={float(current_best_row["macro_f1"]):.4f} '
        f'delta_vs_locked={select_improvement:+.4f} '
        f'gate_pass={str(select_gate_pass).lower()}'
    )
    completed_stages.append('select_2025')
    emit_checkpoint(
        'select_2025_complete',
        selected_family=selected_family,
        select_metrics=current_best_row,
        select_gate_pass=select_gate_pass,
        promotion_status='running'
    )

    if select_gate_pass:
        log_line(f'[single] holdout_2026 start family={selected_family}')
        emit_checkpoint(
            'holdout_2026_started',
            selected_family=selected_family,
            select_metrics=current_best_row,
            select_gate_pass=select_gate_pass,
            promotion_status='running'
        )
        if selected_family == STRUCTURED_FAMILY:
            log_line(
                f'[single] holdout_2026 fitting structured carry-forward '
                f'iteration={int(structured_select["selected_iteration"])}'
            )
            holdout_result = fit_single_structured_holdout(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                locked_params,
                structured_select['selected_iteration'],
                task_type=str(args.task_type).upper(),
                devices=args.devices,
                random_seed=args.random_seed
            )
            log_line(
                f'[single] holdout_2026 structured fit complete '
                f'fit_seconds={float(holdout_result["fit_seconds"]):.2f}'
            )
            holdout_row = build_single_row(
                'single_label',
                STRUCTURED_FAMILY,
                input_path,
                text_sidecar_path,
                'final_holdout',
                'holdout_2026',
                holdout_df[TARGET_COL].astype(str),
                holdout_result['proba'],
                holdout_result['classes'],
                fit_seconds=holdout_result['fit_seconds'],
                selected_iteration=structured_select['selected_iteration']
            )
        elif selected_family == TEXT_ONLY_FAMILY:
            log_line(
                f'[single] holdout_2026 fitting final text-only linear model '
                f'dev_rows={len(dev_select_df):,} '
                f'final_model={args.final_linear_model}'
            )
            holdout_result = fit_single_text_family(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                TEXT_ONLY_FAMILY,
                final_model=True,
                final_model_kind=args.final_linear_model
            )
            log_line(
                f'[single] holdout_2026 text-only fit complete '
                f'fit_seconds={float(holdout_result["fit_seconds"]):.2f}'
            )
            holdout_result = {
                'fit_seconds': holdout_result['fit_seconds'],
                'proba': holdout_result['eval_proba'],
                'pred': holdout_result['pred'],
                'classes': holdout_result['classes']
            }
            holdout_row = build_single_row(
                'single_label',
                TEXT_ONLY_FAMILY,
                input_path,
                text_sidecar_path,
                'final_holdout',
                'holdout_2026',
                holdout_df[TARGET_COL].astype(str),
                holdout_result['proba'],
                holdout_result['classes'],
                fit_seconds=holdout_result['fit_seconds']
            )
        elif selected_family == TEXT_PLUS_STRUCTURED_FAMILY:
            log_line(
                f'[single] holdout_2026 fitting final text+structured linear model '
                f'dev_rows={len(dev_select_df):,} '
                f'final_model={args.final_linear_model}'
            )
            holdout_result = fit_single_text_family(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                TEXT_PLUS_STRUCTURED_FAMILY,
                final_model=True,
                final_model_kind=args.final_linear_model
            )
            log_line(
                f'[single] holdout_2026 text+structured fit complete '
                f'fit_seconds={float(holdout_result["fit_seconds"]):.2f}'
            )
            holdout_result = {
                'fit_seconds': holdout_result['fit_seconds'],
                'proba': holdout_result['eval_proba'],
                'pred': holdout_result['pred'],
                'classes': holdout_result['classes']
            }
            holdout_row = build_single_row(
                'single_label',
                TEXT_PLUS_STRUCTURED_FAMILY,
                input_path,
                text_sidecar_path,
                'final_holdout',
                'holdout_2026',
                holdout_df[TARGET_COL].astype(str),
                holdout_result['proba'],
                holdout_result['classes'],
                fit_seconds=holdout_result['fit_seconds']
            )
        else:
            log_line(
                f'[single] holdout_2026 fitting late-fusion structured branch '
                f'iteration={int(structured_select["selected_iteration"])}'
            )
            structured_holdout = fit_single_structured_holdout(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                locked_params,
                structured_select['selected_iteration'],
                task_type=str(args.task_type).upper(),
                devices=args.devices,
                random_seed=args.random_seed
            )
            log_line(
                f'[single] holdout_2026 structured branch complete '
                f'fit_seconds={float(structured_holdout["fit_seconds"]):.2f}'
            )
            log_line(
                f'[single] holdout_2026 fitting late-fusion text branch '
                f'text_weight={float(late_screen["selected_text_weight"]):.2f} '
                f'dev_rows={len(dev_select_df):,} '
                f'final_model={args.final_linear_model}'
            )
            text_holdout = fit_single_text_family(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                TEXT_ONLY_FAMILY,
                final_model=True,
                final_model_kind=args.final_linear_model
            )
            log_line(
                f'[single] holdout_2026 text branch complete '
                f'fit_seconds={float(text_holdout["fit_seconds"]):.2f}'
            )
            late_holdout = apply_single_fusion_weight(
                holdout_df[TARGET_COL].astype(str),
                text_holdout['eval_proba'],
                text_holdout['classes'],
                structured_holdout['proba'],
                structured_holdout['classes'],
                text_weight=late_screen['selected_text_weight']
            )
            holdout_result = {
                'fit_seconds': round(float(text_holdout['fit_seconds']) + float(structured_holdout['fit_seconds']), 2),
                'proba': late_holdout['proba'],
                'pred': late_holdout['pred'],
                'classes': late_holdout['classes']
            }
            holdout_row = build_single_row(
                'single_label',
                LATE_FUSION_FAMILY,
                input_path,
                text_sidecar_path,
                'final_holdout',
                'holdout_2026',
                holdout_df[TARGET_COL].astype(str),
                holdout_result['proba'],
                holdout_result['classes'],
                fit_seconds=holdout_result['fit_seconds'],
                selected_iteration=structured_select['selected_iteration'],
                selected_text_weight=late_screen['selected_text_weight']
            )

        holdout_rows.append(holdout_row)
        log_single_family('holdout_2026', selected_family, holdout_row)
        class_df = build_multiclass_class_df(
            holdout_df[TARGET_COL].astype(str),
            holdout_result['pred'],
            holdout_result['classes']
        )
        calibration_df = build_multiclass_calibration_df(
            holdout_df[TARGET_COL].astype(str),
            holdout_result['proba'],
            holdout_result['classes']
        )
        focus_groups = (
            dev_select_df[TARGET_COL]
            .value_counts()
            .head(12)
            .index
            .tolist()
        )
        confusion_df = build_multiclass_confusion_df(
            holdout_df[TARGET_COL].astype(str),
            holdout_result['pred'],
            focus_groups
        )

        if selected_family in TEXT_FAMILIES:
            overlap_mask = build_overlap_mask(
                dev_select_df['cdescr_model_text'],
                holdout_df['cdescr_model_text']
            )
            slice_rows = build_single_overlap_rows(
                holdout_row,
                holdout_df[TARGET_COL].astype(str),
                holdout_result['proba'],
                holdout_result['classes'],
                overlap_mask
            )
            holdout_rows.extend(slice_rows)
            overlap_metrics = slice_rows

        holdout_macro_gain = float(holdout_row['macro_f1'] - locked_single_manifest['official_holdout_metrics']['macro_f1'])
        holdout_top3_ok = holdout_row['top_3_accuracy'] >= (
            locked_single_manifest['official_holdout_metrics']['top_3_accuracy'] - SINGLE_TOP3_DROP_LIMIT
        )
        holdout_ece = float(calibration_df.loc[calibration_df['section'].eq('overall'), 'ece'].iloc[0])
        holdout_ece_ok = (holdout_ece - locked_single_ece) <= SINGLE_ECE_WORSE_LIMIT
        promotion_status = 'promoted' if (
            holdout_macro_gain >= SINGLE_PROMOTE_HOLDOUT_DELTA
            and holdout_top3_ok
            and holdout_ece_ok
        ) else 'rejected_holdout'
        log_line(
            f'[single] holdout_2026 promotion_status={promotion_status} '
            f'macro_gain={holdout_macro_gain:+.4f} '
            f'top3_ok={str(bool(holdout_top3_ok)).lower()} '
            f'ece_delta={(holdout_ece - locked_single_ece):+.4f}'
        )
        completed_stages.append('holdout_2026')
        emit_checkpoint(
            'holdout_2026_complete',
            selected_family=selected_family,
            select_metrics=current_best_row,
            select_gate_pass=select_gate_pass,
            promotion_status=promotion_status
        )
    else:
        log_line(
            f'[single] select_2025 gate rejected; skipping holdout '
            f'family={selected_family}'
        )
        emit_checkpoint(
            'select_gate_rejected',
            selected_family=selected_family,
            select_metrics=current_best_row,
            select_gate_pass=select_gate_pass,
            promotion_status=promotion_status
        )

    return {
        'input_path': str(input_path),
        'text_sidecar_path': str(text_sidecar_path),
        'split_df': split_parts['split_df'],
        'screen_df': pd.DataFrame(screen_rows),
        'select_df': select_df_all,
        'holdout_df': pd.DataFrame(holdout_rows) if holdout_rows else empty_single_holdout_df(),
        'class_df': class_df,
        'confusion_df': confusion_df,
        'calibration_df': calibration_df,
        'selected_family': selected_family,
        'final_linear_model': args.final_linear_model,
        'screen_fusion_weight': late_screen['selected_text_weight'],
        'select_metrics': current_best_row,
        'select_gate_pass': select_gate_pass,
        'promotion_status': promotion_status,
        'overlap_metrics': overlap_metrics,
        'checkpoint_stage': 'completed',
        'completed_stages': list(completed_stages)
    }


# -----------------------------------------------------------------------------
# Multi-label wave
# -----------------------------------------------------------------------------
def run_multi_wave(args, structured_feature_info, locked_multi_select_row, locked_multi_manifest, checkpoint_fn=None):
    screen_rows = []
    select_rows = []
    holdout_rows = []
    label_df = empty_multi_label_df()
    overlap_metrics = []
    completed_stages = []

    raw_df, input_path = load_frame(MULTI_INPUT_STEM, input_path=args.multi_input_path)
    sidecar_df, text_sidecar_path = load_frame(SIDECAR_STEM, input_path=args.text_sidecar_path)
    case_df = prep_multi_label_cases(raw_df, structured_feature_info['feature_cols'])
    case_df = merge_text_sidecar(case_df, sidecar_df)
    split_parts = split_multi_label_cases_by_mode(case_df, split_mode=FEATURE_WAVE1_SPLIT_MODE)

    train_core_df = split_parts['train_core']
    screen_df = split_parts['screen_2024']
    dev_screen_df = split_parts['dev_2020_2024']
    select_df = split_parts['select_2025']
    dev_select_df = split_parts['dev_2020_2025']
    holdout_df = split_parts['holdout_2026']

    log_line(
        f'[multi] Split rows | train_core={len(train_core_df):,} screen_2024={len(screen_df):,} '
        f'select_2025={len(select_df):,} holdout_2026={len(holdout_df):,} '
        f'final_linear_model={args.final_linear_model}'
    )

    def emit_checkpoint(checkpoint_stage, selected_family=None, select_metrics=None, select_gate_pass=None, promotion_status='running'):
        if checkpoint_fn is None:
            return
        checkpoint_fn(
            checkpoint_stage,
            {
                'input_path': str(input_path),
                'text_sidecar_path': str(text_sidecar_path),
                'split_df': split_parts['split_df'],
                'screen_df': pd.DataFrame(screen_rows),
                'select_df': pd.DataFrame(select_rows),
                'holdout_df': pd.DataFrame(holdout_rows) if holdout_rows else empty_multi_holdout_df(),
                'label_df': label_df,
                'selected_family': selected_family,
                'final_linear_model': args.final_linear_model,
                'screen_fusion_weight': late_screen['selected_text_weight'] if 'late_screen' in locals() else None,
                'select_metrics': select_metrics,
                'select_gate_pass': select_gate_pass,
                'promotion_status': promotion_status,
                'overlap_metrics': overlap_metrics,
                'checkpoint_stage': checkpoint_stage,
                'completed_stages': list(completed_stages)
            }
        )

    structured_screen = fit_multi_structured_family(
        train_core_df,
        screen_df,
        structured_feature_info,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed
    )
    structured_screen_row = build_multi_row(
        'multi_label',
        STRUCTURED_FAMILY,
        input_path,
        text_sidecar_path,
        'screen_2024',
        'screen_2024',
        structured_screen['y_eval'],
        structured_screen['valid_pred'],
        structured_screen['valid_proba'],
        threshold=structured_screen['selected_threshold'],
        fit_seconds=structured_screen['fit_seconds'],
        selected_iteration=structured_screen['selected_iteration']
    )
    structured_screen_row['actual_task_type'] = structured_screen['actual_task_type']
    screen_rows.append(structured_screen_row)
    log_multi_family('screen_2024', STRUCTURED_FAMILY, structured_screen_row)

    text_only_screen = fit_multi_text_family(
        train_core_df,
        screen_df,
        structured_feature_info,
        TEXT_ONLY_FAMILY,
        final_model=False
    )
    text_only_screen_row = build_multi_row(
        'multi_label',
        TEXT_ONLY_FAMILY,
        input_path,
        text_sidecar_path,
        'screen_2024',
        'screen_2024',
        text_only_screen['y_eval'],
        text_only_screen['pred'],
        text_only_screen['eval_proba'],
        threshold=text_only_screen['threshold_choice']['threshold'],
        fit_seconds=text_only_screen['fit_seconds']
    )
    screen_rows.append(text_only_screen_row)
    log_multi_family('screen_2024', TEXT_ONLY_FAMILY, text_only_screen_row)

    if args.skip_text_plus:
        log_line('[multi] screen_2024 text_plus_structured_linear skipped by flag')
    else:
        text_plus_screen = fit_multi_text_family(
            train_core_df,
            screen_df,
            structured_feature_info,
            TEXT_PLUS_STRUCTURED_FAMILY,
            final_model=False
        )
        text_plus_screen_row = build_multi_row(
            'multi_label',
            TEXT_PLUS_STRUCTURED_FAMILY,
            input_path,
            text_sidecar_path,
            'screen_2024',
            'screen_2024',
            text_plus_screen['y_eval'],
            text_plus_screen['pred'],
            text_plus_screen['eval_proba'],
            threshold=text_plus_screen['threshold_choice']['threshold'],
            fit_seconds=text_plus_screen['fit_seconds']
        )
        screen_rows.append(text_plus_screen_row)
        log_multi_family('screen_2024', TEXT_PLUS_STRUCTURED_FAMILY, text_plus_screen_row)

    late_screen = select_multi_fusion_weight(
        structured_screen['y_eval'],
        text_only_screen['eval_proba'],
        structured_screen['valid_proba']
    )
    late_screen_row = build_multi_row(
        'multi_label',
        LATE_FUSION_FAMILY,
        input_path,
        text_sidecar_path,
        'screen_2024',
        'screen_2024',
        structured_screen['y_eval'],
        late_screen['pred'],
        late_screen['proba'],
        threshold=late_screen['selected_threshold'],
        fit_seconds=round(float(text_only_screen['fit_seconds']) + float(structured_screen['fit_seconds']), 2),
        selected_iteration=structured_screen['selected_iteration'],
        selected_text_weight=late_screen['selected_text_weight']
    )
    screen_rows.append(late_screen_row)
    log_multi_family('screen_2024', LATE_FUSION_FAMILY, late_screen_row)
    completed_stages.append('screen_2024')
    emit_checkpoint('screen_2024_complete')

    structured_select = fit_multi_structured_family(
        dev_screen_df,
        select_df,
        structured_feature_info,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed
    )
    structured_select_row = build_multi_row(
        'multi_label',
        STRUCTURED_FAMILY,
        input_path,
        text_sidecar_path,
        'select_2025',
        'select_2025',
        structured_select['y_eval'],
        structured_select['valid_pred'],
        structured_select['valid_proba'],
        threshold=structured_select['selected_threshold'],
        fit_seconds=structured_select['fit_seconds'],
        selected_iteration=structured_select['selected_iteration']
    )
    structured_select_row['actual_task_type'] = structured_select['actual_task_type']
    select_rows.append(structured_select_row)
    log_multi_family('select_2025', STRUCTURED_FAMILY, structured_select_row)

    text_only_select = fit_multi_text_family(
        dev_screen_df,
        select_df,
        structured_feature_info,
        TEXT_ONLY_FAMILY,
        final_model=False
    )
    text_only_select_row = build_multi_row(
        'multi_label',
        TEXT_ONLY_FAMILY,
        input_path,
        text_sidecar_path,
        'select_2025',
        'select_2025',
        text_only_select['y_eval'],
        text_only_select['pred'],
        text_only_select['eval_proba'],
        threshold=text_only_select['threshold_choice']['threshold'],
        fit_seconds=text_only_select['fit_seconds']
    )
    select_rows.append(text_only_select_row)
    log_multi_family('select_2025', TEXT_ONLY_FAMILY, text_only_select_row)

    if args.skip_text_plus:
        log_line('[multi] select_2025 text_plus_structured_linear skipped by flag')
    else:
        text_plus_select = fit_multi_text_family(
            dev_screen_df,
            select_df,
            structured_feature_info,
            TEXT_PLUS_STRUCTURED_FAMILY,
            final_model=False
        )
        text_plus_select_row = build_multi_row(
            'multi_label',
            TEXT_PLUS_STRUCTURED_FAMILY,
            input_path,
            text_sidecar_path,
            'select_2025',
            'select_2025',
            text_plus_select['y_eval'],
            text_plus_select['pred'],
            text_plus_select['eval_proba'],
            threshold=text_plus_select['threshold_choice']['threshold'],
            fit_seconds=text_plus_select['fit_seconds']
        )
        select_rows.append(text_plus_select_row)
        log_multi_family('select_2025', TEXT_PLUS_STRUCTURED_FAMILY, text_plus_select_row)

    late_select = apply_multi_fusion_weight(
        structured_select['y_eval'],
        text_only_select['eval_proba'],
        structured_select['valid_proba'],
        text_weight=late_screen['selected_text_weight']
    )
    late_select_row = build_multi_row(
        'multi_label',
        LATE_FUSION_FAMILY,
        input_path,
        text_sidecar_path,
        'select_2025',
        'select_2025',
        structured_select['y_eval'],
        late_select['pred'],
        late_select['proba'],
        threshold=late_select['selected_threshold'],
        fit_seconds=round(float(text_only_select['fit_seconds']) + float(structured_select['fit_seconds']), 2),
        selected_iteration=structured_select['selected_iteration'],
        selected_text_weight=late_screen['selected_text_weight']
    )
    select_rows.append(late_select_row)
    log_multi_family('select_2025', LATE_FUSION_FAMILY, late_select_row)

    select_df_all = pd.DataFrame(select_rows)
    current_best_row = select_best_row(
        select_df_all,
        ['macro_f1', 'micro_f1', 'recall_at_3', 'precision_at_3']
    )
    selected_family = current_best_row['family_name']
    select_improvement = float(current_best_row['macro_f1'] - locked_multi_select_row['macro_f1'])
    select_gate_pass = select_improvement >= MULTI_PROMOTE_SELECT_DELTA
    promotion_status = 'rejected_select'
    log_line(
        f'[multi] select_2025 best={selected_family} '
        f'macro_f1={float(current_best_row["macro_f1"]):.4f} '
        f'delta_vs_locked={select_improvement:+.4f} '
        f'gate_pass={str(select_gate_pass).lower()}'
    )
    completed_stages.append('select_2025')
    emit_checkpoint(
        'select_2025_complete',
        selected_family=selected_family,
        select_metrics=current_best_row,
        select_gate_pass=select_gate_pass,
        promotion_status='running'
    )

    if select_gate_pass:
        log_line(f'[multi] holdout_2026 start family={selected_family}')
        emit_checkpoint(
            'holdout_2026_started',
            selected_family=selected_family,
            select_metrics=current_best_row,
            select_gate_pass=select_gate_pass,
            promotion_status='running'
        )
        if selected_family == STRUCTURED_FAMILY:
            log_line(
                f'[multi] holdout_2026 fitting structured carry-forward '
                f'iteration={int(structured_select["selected_iteration"])} '
                f'threshold={float(structured_select["selected_threshold"]):.3f}'
            )
            holdout_result = fit_multi_structured_holdout(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                structured_select,
                devices=args.devices,
                random_seed=args.random_seed
            )
            log_line(
                f'[multi] holdout_2026 structured fit complete '
                f'fit_seconds={float(holdout_result["fit_seconds"]):.2f}'
            )
            y_holdout = holdout_result['y_holdout']
            holdout_pred = holdout_result['holdout_pred']
            holdout_proba = holdout_result['holdout_proba']
            holdout_row = build_multi_row(
                'multi_label',
                STRUCTURED_FAMILY,
                input_path,
                text_sidecar_path,
                'final_holdout',
                'holdout_2026',
                y_holdout,
                holdout_pred,
                holdout_proba,
                threshold=holdout_result['selected_threshold'],
                fit_seconds=holdout_result['fit_seconds'],
                selected_iteration=holdout_result['selected_iteration']
            )
            mlb = holdout_result['mlb']
        elif selected_family == TEXT_ONLY_FAMILY:
            log_line(
                f'[multi] holdout_2026 fitting final text-only linear model '
                f'dev_rows={len(dev_select_df):,} '
                f'final_model={args.final_linear_model}'
            )
            text_holdout = fit_multi_text_family(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                TEXT_ONLY_FAMILY,
                final_model=True,
                final_model_kind=args.final_linear_model
            )
            log_line(
                f'[multi] holdout_2026 text-only fit complete '
                f'fit_seconds={float(text_holdout["fit_seconds"]):.2f}'
            )
            y_holdout = text_holdout['y_eval']
            holdout_proba = text_holdout['eval_proba']
            holdout_pred = apply_multilabel_threshold(
                holdout_proba,
                text_only_select['threshold_choice']['threshold'],
                min_positive_labels=MIN_POSITIVE_LABELS
            )
            holdout_row = build_multi_row(
                'multi_label',
                TEXT_ONLY_FAMILY,
                input_path,
                text_sidecar_path,
                'final_holdout',
                'holdout_2026',
                y_holdout,
                holdout_pred,
                holdout_proba,
                threshold=text_only_select['threshold_choice']['threshold'],
                fit_seconds=text_holdout['fit_seconds']
            )
            mlb = text_holdout['mlb']
        elif selected_family == TEXT_PLUS_STRUCTURED_FAMILY:
            log_line(
                f'[multi] holdout_2026 fitting final text+structured linear model '
                f'dev_rows={len(dev_select_df):,} '
                f'final_model={args.final_linear_model}'
            )
            text_holdout = fit_multi_text_family(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                TEXT_PLUS_STRUCTURED_FAMILY,
                final_model=True,
                final_model_kind=args.final_linear_model
            )
            log_line(
                f'[multi] holdout_2026 text+structured fit complete '
                f'fit_seconds={float(text_holdout["fit_seconds"]):.2f}'
            )
            y_holdout = text_holdout['y_eval']
            holdout_proba = text_holdout['eval_proba']
            holdout_pred = apply_multilabel_threshold(
                holdout_proba,
                text_plus_select['threshold_choice']['threshold'],
                min_positive_labels=MIN_POSITIVE_LABELS
            )
            holdout_row = build_multi_row(
                'multi_label',
                TEXT_PLUS_STRUCTURED_FAMILY,
                input_path,
                text_sidecar_path,
                'final_holdout',
                'holdout_2026',
                y_holdout,
                holdout_pred,
                holdout_proba,
                threshold=text_plus_select['threshold_choice']['threshold'],
                fit_seconds=text_holdout['fit_seconds']
            )
            mlb = text_holdout['mlb']
        else:
            log_line(
                f'[multi] holdout_2026 fitting late-fusion structured branch '
                f'iteration={int(structured_select["selected_iteration"])} '
                f'text_weight={float(late_screen["selected_text_weight"]):.2f}'
            )
            structured_holdout = fit_multi_structured_holdout(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                structured_select,
                devices=args.devices,
                random_seed=args.random_seed
            )
            log_line(
                f'[multi] holdout_2026 structured branch complete '
                f'fit_seconds={float(structured_holdout["fit_seconds"]):.2f}'
            )
            log_line(
                f'[multi] holdout_2026 fitting late-fusion text branch '
                f'dev_rows={len(dev_select_df):,} '
                f'final_model={args.final_linear_model}'
            )
            text_holdout = fit_multi_text_family(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                TEXT_ONLY_FAMILY,
                final_model=True,
                final_model_kind=args.final_linear_model
            )
            log_line(
                f'[multi] holdout_2026 text branch complete '
                f'fit_seconds={float(text_holdout["fit_seconds"]):.2f}'
            )
            y_holdout = text_holdout['y_eval']
            holdout_proba = (
                float(late_screen['selected_text_weight']) * text_holdout['eval_proba']
                + (1.0 - float(late_screen['selected_text_weight'])) * structured_holdout['holdout_proba']
            )
            holdout_pred = apply_multilabel_threshold(
                holdout_proba,
                late_select['selected_threshold'],
                min_positive_labels=MIN_POSITIVE_LABELS
            )
            holdout_row = build_multi_row(
                'multi_label',
                LATE_FUSION_FAMILY,
                input_path,
                text_sidecar_path,
                'final_holdout',
                'holdout_2026',
                y_holdout,
                holdout_pred,
                holdout_proba,
                threshold=late_select['selected_threshold'],
                fit_seconds=round(float(text_holdout['fit_seconds']) + float(structured_holdout['fit_seconds']), 2),
                selected_iteration=structured_holdout['selected_iteration'],
                selected_text_weight=late_screen['selected_text_weight']
            )
            mlb = text_holdout['mlb']

        holdout_rows.append(holdout_row)
        log_multi_family('holdout_2026', selected_family, holdout_row)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_holdout,
            holdout_pred,
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

        if selected_family in TEXT_FAMILIES:
            overlap_mask = build_overlap_mask(
                dev_select_df['cdescr_model_text'],
                holdout_df['cdescr_model_text']
            )
            slice_rows = build_multi_overlap_rows(
                holdout_row,
                y_holdout,
                holdout_pred,
                holdout_proba,
                overlap_mask
            )
            holdout_rows.extend(slice_rows)
            overlap_metrics = slice_rows

        locked_holdout = locked_multi_manifest['official_holdout_metrics']
        macro_gain = float(holdout_row['macro_f1'] - locked_holdout['macro_f1'])
        micro_gain = float(holdout_row['micro_f1'] - locked_holdout['micro_f1'])
        promotion_status = 'promoted' if (
            (macro_gain >= MULTI_PROMOTE_HOLDOUT_MACRO_DELTA or micro_gain >= MULTI_PROMOTE_HOLDOUT_MICRO_DELTA)
            and holdout_row['recall_at_3'] >= locked_holdout['recall_at_3']
            and holdout_row['label_coverage'] >= MULTI_LABEL_COVERAGE_FLOOR
        ) else 'rejected_holdout'
        log_line(
            f'[multi] holdout_2026 promotion_status={promotion_status} '
            f'macro_gain={macro_gain:+.4f} '
            f'micro_gain={micro_gain:+.4f} '
            f'recall3_ok={str(holdout_row["recall_at_3"] >= locked_holdout["recall_at_3"]).lower()} '
            f'label_coverage={float(holdout_row["label_coverage"]):.4f}'
        )
        completed_stages.append('holdout_2026')
        emit_checkpoint(
            'holdout_2026_complete',
            selected_family=selected_family,
            select_metrics=current_best_row,
            select_gate_pass=select_gate_pass,
            promotion_status=promotion_status
        )
    else:
        log_line(
            f'[multi] select_2025 gate rejected; skipping holdout '
            f'family={selected_family}'
        )
        emit_checkpoint(
            'select_gate_rejected',
            selected_family=selected_family,
            select_metrics=current_best_row,
            select_gate_pass=select_gate_pass,
            promotion_status=promotion_status
        )

    return {
        'input_path': str(input_path),
        'text_sidecar_path': str(text_sidecar_path),
        'split_df': split_parts['split_df'],
        'screen_df': pd.DataFrame(screen_rows),
        'select_df': select_df_all,
        'holdout_df': pd.DataFrame(holdout_rows) if holdout_rows else empty_multi_holdout_df(),
        'label_df': label_df,
        'selected_family': selected_family,
        'final_linear_model': args.final_linear_model,
        'screen_fusion_weight': late_screen['selected_text_weight'],
        'select_metrics': current_best_row,
        'select_gate_pass': select_gate_pass,
        'promotion_status': promotion_status,
        'overlap_metrics': overlap_metrics,
        'checkpoint_stage': 'completed',
        'completed_stages': list(completed_stages)
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Run component text wave 2 with complaint narratives plus the carried-forward structured family'
    )
    parser.add_argument(
        '--task-type',
        choices=['CPU', 'GPU', 'cpu', 'gpu'],
        default='CPU',
        help='CatBoost processing target for the structured carry-forward family'
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
    parser.add_argument('--single-input-path', default=None)
    parser.add_argument('--multi-input-path', default=None)
    parser.add_argument('--text-sidecar-path', default=None)
    parser.add_argument('--skip-single', action='store_true')
    parser.add_argument('--skip-multi', action='store_true')
    parser.add_argument(
        '--skip-text-plus',
        action='store_true',
        help='Skip the early-fusion text_plus_structured_linear family'
    )
    parser.add_argument(
        '--final-linear-model',
        choices=FINAL_LINEAR_MODEL_CHOICES,
        default=FINAL_LINEAR_MODEL_DEFAULT,
        help='Final refit estimator for promoted text families'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    if args.skip_single and args.skip_multi:
        raise ValueError('Nothing to do: both single and multi tasks were skipped')

    structured_feature_info = feature_manifest(STRUCTURED_FEATURE_SET)
    locked_single_select_row = read_locked_single_select_baseline()
    locked_multi_select_row = read_locked_multi_select_baseline()
    locked_single_manifest = load_json(LOCKED_SINGLE_MANIFEST)
    locked_multi_manifest = load_json(LOCKED_MULTI_MANIFEST)
    locked_single_selection = load_json(LOCKED_SINGLE_SELECTION)
    locked_single_ece = read_locked_single_ece()

    manifest = {
        'artifact_role': FEATUREWAVE_TASK,
        'feature_wave': 2,
        'split_mode': FEATURE_WAVE1_SPLIT_MODE,
        'public_benchmark_locked': True,
        'run_status': 'running',
        'structured_companion_feature_set': STRUCTURED_FEATURE_SET,
        'final_linear_model': args.final_linear_model,
        'text_config': TEXT_CONFIG,
        'runtime': runtime_manifest(),
        'code_version': {
            'git_head': get_git_head(),
            'git_dirty': get_git_dirty_flag()
        },
        'last_checkpoint': None,
        'tasks': {}
    }
    write_json(manifest, OUTPUTS_DIR / GLOBAL_MANIFEST_NAME)

    def checkpoint_single(stage_name, result):
        write_single_outputs(result)
        manifest['tasks']['single_label'] = build_single_manifest_entry(
            result,
            locked_single_select_row,
            locked_single_manifest['official_holdout_metrics'],
            locked_single_ece
        )
        manifest['last_checkpoint'] = {
            'task': 'single_label',
            'stage': stage_name
        }
        write_json(manifest, OUTPUTS_DIR / GLOBAL_MANIFEST_NAME)

    def checkpoint_multi(stage_name, result):
        write_multi_outputs(result)
        manifest['tasks']['multi_label'] = build_multi_manifest_entry(
            result,
            locked_multi_select_row,
            locked_multi_manifest['official_holdout_metrics']
        )
        manifest['last_checkpoint'] = {
            'task': 'multi_label',
            'stage': stage_name
        }
        write_json(manifest, OUTPUTS_DIR / GLOBAL_MANIFEST_NAME)

    try:
        if not args.skip_single:
            log_line('[run] Single-label text wave 2')
            single_result = run_single_wave(
                args,
                structured_feature_info,
                locked_single_select_row,
                locked_single_manifest,
                locked_single_ece,
                locked_single_selection,
                checkpoint_fn=checkpoint_single
            )
            write_single_outputs(single_result)
            manifest['tasks']['single_label'] = build_single_manifest_entry(
                single_result,
                locked_single_select_row,
                locked_single_manifest['official_holdout_metrics'],
                locked_single_ece
            )

        if not args.skip_multi:
            log_line('[run] Multi-label text wave 2')
            multi_result = run_multi_wave(
                args,
                structured_feature_info,
                locked_multi_select_row,
                locked_multi_manifest,
                checkpoint_fn=checkpoint_multi
            )
            write_multi_outputs(multi_result)
            manifest['tasks']['multi_label'] = build_multi_manifest_entry(
                multi_result,
                locked_multi_select_row,
                locked_multi_manifest['official_holdout_metrics']
            )
    except Exception as exc:
        manifest['run_status'] = 'failed'
        manifest['error'] = str(exc)
        write_json(manifest, OUTPUTS_DIR / GLOBAL_MANIFEST_NAME)
        raise

    manifest['run_status'] = 'completed'
    write_json(manifest, OUTPUTS_DIR / GLOBAL_MANIFEST_NAME)
    print(f'[write] {OUTPUTS_DIR / GLOBAL_MANIFEST_NAME}')
    print('[done] Component text wave 2 finished')
    return 0


if __name__ == '__main__':
    sys.exit(main())
