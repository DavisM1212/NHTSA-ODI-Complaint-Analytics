import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import settings
from src.config.contracts import (
    BENCHMARK_SPLIT_MODE,
    SEVERITY_CASES_STEM,
    SEVERITY_URGENCY_OFFICIAL_CALIBRATION,
    SEVERITY_URGENCY_OFFICIAL_MANIFEST,
    SEVERITY_URGENCY_OFFICIAL_METRICS,
    SEVERITY_URGENCY_OFFICIAL_REVIEW_BUDGETS,
    get_split_policy,
)
from src.config.paths import OUTPUTS_DIR, ensure_project_directories
from src.data.io_utils import load_frame, write_json
from src.modeling.common.helpers import log_line

# -----------------------------------------------------------------------------
# Locked severity urgency contract
# -----------------------------------------------------------------------------
ID_COL = 'odino'
DATE_COL = 'ldate'
TEXT_SOURCE_COL = 'cdescr'
TEXT_MODEL_COL = 'cdescr_model_text'
TARGET_COL = 'severity_primary_flag'

BASELINE_NAME = 'dummy_prior'
RAW_NAME = 'late_fusion_raw'
SIGMOID_NAME = 'late_fusion_sigmoid'
ISOTONIC_NAME = 'late_fusion_isotonic'
NOTEBOOK_REFERENCE_NAME = 'hybrid_late_fusion_tuned_alpha_1em05_word_30000_char_20000_w_0_81_sigmoid_calibrated'

OFFICIAL_SGD_ALPHA = 1e-5
OFFICIAL_TEXT_WEIGHT = 0.81
TEXT_MIN_DF = 5
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (3, 5)
WORD_MAX_FEATURES = 30000
CHAR_MAX_FEATURES = 20000
CALIBRATION_RECALL_TOLERANCE = 0.002
CALIBRATION_BRIER_TIE = 0.002
RELIABILITY_BINS = 10

REVIEW_BUDGETS = [0.01, 0.02, 0.05, 0.10]
REVIEW_BUDGET_LABELS = {
    0.01: 'top_1pct',
    0.02: 'top_2pct',
    0.05: 'top_5pct',
    0.10: 'top_10pct'
}

STRUCTURED_CAT_FEATURES = [
    'mfr_name',
    'maketxt',
    'modeltxt',
    'state',
    'cmpl_type',
    'drive_train',
    'fuel_type',
    'police_rpt_yn',
    'repaired_yn',
    'orig_owner_yn'
]

STRUCTURED_NUM_FEATURES = [
    'yeartxt',
    'miles',
    'veh_speed',
    'lag_days_safe',
    'miles_missing_flag',
    'veh_speed_missing_flag',
    'faildate_trusted_flag',
    'faildate_untrusted_flag',
    'component_count',
    'row_count',
    'complaint_year',
    'complaint_month'
]

BOOLEAN_FEATURE_COLS = [
    TARGET_COL,
    'severity_broad_flag',
    'miles_missing_flag',
    'veh_speed_missing_flag',
    'faildate_trusted_flag',
    'faildate_untrusted_flag'
]

PRESERVE_STOP_WORDS = {
    'no',
    'not',
    'never',
    'none',
    'without',
    'while',
    'after',
    'before',
    'during',
    'against',
    'off',
    'on',
    'over',
    'under',
    'down',
    'up'
}

ERROR_DRIVEN_TEXT_PATTERNS = [
    (r'(?i)^\s*(?:the contact|consumer|owner|driver|complainant)\s+(?:states?|stated|owns|owned|leased)\b[\s,:;-]*', ''),
    (r'(?i)(?:https?://\S+|www\.\S+|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})', ' contact_token '),
    (r'(?i)(?:\+?1[-.\s]*)?(?:\(?\d{3}\)?[-.\s]*)\d{3}[-.\s]*\d{4}', ' phone_token '),
    (r'(?i)\b[a-hj-npr-z0-9]{17}\b', ' vin_token '),
    (r'(?i)\b(?:case|claim|ticket|reference|tracking|repair order|work order|invoice)\s*(?:number|no|#)?\s*[:#-]?\s*[a-z0-9-]{5,}\b', ' case_id_token '),
    (r'(?i)\b(?=[a-z0-9-]{8,}\b)(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)[a-z0-9-]+\b', ' id_token ')
]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def build_custom_stop_words():
    return sorted(set(ENGLISH_STOP_WORDS) - PRESERVE_STOP_WORDS)


def require_columns(df, required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        missing_text = ', '.join(missing)
        raise ValueError(f'Severity cases are missing required columns: {missing_text}')


def coerce_bool_like(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    truthy = {'1', 'true', 't', 'yes', 'y'}
    falsy = {'0', 'false', 'f', 'no', 'n', ''}
    normalized = series.astype('string').fillna('').str.strip().str.lower()
    bool_series = normalized.isin(truthy)
    unknown_mask = ~normalized.isin(truthy | falsy)
    if unknown_mask.any():
        bad_values = sorted(normalized.loc[unknown_mask].unique().tolist())[:5]
        raise ValueError(f'Unexpected boolean-like values: {bad_values}')
    return bool_series.astype(bool)


def safe_average_precision(y_true, score):
    if np.unique(y_true).size < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, score))


def safe_roc_auc(y_true, score):
    if np.unique(y_true).size < 2:
        return np.nan
    return float(roc_auc_score(y_true, score))


def get_budget_label(fraction):
    if fraction in REVIEW_BUDGET_LABELS:
        return REVIEW_BUDGET_LABELS[fraction]
    pct = fraction * 100
    if np.isclose(pct, round(pct)):
        return f'top_{int(round(pct))}pct'
    pct_text = f'{pct:.2f}'.rstrip('0').rstrip('.')
    return f'top_{pct_text}pct'


def sigmoid_score(raw_score):
    raw_score = np.asarray(raw_score, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-raw_score))


def get_model_margin(model, X):
    if hasattr(model, 'decision_function'):
        return np.asarray(model.decision_function(X), dtype=np.float64)
    if hasattr(model, 'predict_proba'):
        proba = np.clip(model.predict_proba(X)[:, 1], 1e-8, 1 - 1e-8)
        return np.log(proba / (1 - proba))
    raise AttributeError('Model must define decision_function or predict_proba')


def build_linear_sgd_model(alpha=OFFICIAL_SGD_ALPHA, random_seed=settings.RANDOM_SEED):
    return SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=alpha,
        max_iter=100,
        tol=1e-3,
        class_weight='balanced',
        random_state=random_seed,
        early_stopping=True,
        validation_fraction=0.05,
        n_iter_no_change=5
    )


def build_word_vectorizer(stop_words):
    return TfidfVectorizer(
        analyzer='word',
        ngram_range=WORD_NGRAM_RANGE,
        min_df=TEXT_MIN_DF,
        max_df=0.995,
        max_features=WORD_MAX_FEATURES,
        sublinear_tf=True,
        stop_words=stop_words,
        strip_accents='unicode',
        lowercase=True,
        dtype=np.float32
    )


def build_char_vectorizer():
    return TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=CHAR_NGRAM_RANGE,
        min_df=TEXT_MIN_DF,
        max_features=CHAR_MAX_FEATURES,
        sublinear_tf=True,
        strip_accents='unicode',
        lowercase=True,
        dtype=np.float32
    )


def build_text_series(source_df, clean_mode='light', domain_cleanup=False, error_cleanup=True):
    text = source_df[TEXT_MODEL_COL].fillna('').astype(str)
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()

    if clean_mode not in {'base', 'light'}:
        raise ValueError(f"clean_mode must be 'base' or 'light', got {clean_mode}")

    if clean_mode == 'light':
        text = text.str.replace(r'(?i)\[\s*x+\s*\]', ' ', regex=True)
        text = text.str.replace(
            r'(?i)information redacted pursuant to the freedom of information act \(foia\) 5 u\.s\.c\. 552\(b\)\(6\) and 49 c\.f\.r\. 512\.8',
            ' ',
            regex=True
        )
        text = text.str.replace(
            r'(?i)^\s*the contact (owns|owned|stated|leased)\b[\s,:;-]*',
            '',
            regex=True
        )

    if error_cleanup:
        for pattern, replacement in ERROR_DRIVEN_TEXT_PATTERNS:
            text = text.str.replace(pattern, replacement, regex=True)

    if domain_cleanup:
        raise ValueError('domain_cleanup is not part of the locked official severity path')

    return text.str.replace(r'\s+', ' ', regex=True).str.strip()


def prepare_structured_frame(source_df, cat_features, num_features):
    work = source_df[cat_features + num_features].copy()
    for col in cat_features:
        work[col] = work[col].astype('string').fillna('missing').astype(str)
    for col in num_features:
        work[col] = pd.to_numeric(work[col], errors='coerce').astype(float)
    return work


def build_structured_matrices(train_df, valid_df, holdout_df):
    preprocess = ColumnTransformer(
        transformers=[
            (
                'cat',
                Pipeline(
                    [
                        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=50))
                    ]
                ),
                STRUCTURED_CAT_FEATURES
            ),
            (
                'num',
                Pipeline(
                    [
                        ('impute', SimpleImputer(strategy='median')),
                        ('scale', StandardScaler())
                    ]
                ),
                STRUCTURED_NUM_FEATURES
            )
        ],
        remainder='drop'
    )

    X_train = preprocess.fit_transform(prepare_structured_frame(train_df, STRUCTURED_CAT_FEATURES, STRUCTURED_NUM_FEATURES))
    X_valid = preprocess.transform(prepare_structured_frame(valid_df, STRUCTURED_CAT_FEATURES, STRUCTURED_NUM_FEATURES))
    X_holdout = preprocess.transform(prepare_structured_frame(holdout_df, STRUCTURED_CAT_FEATURES, STRUCTURED_NUM_FEATURES))
    return X_train, X_valid, X_holdout


def build_text_matrices(train_text, valid_text, holdout_text):
    stop_words = build_custom_stop_words()
    word_vectorizer = build_word_vectorizer(stop_words)
    char_vectorizer = build_char_vectorizer()

    X_train_word = word_vectorizer.fit_transform(train_text)
    X_valid_word = word_vectorizer.transform(valid_text)
    X_holdout_word = word_vectorizer.transform(holdout_text)

    X_train_char = char_vectorizer.fit_transform(train_text)
    X_valid_char = char_vectorizer.transform(valid_text)
    X_holdout_char = char_vectorizer.transform(holdout_text)

    X_train = sparse.hstack([X_train_word, X_train_char], format='csr')
    X_valid = sparse.hstack([X_valid_word, X_valid_char], format='csr')
    X_holdout = sparse.hstack([X_holdout_word, X_holdout_char], format='csr')
    return X_train, X_valid, X_holdout


def fit_branch_model(X_train, X_valid, X_holdout, y_train, random_seed):
    model = build_linear_sgd_model(alpha=OFFICIAL_SGD_ALPHA, random_seed=random_seed)
    model.fit(X_train, y_train)

    valid_margin = get_model_margin(model, X_valid)
    holdout_margin = get_model_margin(model, X_holdout)
    return {
        'model': model,
        'valid_margin': valid_margin,
        'holdout_margin': holdout_margin,
        'valid_score': sigmoid_score(valid_margin),
        'holdout_score': sigmoid_score(holdout_margin)
    }


def fit_dummy_prior(train_df, valid_df, holdout_df, y_train, random_seed):
    model = DummyClassifier(strategy='prior', random_state=random_seed)
    X_train = np.zeros((len(train_df), 1), dtype=np.float64)
    X_valid = np.zeros((len(valid_df), 1), dtype=np.float64)
    X_holdout = np.zeros((len(holdout_df), 1), dtype=np.float64)
    model.fit(X_train, y_train)
    return {
        'model': model,
        'valid_score': model.predict_proba(X_valid)[:, 1],
        'holdout_score': model.predict_proba(X_holdout)[:, 1]
    }


def build_review_budget_row(y_true, score, fraction):
    y_true = np.asarray(y_true).astype(bool)
    score = np.asarray(score, dtype=np.float64)
    n_rows = len(score)
    n_flagged = max(1, int(np.ceil(n_rows * fraction)))
    top_idx = np.argsort(-score, kind='mergesort')[:n_flagged]

    severe_captured = int(y_true[top_idx].sum())
    total_positive = int(y_true.sum())
    base_rate = float(y_true.mean()) if n_rows else np.nan
    precision = severe_captured / n_flagged if n_flagged else np.nan
    recall = severe_captured / total_positive if total_positive else np.nan
    cutoff = float(score[top_idx].min()) if n_flagged else np.nan
    lift = precision / base_rate if base_rate else np.nan
    return {
        'budget_fraction': fraction,
        'budget_label': get_budget_label(fraction),
        'flagged_rows': n_flagged,
        'flagged_share': n_flagged / n_rows if n_rows else np.nan,
        'severe_cases_captured': severe_captured,
        'recall_within_flagged_set': recall,
        'precision_within_flagged_set': precision,
        'lift_vs_base_rate': lift,
        'score_cutoff': cutoff
    }


def build_budget_rows(model_name, split_name, y_true, score, is_baseline=False, is_official=False):
    rows = []
    for fraction in REVIEW_BUDGETS:
        row = build_review_budget_row(y_true, score, fraction)
        row['model'] = model_name
        row['split'] = split_name
        row['is_baseline'] = bool(is_baseline)
        row['is_official'] = bool(is_official)
        rows.append(row)
    return rows


def build_score_row(model_name, split_name, y_true, score, calibration_method, is_baseline=False, is_official=False):
    y_true = np.asarray(y_true).astype(bool)
    score = np.asarray(score, dtype=np.float64)
    pred = score >= 0.5

    row = {
        'model': model_name,
        'split': split_name,
        'rows': len(y_true),
        'positive_rate': float(np.mean(y_true)),
        'pr_auc': safe_average_precision(y_true, score),
        'roc_auc': safe_roc_auc(y_true, score),
        'f1': float(f1_score(y_true, pred, zero_division=0)),
        'precision': float(precision_score(y_true, pred, zero_division=0)),
        'recall': float(recall_score(y_true, pred, zero_division=0)),
        'brier_score': float(brier_score_loss(y_true, score)),
        'calibration_method': calibration_method,
        'is_baseline': bool(is_baseline),
        'is_official': bool(is_official)
    }

    for fraction, label in REVIEW_BUDGET_LABELS.items():
        budget_row = build_review_budget_row(y_true, score, fraction)
        row[f'recall_{label}'] = float(budget_row['recall_within_flagged_set'])
        if fraction in {0.05, 0.10}:
            row[f'precision_{label}'] = float(budget_row['precision_within_flagged_set'])
            row[f'lift_{label}'] = float(budget_row['lift_vs_base_rate'])
    return row


def build_reliability_table(y_true, score, model_name, split_name, is_official=False):
    frame = pd.DataFrame({
        'y_true': np.asarray(y_true).astype(bool),
        'score': np.asarray(score, dtype=np.float64)
    })

    if frame.empty:
        return pd.DataFrame(
            columns=[
                'model',
                'split',
                'bin',
                'rows',
                'avg_score',
                'observed_rate',
                'score_gap',
                'is_official'
            ]
        )

    unique_scores = frame['score'].nunique(dropna=False)
    if unique_scores <= 1:
        frame['bin'] = 0
    else:
        q = min(RELIABILITY_BINS, int(unique_scores))
        try:
            frame['bin'] = pd.qcut(frame['score'], q=q, labels=False, duplicates='drop')
        except ValueError:
            frame['bin'] = 0

    reliability_df = frame.groupby('bin', observed=False).agg(
        rows=('score', 'size'),
        avg_score=('score', 'mean'),
        observed_rate=('y_true', 'mean')
    ).reset_index()
    reliability_df['score_gap'] = reliability_df['avg_score'] - reliability_df['observed_rate']
    reliability_df.insert(0, 'split', split_name)
    reliability_df.insert(0, 'model', model_name)
    reliability_df['is_official'] = bool(is_official)
    return reliability_df


def pick_calibration_winner(raw_row, sigmoid_row, isotonic_row):
    recall_pool = []
    for candidate_row in [sigmoid_row, isotonic_row]:
        if abs(float(candidate_row['recall_top_5pct']) - float(raw_row['recall_top_5pct'])) <= CALIBRATION_RECALL_TOLERANCE:
            recall_pool.append(candidate_row)

    if not recall_pool:
        return RAW_NAME

    if len(recall_pool) == 2:
        brier_gap = abs(float(sigmoid_row['brier_score']) - float(isotonic_row['brier_score']))
        if brier_gap <= CALIBRATION_BRIER_TIE:
            preferred = sigmoid_row
        else:
            preferred = min(recall_pool, key=lambda row: float(row['brier_score']))
    else:
        preferred = recall_pool[0]

    if float(preferred['brier_score']) < float(raw_row['brier_score']):
        return preferred['model']
    return RAW_NAME


def prepare_severity_cases(raw_df):
    required_cols = [
        ID_COL,
        DATE_COL,
        TEXT_SOURCE_COL,
        TARGET_COL,
        *STRUCTURED_CAT_FEATURES,
        *[col for col in STRUCTURED_NUM_FEATURES if col not in {'complaint_year', 'complaint_month'}]
    ]
    require_columns(raw_df, required_cols)

    work = raw_df.copy()
    work[ID_COL] = work[ID_COL].astype('string').astype(str)
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors='coerce')
    if work[DATE_COL].isna().any():
        raise ValueError('Severity cases contain missing or invalid complaint dates')

    for col in [col for col in BOOLEAN_FEATURE_COLS if col in work.columns]:
        work[col] = coerce_bool_like(work[col])
    work[TEXT_MODEL_COL] = work[TEXT_SOURCE_COL].fillna('').astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    work['complaint_year'] = work[DATE_COL].dt.year.astype('int64')
    work['complaint_month'] = work[DATE_COL].dt.month.astype('int64')

    return work.sort_values([DATE_COL, ID_COL], kind='mergesort').reset_index(drop=True)


def split_severity_cases(case_df):
    split_policy = get_split_policy(BENCHMARK_SPLIT_MODE)
    train_end = split_policy['train_end']
    valid_end = split_policy['valid_end']

    train_df = case_df.loc[case_df[DATE_COL] <= train_end].copy()
    valid_df = case_df.loc[(case_df[DATE_COL] > train_end) & (case_df[DATE_COL] <= valid_end)].copy()
    holdout_df = case_df.loc[case_df[DATE_COL] > valid_end].copy()

    split_lookup = {
        split_policy['train_name']: train_df,
        split_policy['valid_name']: valid_df,
        split_policy['holdout_name']: holdout_df
    }

    for split_name, split_df in split_lookup.items():
        if split_df.empty:
            raise ValueError(f'{split_name} is empty under the benchmark split contract')
        if split_df[TARGET_COL].nunique(dropna=False) < 2:
            raise ValueError(f'{split_name} does not contain both target classes')

    summary_rows = []
    for split_name, split_df in split_lookup.items():
        summary_rows.append(
            {
                'split': split_name,
                'rows': int(len(split_df)),
                'positive_rate': float(split_df[TARGET_COL].mean())
            }
        )
    return split_lookup, pd.DataFrame(summary_rows), split_policy


def run_severity_pipeline(raw_df, input_path, output_dir=OUTPUTS_DIR, random_seed=settings.RANDOM_SEED, publish_status='official'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_total = perf_counter()
    case_df = prepare_severity_cases(raw_df)
    split_lookup, split_summary_df, split_policy = split_severity_cases(case_df)

    train_name = split_policy['train_name']
    valid_name = split_policy['valid_name']
    holdout_name = split_policy['holdout_name']

    train_df = split_lookup[train_name]
    valid_df = split_lookup[valid_name]
    holdout_df = split_lookup[holdout_name]
    y_train = train_df[TARGET_COL].to_numpy()
    y_valid = valid_df[TARGET_COL].to_numpy()
    y_holdout = holdout_df[TARGET_COL].to_numpy()

    log_line(
        f'[severity] rows | {train_name}={len(train_df):,} '
        f'{valid_name}={len(valid_df):,} {holdout_name}={len(holdout_df):,}'
    )
    log_line(
        f'[severity] positive_rate | {train_name}={train_df[TARGET_COL].mean():.4f} '
        f'{valid_name}={valid_df[TARGET_COL].mean():.4f} {holdout_name}={holdout_df[TARGET_COL].mean():.4f}'
    )

    log_line(f'[severity] fitting baseline={BASELINE_NAME}')
    baseline_start = perf_counter()
    baseline_state = fit_dummy_prior(train_df, valid_df, holdout_df, y_train, random_seed)
    baseline_seconds = round(perf_counter() - baseline_start, 2)

    log_line('[severity] building structured branch matrices')
    X_train_structured, X_valid_structured, X_holdout_structured = build_structured_matrices(train_df, valid_df, holdout_df)

    log_line('[severity] fitting structured branch')
    structured_start = perf_counter()
    structured_state = fit_branch_model(
        X_train_structured,
        X_valid_structured,
        X_holdout_structured,
        y_train,
        random_seed
    )
    structured_seconds = round(perf_counter() - structured_start, 2)

    log_line('[severity] building tuned text branch matrices')
    train_text = build_text_series(train_df, clean_mode='light', domain_cleanup=False, error_cleanup=True)
    valid_text = build_text_series(valid_df, clean_mode='light', domain_cleanup=False, error_cleanup=True)
    holdout_text = build_text_series(holdout_df, clean_mode='light', domain_cleanup=False, error_cleanup=True)
    X_train_text, X_valid_text, X_holdout_text = build_text_matrices(train_text, valid_text, holdout_text)

    log_line('[severity] fitting text branch')
    text_start = perf_counter()
    text_state = fit_branch_model(
        X_train_text,
        X_valid_text,
        X_holdout_text,
        y_train,
        random_seed
    )
    text_seconds = round(perf_counter() - text_start, 2)

    log_line(f'[severity] blending late-fusion scores with text_weight={OFFICIAL_TEXT_WEIGHT:.2f}')
    raw_valid_score = (
        (OFFICIAL_TEXT_WEIGHT * text_state['valid_score'])
        + ((1.0 - OFFICIAL_TEXT_WEIGHT) * structured_state['valid_score'])
    )
    raw_holdout_score = (
        (OFFICIAL_TEXT_WEIGHT * text_state['holdout_score'])
        + ((1.0 - OFFICIAL_TEXT_WEIGHT) * structured_state['holdout_score'])
    )

    log_line('[severity] fitting calibration candidates on valid_2025')
    sigmoid_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=random_seed)
    sigmoid_model.fit(raw_valid_score.reshape(-1, 1), y_valid)
    sigmoid_valid_score = sigmoid_model.predict_proba(raw_valid_score.reshape(-1, 1))[:, 1]
    sigmoid_holdout_score = sigmoid_model.predict_proba(raw_holdout_score.reshape(-1, 1))[:, 1]

    isotonic_model = IsotonicRegression(out_of_bounds='clip')
    isotonic_model.fit(raw_valid_score, y_valid)
    isotonic_valid_score = isotonic_model.predict(raw_valid_score)
    isotonic_holdout_score = isotonic_model.predict(raw_holdout_score)

    model_scores = {
        BASELINE_NAME: {
            'valid': baseline_state['valid_score'],
            'holdout': baseline_state['holdout_score'],
            'calibration_method': 'prior'
        },
        RAW_NAME: {
            'valid': raw_valid_score,
            'holdout': raw_holdout_score,
            'calibration_method': 'none'
        },
        SIGMOID_NAME: {
            'valid': sigmoid_valid_score,
            'holdout': sigmoid_holdout_score,
            'calibration_method': 'sigmoid'
        },
        ISOTONIC_NAME: {
            'valid': isotonic_valid_score,
            'holdout': isotonic_holdout_score,
            'calibration_method': 'isotonic'
        }
    }

    raw_row = build_score_row(RAW_NAME, valid_name, y_valid, raw_valid_score, 'none')
    sigmoid_row = build_score_row(SIGMOID_NAME, valid_name, y_valid, sigmoid_valid_score, 'sigmoid')
    isotonic_row = build_score_row(ISOTONIC_NAME, valid_name, y_valid, isotonic_valid_score, 'isotonic')
    official_model_name = pick_calibration_winner(raw_row, sigmoid_row, isotonic_row)
    log_line(f'[severity] promoted official candidate={official_model_name}')

    metrics_rows = []
    for model_name, score_info in model_scores.items():
        is_baseline = model_name == BASELINE_NAME
        is_official = model_name == official_model_name
        metrics_rows.append(
            build_score_row(
                model_name,
                valid_name,
                y_valid,
                score_info['valid'],
                score_info['calibration_method'],
                is_baseline=is_baseline,
                is_official=is_official
            )
        )
        metrics_rows.append(
            build_score_row(
                model_name,
                holdout_name,
                y_holdout,
                score_info['holdout'],
                score_info['calibration_method'],
                is_baseline=is_baseline,
                is_official=is_official
            )
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ['split', 'is_official', 'is_baseline', 'model'],
        ascending=[True, False, False, True]
    ).reset_index(drop=True)

    review_budget_rows = []
    for model_name in [BASELINE_NAME, official_model_name]:
        review_budget_rows.extend(
            build_budget_rows(
                model_name,
                valid_name,
                y_valid,
                model_scores[model_name]['valid'],
                is_baseline=model_name == BASELINE_NAME,
                is_official=model_name == official_model_name
            )
        )
        review_budget_rows.extend(
            build_budget_rows(
                model_name,
                holdout_name,
                y_holdout,
                model_scores[model_name]['holdout'],
                is_baseline=model_name == BASELINE_NAME,
                is_official=model_name == official_model_name
            )
        )
    review_budget_df = pd.DataFrame(review_budget_rows).sort_values(
        ['split', 'is_official', 'is_baseline', 'budget_fraction', 'model'],
        ascending=[True, False, False, True, True]
    ).reset_index(drop=True)

    calibration_rows = [
        build_reliability_table(y_valid, raw_valid_score, RAW_NAME, valid_name),
        build_reliability_table(y_valid, sigmoid_valid_score, SIGMOID_NAME, valid_name, is_official=official_model_name == SIGMOID_NAME),
        build_reliability_table(y_valid, isotonic_valid_score, ISOTONIC_NAME, valid_name, is_official=official_model_name == ISOTONIC_NAME),
        build_reliability_table(y_holdout, raw_holdout_score, RAW_NAME, holdout_name, is_official=official_model_name == RAW_NAME),
    ]
    if official_model_name != RAW_NAME:
        calibration_rows.append(
            build_reliability_table(
                y_holdout,
                model_scores[official_model_name]['holdout'],
                official_model_name,
                holdout_name,
                is_official=True
            )
        )
    calibration_df = pd.concat(calibration_rows, ignore_index=True)

    metrics_path = output_dir / SEVERITY_URGENCY_OFFICIAL_METRICS
    budgets_path = output_dir / SEVERITY_URGENCY_OFFICIAL_REVIEW_BUDGETS
    calibration_path = output_dir / SEVERITY_URGENCY_OFFICIAL_CALIBRATION
    manifest_path = output_dir / SEVERITY_URGENCY_OFFICIAL_MANIFEST

    metrics_df.to_csv(metrics_path, index=False)
    review_budget_df.to_csv(budgets_path, index=False)
    calibration_df.to_csv(calibration_path, index=False)

    official_valid_row = metrics_df.loc[
        metrics_df['model'].eq(official_model_name) & metrics_df['split'].eq(valid_name)
    ].iloc[0].to_dict()
    official_holdout_row = metrics_df.loc[
        metrics_df['model'].eq(official_model_name) & metrics_df['split'].eq(holdout_name)
    ].iloc[0].to_dict()
    baseline_valid_row = metrics_df.loc[
        metrics_df['model'].eq(BASELINE_NAME) & metrics_df['split'].eq(valid_name)
    ].iloc[0].to_dict()
    baseline_holdout_row = metrics_df.loc[
        metrics_df['model'].eq(BASELINE_NAME) & metrics_df['split'].eq(holdout_name)
    ].iloc[0].to_dict()

    manifest = {
        'scope': 'official severity urgency benchmark',
        'publish_status': publish_status,
        'input_path': str(input_path),
        'target_col': TARGET_COL,
        'baseline_model_name': BASELINE_NAME,
        'official_model_name': official_model_name,
        'official_model_family': 'late_fusion',
        'notebook_reference_model_name': NOTEBOOK_REFERENCE_NAME,
        'split_mode': BENCHMARK_SPLIT_MODE,
        'split_contract': {
            'train_end': split_policy['train_end'],
            'valid_end': split_policy['valid_end'],
            'train_name': train_name,
            'valid_name': valid_name,
            'holdout_name': holdout_name,
            'holdout_policy': split_policy['holdout_policy']
        },
        'split_summary': split_summary_df.to_dict(orient='records'),
        'locked_params': {
            'structured_cat_features': STRUCTURED_CAT_FEATURES,
            'structured_num_features': STRUCTURED_NUM_FEATURES,
            'text_source_col': TEXT_SOURCE_COL,
            'text_clean_mode': 'light',
            'domain_cleanup': False,
            'error_cleanup': True,
            'preserved_stop_words': sorted(PRESERVE_STOP_WORDS),
            'word_ngram_range': list(WORD_NGRAM_RANGE),
            'char_ngram_range': list(CHAR_NGRAM_RANGE),
            'word_max_features': WORD_MAX_FEATURES,
            'char_max_features': CHAR_MAX_FEATURES,
            'text_min_df': TEXT_MIN_DF,
            'sgd_alpha': OFFICIAL_SGD_ALPHA,
            'text_weight': OFFICIAL_TEXT_WEIGHT
        },
        'promotion_rule': (
            'Fit raw, sigmoid, and isotonic candidates on valid_2025; only allow a calibrated winner '
            'if recall_top_5pct stays within 0.002 of raw; if sigmoid and isotonic Brier differ by '
            '0.002 or less, prefer sigmoid; otherwise choose lower Brier; calibrated candidate must '
            'also beat raw on Brier to be promoted'
        ),
        'timing_seconds': {
            'baseline_fit': baseline_seconds,
            'structured_fit': structured_seconds,
            'text_fit': text_seconds,
            'total': round(perf_counter() - start_total, 2)
        },
        'validation_metrics': {
            'baseline': baseline_valid_row,
            'official': official_valid_row
        },
        'holdout_metrics': {
            'baseline': baseline_holdout_row,
            'official': official_holdout_row
        },
        'artifact_names': {
            'manifest': SEVERITY_URGENCY_OFFICIAL_MANIFEST,
            'metrics': SEVERITY_URGENCY_OFFICIAL_METRICS,
            'review_budgets': SEVERITY_URGENCY_OFFICIAL_REVIEW_BUDGETS,
            'calibration': SEVERITY_URGENCY_OFFICIAL_CALIBRATION
        }
    }
    write_json(manifest, manifest_path)

    log_line(f'[severity] wrote {metrics_path}')
    log_line(f'[severity] wrote {budgets_path}')
    log_line(f'[severity] wrote {calibration_path}')
    log_line(f'[severity] wrote {manifest_path}')

    return {
        'manifest': manifest,
        'metrics_df': metrics_df,
        'review_budget_df': review_budget_df,
        'calibration_df': calibration_df
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the official severity urgency benchmark from the locked notebook path'
    )
    parser.add_argument(
        '--input-path',
        default=None,
        help='Optional path to the severity case parquet or csv file'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=settings.RANDOM_SEED,
        help='Random seed used by the locked severity model'
    )
    parser.add_argument(
        '--publish-status',
        default='official',
        help='Release status recorded in the severity manifest'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    log_line("[severity] loading severity cases")
    raw_df, input_path = load_frame(SEVERITY_CASES_STEM, input_path=args.input_path)
    run_severity_pipeline(
        raw_df,
        input_path=input_path,
        output_dir=OUTPUTS_DIR,
        random_seed=args.random_seed,
        publish_status=args.publish_status
    )


if __name__ == '__main__':
    main()
