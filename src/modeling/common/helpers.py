# Consolidated helpers module for modeling tasks
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

from src.config.contracts import (
    BENCHMARK_SPLIT_MODE,
    COMPONENT_MULTILABEL_CASES_STEM,
    COMPONENT_SINGLE_LABEL_CASES_STEM,
    get_split_policy,
)

# -----------------------------------------------------------------------------
# Core dataset and split definitions
# -----------------------------------------------------------------------------
ID_COL = 'odino'
DATE_COL = 'ldate'
TARGET_COL = 'component_group'
MULTI_TARGET_COL = 'component_groups'
SINGLE_INPUT_STEM = COMPONENT_SINGLE_LABEL_CASES_STEM
MULTI_INPUT_STEM = COMPONENT_MULTILABEL_CASES_STEM
DEFAULT_SELECTION_SEEDS = [42, 43, 44, 45, 46]
MAX_TOP_K = 3

STATE_REGION_MAP = {
    'CT': 'NORTHEAST',
    'ME': 'NORTHEAST',
    'MA': 'NORTHEAST',
    'NH': 'NORTHEAST',
    'RI': 'NORTHEAST',
    'VT': 'NORTHEAST',
    'NJ': 'NORTHEAST',
    'NY': 'NORTHEAST',
    'PA': 'NORTHEAST',
    'IL': 'MIDWEST',
    'IN': 'MIDWEST',
    'MI': 'MIDWEST',
    'OH': 'MIDWEST',
    'WI': 'MIDWEST',
    'IA': 'MIDWEST',
    'KS': 'MIDWEST',
    'MN': 'MIDWEST',
    'MO': 'MIDWEST',
    'NE': 'MIDWEST',
    'ND': 'MIDWEST',
    'SD': 'MIDWEST',
    'DE': 'SOUTH',
    'FL': 'SOUTH',
    'GA': 'SOUTH',
    'MD': 'SOUTH',
    'NC': 'SOUTH',
    'SC': 'SOUTH',
    'VA': 'SOUTH',
    'DC': 'SOUTH',
    'WV': 'SOUTH',
    'AL': 'SOUTH',
    'KY': 'SOUTH',
    'MS': 'SOUTH',
    'TN': 'SOUTH',
    'AR': 'SOUTH',
    'LA': 'SOUTH',
    'OK': 'SOUTH',
    'TX': 'SOUTH',
    'AZ': 'WEST',
    'CO': 'WEST',
    'ID': 'WEST',
    'MT': 'WEST',
    'NV': 'WEST',
    'NM': 'WEST',
    'UT': 'WEST',
    'WY': 'WEST',
    'AK': 'WEST',
    'CA': 'WEST',
    'HI': 'WEST',
    'OR': 'WEST',
    'WA': 'WEST',
    'AS': 'TERRITORY',
    'FM': 'TERRITORY',
    'GU': 'TERRITORY',
    'MH': 'TERRITORY',
    'MP': 'TERRITORY',
    'PR': 'TERRITORY',
    'PW': 'TERRITORY',
    'VI': 'TERRITORY',
    'AA': 'MILITARY',
    'AE': 'MILITARY',
    'AP': 'MILITARY'
}
VEHICLE_AGE_BUCKETS = [-1, 0, 3, 7, 12, 200]
VEHICLE_AGE_LABELS = ['AGE_0', 'AGE_1_3', 'AGE_4_7', 'AGE_8_12', 'AGE_13_PLUS']


# -----------------------------------------------------------------------------
# Feature ladder
# -----------------------------------------------------------------------------
CAT_FEATURES = [
    'mfr_name',
    'maketxt',
    'modeltxt',
    'state',
    'state_region',
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
    'repaired_yn',
    'vehicle_age_bucket'
]

NUM_FEATURES = [
    'yeartxt',
    'miles',
    'veh_speed',
    'injured',
    'lag_days_safe',
    'complaint_year',
    'complaint_month',
    'complaint_quarter',
    'vehicle_age_years',
    'prior_cmpl_mfr_all',
    'prior_cmpl_make_model_all',
    'prior_cmpl_make_model_year_all',
    'prior_severity_share_mfr_all',
    'prior_severity_share_make_model_all',
    'prior_severity_share_make_model_year_all'
]

FLAG_FEATURES = [
    'miles_missing_flag',
    'veh_speed_missing_flag',
    'miles_zero_flag',
    'veh_speed_zero_flag',
    'faildate_trusted_flag',
    'flag_date_order_bad',
    'flag_fail_pre_model',
    'flag_fail_pre_model_far',
    'severity_primary_flag',
    'severity_broad_flag',
    'flag_year_out_of_range'
]

CORE_STRUCTURED_FEATURES = [
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
    'yeartxt',
    'miles',
    'veh_speed',
    'injured',
    'lag_days_safe',
    'miles_missing_flag',
    'veh_speed_missing_flag',
    'miles_zero_flag',
    'veh_speed_zero_flag'
]

WAVE1_BUNDLE_DEFS = {
    'incident_bundle': [
        'police_rpt_yn',
        'repaired_yn',
        'severity_broad_flag',
        'severity_primary_flag'
    ],
    'date_quality_bundle': [
        'faildate_trusted_flag',
        'flag_date_order_bad',
        'flag_fail_pre_model',
        'flag_fail_pre_model_far',
        'flag_year_out_of_range'
    ],
    'geo_time_bundle': [
        'complaint_year',
        'complaint_month',
        'complaint_quarter',
        'vehicle_age_years',
        'vehicle_age_bucket',
        'state_region'
    ],
    'cohort_history_bundle': [
        'prior_cmpl_mfr_all',
        'prior_cmpl_make_model_all',
        'prior_cmpl_make_model_year_all',
        'prior_severity_share_mfr_all',
        'prior_severity_share_make_model_all',
        'prior_severity_share_make_model_year_all'
    ]
}

WAVE1_PAIRWISE_FAMILIES = {
    'wave1_incident_geo_time': ['incident_bundle', 'geo_time_bundle'],
    'wave1_incident_cohort_history': ['incident_bundle', 'cohort_history_bundle'],
    'wave1_date_quality_cohort_history': ['date_quality_bundle', 'cohort_history_bundle']
}

WAVE1_PRUNE_QUEUE = [
    'miles_zero_flag',
    'fire',
    'medical_attn',
    'injured',
    'fuel_sys',
    'veh_speed_missing_flag',
    'vehicles_towed_yn'
]

WAVE1_PRUNE_PROTECTED = [
    'mfr_name',
    'maketxt',
    'modeltxt',
    'yeartxt',
    'state',
    'cmpl_type',
    'miles',
    'veh_speed',
    'lag_days_safe'
]

WAVE1_EXCLUDED_COLS = [
    'orig_owner_yn',
    'anti_brakes_yn',
    'cruise_cont_yn',
    'num_cyls',
    'deaths',
    'faildate_untrusted_flag',
    'flag_year_unknown',
    'flag_speed_high',
    'flag_miles_high'
]

AUDIT_ONLY_COLS = [
    'source_era',
    'source_zip',
    'source_file'
]

FEATURE_SET_DEFS = {
    'core_structured': CORE_STRUCTURED_FEATURES,
    'core_plus_quality': [
        *CORE_STRUCTURED_FEATURES,
        'faildate_trusted_flag',
        'flag_date_order_bad',
        'flag_fail_pre_model',
        'flag_fail_pre_model_far'
    ],
    'core_plus_stable_incident': [
        *CORE_STRUCTURED_FEATURES,
        'police_rpt_yn',
        'repaired_yn',
        'faildate_trusted_flag',
        'flag_date_order_bad',
        'flag_fail_pre_model',
        'flag_fail_pre_model_far'
    ],
    'wave1_incident_bundle': CORE_STRUCTURED_FEATURES + WAVE1_BUNDLE_DEFS['incident_bundle'],
    'wave1_date_quality_bundle': CORE_STRUCTURED_FEATURES + WAVE1_BUNDLE_DEFS['date_quality_bundle'],
    'wave1_geo_time_bundle': CORE_STRUCTURED_FEATURES + WAVE1_BUNDLE_DEFS['geo_time_bundle'],
    'wave1_cohort_history_bundle': CORE_STRUCTURED_FEATURES + WAVE1_BUNDLE_DEFS['cohort_history_bundle'],
    'wave1_incident_geo_time': (
        CORE_STRUCTURED_FEATURES
        + WAVE1_BUNDLE_DEFS['incident_bundle']
        + WAVE1_BUNDLE_DEFS['geo_time_bundle']
    ),
    'wave1_incident_cohort_history': (
        CORE_STRUCTURED_FEATURES
        + WAVE1_BUNDLE_DEFS['incident_bundle']
        + WAVE1_BUNDLE_DEFS['cohort_history_bundle']
    ),
    'wave1_date_quality_cohort_history': (
        CORE_STRUCTURED_FEATURES
        + WAVE1_BUNDLE_DEFS['date_quality_bundle']
        + WAVE1_BUNDLE_DEFS['cohort_history_bundle']
    )
}

BENCHMARK_FEATURE_SET_NAMES = [
    'core_structured',
    'core_plus_quality',
    'core_plus_stable_incident'
]

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

# -----------------------------------------------------------------------------
# Feature helpers
# -----------------------------------------------------------------------------
def dedupe_feature_cols(feature_cols):
    seen = set()
    ordered = []
    for column in feature_cols:
        if column in seen:
            continue
        seen.add(column)
        ordered.append(column)
    return ordered


def classify_feature_columns(feature_cols):
    feature_cols = dedupe_feature_cols(feature_cols)
    cat_cols = [column for column in feature_cols if column in CAT_FEATURES]
    num_cols = [column for column in feature_cols if column in NUM_FEATURES]
    flag_cols = [column for column in feature_cols if column in FLAG_FEATURES]
    return {
        'feature_cols': feature_cols,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'flag_cols': flag_cols
    }


def feature_manifest(feature_set_name):
    if feature_set_name not in FEATURE_SET_DEFS:
        choices = ', '.join(sorted(FEATURE_SET_DEFS))
        raise ValueError(f'Unknown feature set {feature_set_name}. Choices: {choices}')

    feature_cols = list(FEATURE_SET_DEFS[feature_set_name])
    feature_info = classify_feature_columns(feature_cols)

    return {
        'feature_set_name': feature_set_name,
        'feature_cols': feature_info['feature_cols'],
        'cat_cols': feature_info['cat_cols'],
        'num_cols': feature_info['num_cols'],
        'flag_cols': feature_info['flag_cols']
    }


def compose_feature_manifest(feature_set_name, add_cols=None, remove_cols=None, base_feature_set='core_structured'):
    base_manifest = feature_manifest(base_feature_set)
    add_cols = [] if add_cols is None else list(add_cols)
    remove_cols = [] if remove_cols is None else list(remove_cols)
    feature_cols = [column for column in base_manifest['feature_cols'] if column not in remove_cols]
    feature_cols.extend(add_cols)
    feature_info = classify_feature_columns(feature_cols)
    return {
        'feature_set_name': feature_set_name,
        'base_feature_set': base_feature_set,
        'added_cols': dedupe_feature_cols(add_cols),
        'removed_cols': dedupe_feature_cols(remove_cols),
        'feature_cols': feature_info['feature_cols'],
        'cat_cols': feature_info['cat_cols'],
        'num_cols': feature_info['num_cols'],
        'flag_cols': feature_info['flag_cols']
    }


def all_feature_columns():
    cols = []
    for feature_cols in FEATURE_SET_DEFS.values():
        cols.extend(feature_cols)
    return dedupe_feature_cols(cols)


def build_case_sort_keys(df):
    work = df.copy()
    work['__odino_num'] = pd.to_numeric(work[ID_COL], errors='coerce')
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors='coerce')
    return work.sort_values(
        [DATE_COL, '__odino_num', ID_COL],
        na_position='last'
    ).reset_index(drop=True)


def derive_vehicle_age_features(df):
    work = df.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors='coerce')
    work['complaint_year'] = work[DATE_COL].dt.year.astype('Int64')
    work['complaint_month'] = work[DATE_COL].dt.month.astype('Int64')
    work['complaint_quarter'] = work[DATE_COL].dt.quarter.astype('Int64')

    model_year = pd.to_numeric(work['yeartxt'], errors='coerce')
    vehicle_age = work['complaint_year'].astype('Float64') - model_year.astype('Float64')
    vehicle_age = vehicle_age.where(vehicle_age.ge(0))
    work['vehicle_age_years'] = vehicle_age
    age_bucket = pd.cut(
        vehicle_age,
        bins=VEHICLE_AGE_BUCKETS,
        labels=VEHICLE_AGE_LABELS,
        include_lowest=True
    )
    work['vehicle_age_bucket'] = age_bucket.astype('string')
    return work


def derive_state_region(df):
    work = df.copy()
    state = work['state'].astype('string').str.upper()
    work['state_region'] = state.map(STATE_REGION_MAP).fillna('UNKNOWN').astype(str)
    return work


def derive_prior_history_features(df):
    work = build_case_sort_keys(df)
    work['__event_day'] = work[DATE_COL].dt.normalize()
    work['__severity_int'] = work['severity_broad_flag'].fillna(False).astype(int)

    history_specs = [
        ('mfr_name', 'prior_cmpl_mfr_all', 'prior_severity_share_mfr_all'),
        (['maketxt', 'modeltxt'], 'prior_cmpl_make_model_all', 'prior_severity_share_make_model_all'),
        (
            ['maketxt', 'modeltxt', 'yeartxt'],
            'prior_cmpl_make_model_year_all',
            'prior_severity_share_make_model_year_all'
        )
    ]

    for key_cols, count_name, share_name in history_specs:
        key_cols = [key_cols] if isinstance(key_cols, str) else list(key_cols)
        daily_df = (
            work.groupby(key_cols + ['__event_day'], dropna=False, sort=False)
            .agg(
                day_case_count=(ID_COL, 'size'),
                day_severity_count=('__severity_int', 'sum')
            )
            .reset_index()
        )
        grouped = daily_df.groupby(key_cols, sort=False, dropna=False)
        daily_df['__prior_case_count'] = grouped['day_case_count'].cumsum() - daily_df['day_case_count']
        daily_df['__prior_severity_count'] = grouped['day_severity_count'].cumsum() - daily_df['day_severity_count']
        daily_df[count_name] = daily_df['__prior_case_count'].astype('Int64')
        prior_share = daily_df['__prior_severity_count'].astype(float) / daily_df['__prior_case_count'].replace(0, pd.NA)
        daily_df[share_name] = pd.Series(prior_share, index=daily_df.index).astype('Float64')
        work = work.merge(
            daily_df[key_cols + ['__event_day', count_name, share_name]],
            on=key_cols + ['__event_day'],
            how='left',
            validate='many_to_one'
        )

    return work.drop(columns=['__odino_num', '__event_day', '__severity_int'])


def add_requested_case_features(df, feature_cols):
    work = df.copy()
    requested = set(feature_cols)
    missing = requested - set(work.columns)

    age_feature_cols = {
        'complaint_year',
        'complaint_month',
        'complaint_quarter',
        'vehicle_age_years',
        'vehicle_age_bucket'
    }
    history_feature_cols = {
        'prior_cmpl_mfr_all',
        'prior_cmpl_make_model_all',
        'prior_cmpl_make_model_year_all',
        'prior_severity_share_mfr_all',
        'prior_severity_share_make_model_all',
        'prior_severity_share_make_model_year_all'
    }

    if missing & age_feature_cols:
        work = derive_vehicle_age_features(work)
    if 'state_region' in missing:
        work = derive_state_region(work)
    if missing & history_feature_cols:
        work = derive_prior_history_features(work)
    return work


def validate_unseen_single_label(train_target, eval_target, split_name):
    unseen = sorted(set(eval_target) - set(train_target))
    if unseen:
        unseen_text = ', '.join(unseen)
        raise ValueError(f'{split_name} has unseen target labels: {unseen_text}')


def require_case_columns(df, feature_cols, target_col=TARGET_COL):
    required = list(feature_cols) + [ID_COL, DATE_COL, target_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        missing_text = ', '.join(missing)
        raise ValueError(f'Missing required columns: {missing_text}')


def prep_single_label_cases(df, feature_cols):
    work = add_requested_case_features(df, feature_cols)
    if 'single_label_keep_flag' in work.columns:
        keep_mask = work['single_label_keep_flag'].fillna(False).astype(bool)
        work = work.loc[keep_mask].copy()
    require_case_columns(work, feature_cols, target_col=TARGET_COL)
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors='coerce')
    if work[DATE_COL].isna().any():
        missing_dates = int(work[DATE_COL].isna().sum())
        raise ValueError(f'Found {missing_dates} rows with invalid {DATE_COL} values')

    feature_info = classify_feature_columns(feature_cols)
    for column in feature_info['cat_cols']:
        work[column] = work[column].astype('string').fillna('__MISSING__').astype(str)

    for column in feature_info['num_cols'] + feature_info['flag_cols']:
        work[column] = pd.to_numeric(work[column], errors='coerce')

    return work.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)


def prep_multi_label_cases(df, feature_cols):
    require_case_columns(df, feature_cols, target_col=MULTI_TARGET_COL)
    work = prep_single_label_cases(
        df.rename(columns={MULTI_TARGET_COL: TARGET_COL}),
        feature_cols
    ).rename(columns={TARGET_COL: MULTI_TARGET_COL})
    work[MULTI_TARGET_COL] = (
        work[MULTI_TARGET_COL]
        .astype('string')
        .fillna('')
        .astype(str)
    )
    return work


def subset_case_frame(df, feature_cols, target_col=TARGET_COL):
    require_case_columns(df, feature_cols, target_col=target_col)
    keep_cols = dedupe_feature_cols([ID_COL, DATE_COL, target_col, *feature_cols])
    return df[keep_cols].copy()


def split_single_label_cases_by_mode(df, split_mode=BENCHMARK_SPLIT_MODE):
    policy = get_split_policy(split_mode)

    if split_mode == BENCHMARK_SPLIT_MODE:
        train_df = df.loc[df[DATE_COL] <= policy['train_end']].copy()
        valid_df = df.loc[(df[DATE_COL] > policy['train_end']) & (df[DATE_COL] <= policy['valid_end'])].copy()
        holdout_df = df.loc[df[DATE_COL] > policy['valid_end']].copy()

        if train_df.empty:
            raise ValueError('Training split is empty')
        if valid_df.empty:
            raise ValueError('Validation split is empty')
        if holdout_df.empty:
            raise ValueError('Holdout split is empty')

        validate_unseen_single_label(train_df[TARGET_COL], valid_df[TARGET_COL], 'Validation split')
        validate_unseen_single_label(train_df[TARGET_COL], holdout_df[TARGET_COL], 'Holdout split')

        split_rows = [
            {
                'split': policy['train_name'],
                'rows': int(len(train_df)),
                'cases': int(train_df[ID_COL].nunique()),
                'date_min': train_df[DATE_COL].min(),
                'date_max': train_df[DATE_COL].max(),
                'target_groups': int(train_df[TARGET_COL].nunique())
            },
            {
                'split': policy['valid_name'],
                'rows': int(len(valid_df)),
                'cases': int(valid_df[ID_COL].nunique()),
                'date_min': valid_df[DATE_COL].min(),
                'date_max': valid_df[DATE_COL].max(),
                'target_groups': int(valid_df[TARGET_COL].nunique())
            },
            {
                'split': policy['holdout_name'],
                'rows': int(len(holdout_df)),
                'cases': int(holdout_df[ID_COL].nunique()),
                'date_min': holdout_df[DATE_COL].min(),
                'date_max': holdout_df[DATE_COL].max(),
                'target_groups': int(holdout_df[TARGET_COL].nunique())
            }
        ]
        return {
            policy['train_name']: train_df,
            policy['valid_name']: valid_df,
            policy['holdout_name']: holdout_df,
            'split_df': pd.DataFrame(split_rows)
        }

    train_df = df.loc[df[DATE_COL] <= policy['train_core_end']].copy()
    screen_df = df.loc[(df[DATE_COL] > policy['train_core_end']) & (df[DATE_COL] <= policy['screen_end'])].copy()
    select_df = df.loc[(df[DATE_COL] > policy['screen_end']) & (df[DATE_COL] <= policy['select_end'])].copy()
    holdout_df = df.loc[df[DATE_COL] > policy['select_end']].copy()

    if train_df.empty:
        raise ValueError('Training split is empty')
    if screen_df.empty:
        raise ValueError('Screen split is empty')
    if select_df.empty:
        raise ValueError('Select split is empty')
    if holdout_df.empty:
        raise ValueError('Holdout split is empty')

    dev_screen_df = pd.concat([train_df, screen_df], ignore_index=True).sort_values([DATE_COL, ID_COL]).reset_index(drop=True)
    dev_select_df = pd.concat([dev_screen_df, select_df], ignore_index=True).sort_values([DATE_COL, ID_COL]).reset_index(drop=True)
    validate_unseen_single_label(train_df[TARGET_COL], screen_df[TARGET_COL], 'Screen split')
    validate_unseen_single_label(dev_screen_df[TARGET_COL], select_df[TARGET_COL], 'Select split')
    validate_unseen_single_label(dev_select_df[TARGET_COL], holdout_df[TARGET_COL], 'Holdout split')

    split_rows = [
        {
            'split': policy['train_name'],
            'rows': int(len(train_df)),
            'cases': int(train_df[ID_COL].nunique()),
            'date_min': train_df[DATE_COL].min(),
            'date_max': train_df[DATE_COL].max(),
            'target_groups': int(train_df[TARGET_COL].nunique())
        },
        {
            'split': policy['screen_name'],
            'rows': int(len(screen_df)),
            'cases': int(screen_df[ID_COL].nunique()),
            'date_min': screen_df[DATE_COL].min(),
            'date_max': screen_df[DATE_COL].max(),
            'target_groups': int(screen_df[TARGET_COL].nunique())
        },
        {
            'split': policy['select_name'],
            'rows': int(len(select_df)),
            'cases': int(select_df[ID_COL].nunique()),
            'date_min': select_df[DATE_COL].min(),
            'date_max': select_df[DATE_COL].max(),
            'target_groups': int(select_df[TARGET_COL].nunique())
        },
        {
            'split': policy['holdout_name'],
            'rows': int(len(holdout_df)),
            'cases': int(holdout_df[ID_COL].nunique()),
            'date_min': holdout_df[DATE_COL].min(),
            'date_max': holdout_df[DATE_COL].max(),
            'target_groups': int(holdout_df[TARGET_COL].nunique())
        }
    ]
    return {
        policy['train_name']: train_df,
        policy['screen_name']: screen_df,
        policy['select_name']: select_df,
        policy['holdout_name']: holdout_df,
        policy['select_train_name']: dev_screen_df,
        policy['dev_name']: dev_select_df,
        'split_df': pd.DataFrame(split_rows)
    }


def split_single_label_cases(df):
    split_parts = split_single_label_cases_by_mode(df, split_mode=BENCHMARK_SPLIT_MODE)
    policy = get_split_policy(BENCHMARK_SPLIT_MODE)
    return (
        split_parts[policy['train_name']],
        split_parts[policy['valid_name']],
        split_parts[policy['holdout_name']],
        split_parts['split_df']
    )


def split_multi_label_cases_by_mode(df, split_mode=BENCHMARK_SPLIT_MODE):
    policy = get_split_policy(split_mode)

    if split_mode == BENCHMARK_SPLIT_MODE:
        train_df = df.loc[df[DATE_COL] <= policy['train_end']].copy()
        valid_df = df.loc[(df[DATE_COL] > policy['train_end']) & (df[DATE_COL] <= policy['valid_end'])].copy()
        holdout_df = df.loc[df[DATE_COL] > policy['valid_end']].copy()

        if train_df.empty:
            raise ValueError('Training split is empty')
        if valid_df.empty:
            raise ValueError('Validation split is empty')
        if holdout_df.empty:
            raise ValueError('Holdout split is empty')

        split_rows = [
            {
                'split': policy['train_name'],
                'rows': int(len(train_df)),
                'cases': int(train_df[ID_COL].nunique()),
                'date_min': train_df[DATE_COL].min(),
                'date_max': train_df[DATE_COL].max()
            },
            {
                'split': policy['valid_name'],
                'rows': int(len(valid_df)),
                'cases': int(valid_df[ID_COL].nunique()),
                'date_min': valid_df[DATE_COL].min(),
                'date_max': valid_df[DATE_COL].max()
            },
            {
                'split': policy['holdout_name'],
                'rows': int(len(holdout_df)),
                'cases': int(holdout_df[ID_COL].nunique()),
                'date_min': holdout_df[DATE_COL].min(),
                'date_max': holdout_df[DATE_COL].max()
            }
        ]
        return {
            policy['train_name']: train_df,
            policy['valid_name']: valid_df,
            policy['holdout_name']: holdout_df,
            'split_df': pd.DataFrame(split_rows)
        }

    train_df = df.loc[df[DATE_COL] <= policy['train_core_end']].copy()
    screen_df = df.loc[(df[DATE_COL] > policy['train_core_end']) & (df[DATE_COL] <= policy['screen_end'])].copy()
    select_df = df.loc[(df[DATE_COL] > policy['screen_end']) & (df[DATE_COL] <= policy['select_end'])].copy()
    holdout_df = df.loc[df[DATE_COL] > policy['select_end']].copy()

    if train_df.empty:
        raise ValueError('Training split is empty')
    if screen_df.empty:
        raise ValueError('Screen split is empty')
    if select_df.empty:
        raise ValueError('Select split is empty')
    if holdout_df.empty:
        raise ValueError('Holdout split is empty')

    dev_screen_df = pd.concat([train_df, screen_df], ignore_index=True).sort_values([DATE_COL, ID_COL]).reset_index(drop=True)
    dev_select_df = pd.concat([dev_screen_df, select_df], ignore_index=True).sort_values([DATE_COL, ID_COL]).reset_index(drop=True)
    split_rows = [
        {
            'split': policy['train_name'],
            'rows': int(len(train_df)),
            'cases': int(train_df[ID_COL].nunique()),
            'date_min': train_df[DATE_COL].min(),
            'date_max': train_df[DATE_COL].max()
        },
        {
            'split': policy['screen_name'],
            'rows': int(len(screen_df)),
            'cases': int(screen_df[ID_COL].nunique()),
            'date_min': screen_df[DATE_COL].min(),
            'date_max': screen_df[DATE_COL].max()
        },
        {
            'split': policy['select_name'],
            'rows': int(len(select_df)),
            'cases': int(select_df[ID_COL].nunique()),
            'date_min': select_df[DATE_COL].min(),
            'date_max': select_df[DATE_COL].max()
        },
        {
            'split': policy['holdout_name'],
            'rows': int(len(holdout_df)),
            'cases': int(holdout_df[ID_COL].nunique()),
            'date_min': holdout_df[DATE_COL].min(),
            'date_max': holdout_df[DATE_COL].max()
        }
    ]
    return {
        policy['train_name']: train_df,
        policy['screen_name']: screen_df,
        policy['select_name']: select_df,
        policy['holdout_name']: holdout_df,
        policy['select_train_name']: dev_screen_df,
        policy['dev_name']: dev_select_df,
        'split_df': pd.DataFrame(split_rows)
    }


def split_multi_label_cases(df):
    split_parts = split_multi_label_cases_by_mode(df, split_mode=BENCHMARK_SPLIT_MODE)
    policy = get_split_policy(BENCHMARK_SPLIT_MODE)
    return (
        split_parts[policy['train_name']],
        split_parts[policy['valid_name']],
        split_parts[policy['holdout_name']],
        split_parts['split_df']
    )


# -----------------------------------------------------------------------------
# Multiclass scoring
# -----------------------------------------------------------------------------
def score_multiclass_from_proba(y_true, proba, classes):
    classes = np.asarray(classes)
    top_k = min(MAX_TOP_K, len(classes))
    pred = classes[np.argmax(proba, axis=1)]
    metrics = {
        'top_1_accuracy': round(float(accuracy_score(y_true, pred)), 4),
        'macro_f1': round(float(f1_score(y_true, pred, average='macro')), 4),
        'top_3_accuracy': round(
            float(top_k_accuracy_score(y_true, proba, labels=classes, k=top_k)),
            4
        )
    }
    return pred, metrics


def build_multiclass_metric_row(model_name, stage_name, split_name, y_true, proba, classes, fit_seconds=pd.NA, selected_iteration=pd.NA):
    _, metrics = score_multiclass_from_proba(y_true, proba, classes)
    return {
        'model': model_name,
        'stage': stage_name,
        'split': split_name,
        'rows': int(len(y_true)),
        'fit_seconds': fit_seconds,
        'selected_iteration': selected_iteration,
        **metrics
    }


def build_multiclass_class_df(y_true, pred, classes):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=classes,
        zero_division=0
    )
    return pd.DataFrame(
        {
            'component_group': classes,
            'support': support,
            'precision': np.round(precision, 4),
            'recall': np.round(recall, 4),
            'f1': np.round(f1, 4)
        }
    ).sort_values(['support', 'f1'], ascending=[False, False]).reset_index(drop=True)


def build_multiclass_confusion_df(y_true, pred, focus_groups):
    counts = pd.crosstab(
        pd.Categorical(y_true, categories=focus_groups),
        pd.Categorical(pred, categories=focus_groups),
        dropna=False
    )
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)

    rows = []
    for true_group in counts.index:
        for pred_group in counts.columns:
            rows.append(
                {
                    'true_group': true_group,
                    'pred_group': pred_group,
                    'count': int(counts.loc[true_group, pred_group]),
                    'row_share': round(float(shares.loc[true_group, pred_group]), 4)
                    if not pd.isna(shares.loc[true_group, pred_group])
                    else np.nan
                }
            )

    return pd.DataFrame(rows)


def build_multiclass_calibration_df(y_true, proba, classes, bins=10):
    classes = np.asarray(classes)
    pred_idx = np.argmax(proba, axis=1)
    pred = classes[pred_idx]
    confidence = proba[np.arange(len(proba)), pred_idx]
    correct = pred == np.asarray(y_true)
    top_series = pd.DataFrame(
        {
            'confidence': confidence,
            'correct': correct
        }
    )

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    top_series['bin'] = pd.cut(
        top_series['confidence'],
        bins=bin_edges,
        include_lowest=True
    )
    bin_df = (
        top_series.groupby('bin', observed=False)
        .agg(
            count=('confidence', 'size'),
            accuracy=('correct', 'mean'),
            avg_confidence=('confidence', 'mean')
        )
        .reset_index()
    )
    bin_df['share'] = bin_df['count'] / max(len(top_series), 1)
    bin_df['gap'] = (bin_df['accuracy'] - bin_df['avg_confidence']).abs()
    ece = float((bin_df['share'] * bin_df['gap']).sum())

    truth_idx = pd.Categorical(y_true, categories=classes).codes
    one_hot = np.zeros_like(proba)
    valid_mask = truth_idx >= 0
    one_hot[np.arange(len(proba))[valid_mask], truth_idx[valid_mask]] = 1.0
    brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))

    overall_row = pd.DataFrame(
        [
            {
                'section': 'overall',
                'bin': 'overall',
                'count': int(len(top_series)),
                'share': 1.0,
                'accuracy': round(float(correct.mean()), 4),
                'avg_confidence': round(float(confidence.mean()), 4),
                'gap': round(float(abs(correct.mean() - confidence.mean())), 4),
                'ece': round(ece, 4),
                'multiclass_brier': round(brier, 6)
            }
        ]
    )

    bin_df = bin_df.assign(
        section='bin',
        ece=np.nan,
        multiclass_brier=np.nan
    )
    bin_df['accuracy'] = bin_df['accuracy'].round(4)
    bin_df['avg_confidence'] = bin_df['avg_confidence'].round(4)
    bin_df['share'] = bin_df['share'].round(4)
    bin_df['gap'] = bin_df['gap'].round(4)
    bin_df['bin'] = bin_df['bin'].astype(str)

    return pd.concat(
        [overall_row, bin_df[['section', 'bin', 'count', 'share', 'accuracy', 'avg_confidence', 'gap', 'ece', 'multiclass_brier']]],
        ignore_index=True
    )


# -----------------------------------------------------------------------------
# CatBoost helpers
# -----------------------------------------------------------------------------
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

def prep_catboost_frames(train_df, eval_df, feature_info):
    X_train = train_df[feature_info['feature_cols']].copy()
    X_eval = eval_df[feature_info['feature_cols']].copy()

    for column in feature_info['cat_cols']:
        X_train[column] = X_train[column].astype('string').fillna('__MISSING__').astype(str)
        X_eval[column] = X_eval[column].astype('string').fillna('__MISSING__').astype(str)

    for column in feature_info['num_cols'] + feature_info['flag_cols']:
        X_train[column] = pd.to_numeric(X_train[column], errors='coerce')
        X_eval[column] = pd.to_numeric(X_eval[column], errors='coerce')

    return X_train, X_eval


def pick_best_iteration(model, X_valid, y_valid, eval_period=1):
    best = None
    classes = model.classes_
    eval_period = max(int(eval_period), 1)

    for iteration_idx, proba in enumerate(
        model.staged_predict_proba(X_valid, eval_period=eval_period),
        start=1
    ):
        current_iteration = min(iteration_idx * eval_period, int(model.tree_count_))
        _, metrics = score_multiclass_from_proba(y_valid, proba, classes)
        candidate = {
            'selected_iteration': current_iteration,
            **metrics
        }

        if best is None:
            best = candidate
            continue

        ranking = (
            candidate['macro_f1'],
            candidate['top_1_accuracy'],
            candidate['top_3_accuracy'],
            -candidate['selected_iteration']
        )
        best_ranking = (
            best['macro_f1'],
            best['top_1_accuracy'],
            best['top_3_accuracy'],
            -best['selected_iteration']
        )
        if ranking > best_ranking:
            best = candidate

    if best is None:
        raise ValueError('Unable to select a best CatBoost iteration')
    return best


def select_catboost_iteration(model, X_valid, y_valid, eval_period, thresholds, min_positive_labels):
    best = None
    total_trees = int(model.tree_count_)
    eval_period = max(int(eval_period), 1)

    for step_idx, proba in enumerate(model.staged_predict_proba(X_valid, eval_period=eval_period), start=1):
        current_iteration = min(step_idx * eval_period, total_trees)
        threshold_choice = select_multilabel_threshold(
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


def fit_catboost_with_external_selection(
    train_df,
    valid_df,
    feature_info,
    params,
    task_type='CPU',
    devices='0',
    random_seed=None,
    verbose=0,
    selection_eval_period=1,
    include_train_outputs=True,
    include_valid_outputs=True
):
    X_train, X_valid = prep_catboost_frames(train_df, valid_df, feature_info)
    y_train = train_df[TARGET_COL].copy()
    y_valid = valid_df[TARGET_COL].copy()

    model = build_catboost_model(
        params,
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

    selection_start = perf_counter()
    best = pick_best_iteration(model, X_valid, y_valid, eval_period=selection_eval_period)
    selection_seconds = round(perf_counter() - selection_start, 2)
    selected_iteration = int(best['selected_iteration'])
    train_pred = None
    train_proba = None
    train_metrics = None
    valid_pred = None
    valid_proba = None
    valid_metrics = {
        'top_1_accuracy': float(best['top_1_accuracy']),
        'macro_f1': float(best['macro_f1']),
        'top_3_accuracy': float(best['top_3_accuracy'])
    }

    prediction_start = perf_counter()
    if include_train_outputs:
        train_proba = model.predict_proba(X_train, ntree_end=selected_iteration)
        train_pred, train_metrics = score_multiclass_from_proba(y_train, train_proba, model.classes_)
    if include_valid_outputs:
        valid_proba = model.predict_proba(X_valid, ntree_end=selected_iteration)
        valid_pred, valid_metrics = score_multiclass_from_proba(y_valid, valid_proba, model.classes_)
    prediction_seconds = round(perf_counter() - prediction_start, 2) if (
        include_train_outputs or include_valid_outputs
    ) else 0.0

    return {
        'model': model,
        'classes': model.classes_,
        'fit_seconds': fit_seconds,
        'selection_seconds': selection_seconds,
        'prediction_seconds': prediction_seconds,
        'selected_iteration': selected_iteration,
        'train_pred': train_pred,
        'train_proba': train_proba,
        'train_metrics': train_metrics,
        'valid_pred': valid_pred,
        'valid_proba': valid_proba,
        'valid_metrics': valid_metrics,
        'raw_best_score': model.get_best_score()
    }


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


# -----------------------------------------------------------------------------
# Multi-label helpers
# -----------------------------------------------------------------------------
def parse_pipe_labels(series):
    labels = []
    for value in series.astype('string').fillna(''):
        parts = [part for part in str(value).split('|') if part]
        labels.append(parts)
    return labels


def score_multilabel_predictions(y_true, y_pred, proba, top_k=3):
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    top_k = min(int(top_k), proba.shape[1])
    top_idx = np.argsort(proba, axis=1)[:, -top_k:][:, ::-1]

    recall_rows = []
    precision_rows = []
    for row_idx in range(len(truth)):
        true_set = set(np.flatnonzero(truth[row_idx] > 0))
        pred_set = set(top_idx[row_idx])
        overlap = len(true_set & pred_set)
        recall_rows.append(overlap / max(len(true_set), 1))
        precision_rows.append(overlap / max(top_k, 1))

    coverage = float(pred.any(axis=0).mean())
    return {
        'micro_f1': round(float(f1_score(truth, pred, average='micro', zero_division=0)), 4),
        'macro_f1': round(float(f1_score(truth, pred, average='macro', zero_division=0)), 4),
        'recall_at_3': round(float(np.mean(recall_rows)), 4),
        'precision_at_3': round(float(np.mean(precision_rows)), 4),
        'label_coverage': round(coverage, 4)
    }


def apply_multilabel_threshold(proba, threshold, min_positive_labels=0):
    proba = np.asarray(proba)
    pred = (proba >= float(threshold)).astype(int)
    min_positive_labels = max(int(min_positive_labels), 0)

    if min_positive_labels == 0 or pred.size == 0:
        return pred

    min_positive_labels = min(min_positive_labels, pred.shape[1])
    missing_mask = pred.sum(axis=1) < min_positive_labels
    if not np.any(missing_mask):
        return pred

    top_idx = np.argsort(proba[missing_mask], axis=1)[:, -min_positive_labels:]
    missing_rows = np.flatnonzero(missing_mask)
    for row_idx, column_idx in zip(missing_rows, top_idx):
        pred[row_idx, column_idx] = 1
    return pred


def select_multilabel_threshold(y_true, proba, thresholds=None, min_positive_labels=0):
    thresholds = thresholds or [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    best = None

    for threshold in thresholds:
        pred = apply_multilabel_threshold(
            proba,
            threshold,
            min_positive_labels=min_positive_labels
        )
        scores = score_multilabel_predictions(y_true, pred, proba, top_k=MAX_TOP_K)
        candidate = {
            'threshold': float(threshold),
            **scores
        }
        ranking = (
            candidate['macro_f1'],
            candidate['micro_f1'],
            candidate['recall_at_3'],
            candidate['precision_at_3'],
            -candidate['threshold']
        )
        if best is None:
            best = candidate
            continue
        best_ranking = (
            best['macro_f1'],
            best['micro_f1'],
            best['recall_at_3'],
            best['precision_at_3'],
            -best['threshold']
        )
        if ranking > best_ranking:
            best = candidate

    return best


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
