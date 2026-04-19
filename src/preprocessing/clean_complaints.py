import argparse
import re
import sys

import pandas as pd

from src.config import settings
from src.config.contracts import (
    BENCHMARK_SPLIT_MODE,
    CLEANED_COMPLAINTS_STEM,
    CLEANING_DRIFT_NAME,
    CLEANING_SUMMARY_NAME,
    COMBINED_COMPLAINTS_STEM,
    COMPONENT_CONFLICT_NAME,
    COMPONENT_MULTILABEL_CASES_STEM,
    COMPONENT_SINGLE_LABEL_CASES_STEM,
    COMPONENT_SUMMARY_NAME,
    COMPONENT_TARGET_GROUP_NAME,
    COMPONENT_TARGET_SCOPE_NAME,
    COMPONENT_TEXT_CONFLICT_NAME,
    COMPONENT_TEXT_OVERLAP_NAME,
    COMPONENT_TEXT_SIDECAR_STEM,
    FEATURE_WAVE1_SPLIT_MODE,
    REFERENCE_MODEL_YEAR_MAX,
    SEVERITY_CASES_STEM,
    get_split_policy,
)
from src.config.paths import OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_project_directories
from src.data.io_utils import load_frame, write_dataframe

# -----------------------------------------------------------------------------
# Output names
# -----------------------------------------------------------------------------
INPUT_STEM = COMBINED_COMPLAINTS_STEM
CLEAN_STEM = CLEANED_COMPLAINTS_STEM
SEVERITY_STEM = SEVERITY_CASES_STEM
SINGLE_CASE_STEM = COMPONENT_SINGLE_LABEL_CASES_STEM
MULTI_CASE_STEM = COMPONENT_MULTILABEL_CASES_STEM
SIDECAR_STEM = COMPONENT_TEXT_SIDECAR_STEM
DRIFT_NAME = CLEANING_DRIFT_NAME
CONFLICT_NAME = COMPONENT_CONFLICT_NAME
TARGET_SCOPE_NAME = COMPONENT_TARGET_SCOPE_NAME
TARGET_GROUP_NAME = COMPONENT_TARGET_GROUP_NAME
TEXT_CONFLICT_NAME = COMPONENT_TEXT_CONFLICT_NAME
OVERLAP_NAME = COMPONENT_TEXT_OVERLAP_NAME


# -----------------------------------------------------------------------------
# Cleaning constants
# -----------------------------------------------------------------------------
VEHICLE_TYPE = 'V'
SCHEMA_CHANGE_DATE = pd.Timestamp('2021-05-17')
VALID_PROD_TYPES = {'C', 'E', 'T', 'V'}

COMPONENT_DROP_CHILD_PARENTS = {
    'CHEST CLIP, BUCKLE, HARNESS',
    'CARRY HANDLE, SHELL, BASE',
    'CHILD SEAT',
    'TETHER, LOWER ANCHOR (ON CAR SEAT OR VEHICLE)',
    'I SUSPECT THE CAR SEAT IS COUNTERFEIT',
    'INSERT, PADDING'
}

COMPONENT_UNKNOWN_LABELS = {'UNKNOWN OR OTHER', 'OTHER/I AM NOT SURE'}

COMPONENT_FUEL_PARENTS = {
    'FUEL/PROPULSION SYSTEM',
    'FUEL SYSTEM, GASOLINE',
    'FUEL SYSTEM, DIESEL',
    'FUEL SYSTEM, OTHER',
    'HYBRID PROPULSION SYSTEM'
}

COMPONENT_SERVICE_BRAKE_PARENTS = {
    'SERVICE BRAKES',
    'SERVICE BRAKES, HYDRAULIC',
    'SERVICE BRAKES, AIR',
    'SERVICE BRAKES, ELECTRIC'
}

POSTAL_CODES = {
    'AA', 'AE', 'AK', 'AL', 'AP', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL',
    'FM', 'GA', 'GU', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
    'MH', 'MI', 'MN', 'MO', 'MP', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV',
    'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'PW', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA',
    'VI', 'VT', 'WA', 'WI', 'WV', 'WY'
}

REQ_COLS = [
    'cmplid',
    'odino',
    'mfr_name',
    'maketxt',
    'modeltxt',
    'yeartxt',
    'crash',
    'faildate',
    'fire',
    'injured',
    'deaths',
    'compdesc',
    'city',
    'state',
    'vin',
    'datea',
    'ldate',
    'miles',
    'cdescr',
    'veh_speed',
    'dealer_state',
    'prod_type',
    'medical_attn',
    'vehicles_towed_yn'
]

DATE_COLS = [
    'faildate',
    'datea',
    'ldate',
    'purch_dt',
    'manuf_dt'
]

INT_COLS = [
    'yeartxt',
    'injured',
    'deaths',
    'miles',
    'occurences',
    'num_cyls',
    'veh_speed'
]

UPPER_COLS = [
    'mfr_name',
    'maketxt',
    'modeltxt',
    'compdesc',
    'city',
    'state',
    'vin',
    'cmpl_type',
    'drive_train',
    'fuel_sys',
    'fuel_type',
    'trans_type',
    'dot',
    'tire_size',
    'loc_of_tire',
    'tire_fail_type',
    'seat_type',
    'restraint_type',
    'dealer_name',
    'dealer_city',
    'dealer_state',
    'dealer_zip',
    'prod_type'
]

YN_COLS = [
    'crash',
    'fire',
    'police_rpt_yn',
    'orig_owner_yn',
    'anti_brakes_yn',
    'cruise_cont_yn',
    'orig_equip_yn',
    'repaired_yn',
    'medical_attn',
    'vehicles_towed_yn'
]

DRIFT_YN_COLS = [
    'crash',
    'fire',
    'medical_attn',
    'vehicles_towed_yn',
    'police_rpt_yn',
    'orig_owner_yn',
    'anti_brakes_yn',
    'cruise_cont_yn',
    'repaired_yn'
]

DRIFT_NUM_COLS = [
    'miles',
    'veh_speed',
    'injured',
    'deaths',
    'lag_days_safe'
]

SEVERITY_FIRST_COLS = [
    'source_era',
    'source_zip',
    'source_file',
    'mfr_name',
    'maketxt',
    'modeltxt',
    'yeartxt',
    'city',
    'state',
    'vin',
    'ldate',
    'faildate',
    'datea',
    'cdescr',
    'cmpl_type',
    'police_rpt_yn',
    'orig_owner_yn',
    'anti_brakes_yn',
    'cruise_cont_yn',
    'num_cyls',
    'drive_train',
    'fuel_sys',
    'fuel_type',
    'trans_type',
    'dealer_city',
    'dealer_state',
    'dealer_zip',
    'repaired_yn'
]

SEVERITY_MAX_COLS = [
    'injured',
    'deaths',
    'miles',
    'veh_speed'
]

SEVERITY_YN_COLS = [
    'fire',
    'crash',
    'medical_attn',
    'vehicles_towed_yn'
]

SEVERITY_ANY_FLAGS = [
    'flag_prod_type_bad',
    'flag_year_unknown',
    'flag_year_out_of_range',
    'flag_speed_999',
    'flag_speed_high',
    'flag_miles_high',
    'flag_injured_99',
    'flag_deaths_99',
    'flag_state_bad',
    'flag_dealer_state_bad',
    'flag_vin_len_bad',
    'flag_fail_after_added',
    'flag_fail_after_received',
    'flag_added_before_received',
    'flag_date_order_bad',
    'flag_fail_old_new_vehicle',
    'flag_fail_pre_model',
    'flag_fail_pre_model_far',
    'miles_zero_flag',
    'veh_speed_zero_flag'
]

COMPONENT_COLS = [
    'cmplid',
    'odino',
    'source_era',
    'source_zip',
    'source_file',
    'mfr_name',
    'maketxt',
    'modeltxt',
    'yeartxt',
    'component_raw',
    'component_raw_std',
    'component_parent',
    'component_group',
    'component_group_rows',
    'component_drop_reason',
    'component_keep_flag',
    'state',
    'ldate',
    'faildate',
    'datea',
    'lag_days_safe',
    'miles',
    'veh_speed',
    'cmpl_type',
    'police_rpt_yn',
    'orig_owner_yn',
    'anti_brakes_yn',
    'cruise_cont_yn',
    'repaired_yn',
    'drive_train',
    'fuel_sys',
    'fuel_type',
    'trans_type',
    'num_cyls',
    'fire',
    'crash',
    'injured',
    'deaths',
    'medical_attn',
    'vehicles_towed_yn',
    'miles_missing_flag',
    'veh_speed_missing_flag',
    'miles_zero_flag',
    'veh_speed_zero_flag',
    'faildate_trusted_flag',
    'faildate_untrusted_flag',
    'severity_primary_row_flag',
    'severity_broad_row_flag',
    'flag_year_unknown',
    'flag_year_out_of_range',
    'flag_speed_999',
    'flag_speed_high',
    'flag_miles_high',
    'flag_state_bad',
    'flag_dealer_state_bad',
    'flag_vin_len_bad',
    'flag_date_order_bad',
    'flag_fail_pre_model',
    'flag_fail_pre_model_far'
]

AUDIT_VALUE_COLS = [
    'flag_prod_type_bad',
    'flag_year_unknown',
    'flag_year_out_of_range',
    'flag_speed_999',
    'flag_speed_high',
    'flag_miles_high',
    'flag_injured_99',
    'flag_deaths_99',
    'flag_state_bad',
    'flag_dealer_state_bad',
    'flag_vin_len_bad',
    'flag_fail_after_added',
    'flag_fail_after_received',
    'flag_added_before_received',
    'flag_date_order_bad',
    'flag_fail_old_new_vehicle',
    'flag_fail_pre_model',
    'flag_fail_pre_model_far',
    'miles_zero_flag',
    'veh_speed_zero_flag',
    'miles_missing_flag',
    'veh_speed_missing_flag',
    'faildate_trusted_flag',
    'faildate_untrusted_flag',
    'lag_days_safe',
    'severity_primary_row_flag',
    'severity_broad_row_flag'
]

AUDIT_KEY_COLS = [
    'cmplid',
    'odino',
    'source_era'
]


# -----------------------------------------------------------------------------
# Collapse rules
# -----------------------------------------------------------------------------
SINGLE_LABEL_MIN_CASES = 250

COLLAPSE_FIRST_COLS = [
    'source_era',
    'source_zip',
    'source_file',
    'mfr_name',
    'maketxt',
    'modeltxt',
    'yeartxt',
    'state',
    'ldate',
    'faildate',
    'cmpl_type',
    'drive_train',
    'fuel_sys',
    'fuel_type',
    'trans_type',
    'num_cyls',
    'police_rpt_yn',
    'orig_owner_yn',
    'anti_brakes_yn',
    'cruise_cont_yn',
    'repaired_yn'
]

COLLAPSE_MAX_COLS = [
    'component_group_rows',
    'injured',
    'deaths',
    'miles',
    'veh_speed'
]

COLLAPSE_YN_COLS = [
    'fire',
    'crash',
    'medical_attn',
    'vehicles_towed_yn',
    'police_rpt_yn',
    'orig_owner_yn',
    'anti_brakes_yn',
    'cruise_cont_yn',
    'repaired_yn'
]

COLLAPSE_ANY_FLAG_COLS = [
    'flag_year_unknown',
    'flag_year_out_of_range',
    'flag_speed_999',
    'flag_speed_high',
    'flag_miles_high',
    'flag_state_bad',
    'flag_date_order_bad',
    'flag_fail_pre_model',
    'flag_fail_pre_model_far',
    'miles_zero_flag',
    'veh_speed_zero_flag'
]

BASE_CASE_COLS = [
    'odino',
    'source_era',
    'source_zip',
    'source_file',
    'component_row_count',
    'component_group_rows',
    'mfr_name',
    'maketxt',
    'modeltxt',
    'yeartxt',
    'state',
    'ldate',
    'faildate',
    'lag_days_safe',
    'cmpl_type',
    'drive_train',
    'fuel_sys',
    'fuel_type',
    'trans_type',
    'miles',
    'veh_speed',
    'injured',
    'fire',
    'crash',
    'medical_attn',
    'vehicles_towed_yn',
    'police_rpt_yn',
    'repaired_yn',
    'miles_missing_flag',
    'veh_speed_missing_flag',
    'miles_zero_flag',
    'veh_speed_zero_flag',
    'faildate_trusted_flag',
    'severity_primary_flag',
    'severity_broad_flag'
]

SINGLE_CASE_COLS = BASE_CASE_COLS + [
    'component_group',
    'component_group_fit_case_count',
    'single_label_keep_flag'
]

MULTI_CASE_COLS = BASE_CASE_COLS + [
    'component_groups',
    'component_group_count'
]


# -----------------------------------------------------------------------------
# Text rules
# -----------------------------------------------------------------------------
SPACE_RE = re.compile(r'\s+')
ALPHA_RE = re.compile(r'[A-Z]')
PLACEHOLDER_TEXTS = {
    'N/A',
    'NA',
    'NONE',
    'NO DESCRIPTION',
    'NO DESCRIPTION PROVIDED',
    'UNKNOWN',
    'UNK',
    'TEST',
    'TEST TEST'
}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def require_columns(df):
    missing = [column for column in REQ_COLS if column not in df.columns]
    if missing:
        missing_text = ', '.join(missing)
        raise ValueError(f'Missing required complaint columns: {missing_text}')


def filter_columns(col_list, df):
    return [column for column in col_list if column in df.columns]


def add_safe_lag_fields(df):
    fail_year = df['faildate'].dt.year.astype('Int64')
    year_ok = df['yeartxt'].isna() | (fail_year >= df['yeartxt'] - 1)
    trusted = (
        df['faildate'].notna()
        & df['ldate'].notna()
        & (df['faildate'] <= df['ldate'])
        & year_ok.fillna(False)
    )

    df['faildate_trusted_flag'] = trusted
    df['faildate_untrusted_flag'] = ~trusted
    df['lag_days_safe'] = (df['ldate'] - df['faildate']).dt.days.where(trusted).astype('Int64')
    return df


def add_severity_flags(df, primary_name, broad_name):
    fire_yes = df['fire'].fillna('N').eq('Y')
    crash_yes = df['crash'].fillna('N').eq('Y')
    medical_yes = df['medical_attn'].fillna('N').eq('Y')
    towed_yes = df['vehicles_towed_yn'].fillna('N').eq('Y')
    primary = (
        df['deaths'].fillna(0).gt(0)
        | df['injured'].fillna(0).gt(0)
        | fire_yes
        | crash_yes
    )
    broad = primary | medical_yes | towed_yes

    df[primary_name] = primary
    df[broad_name] = broad
    return df


def reconstruct_yn_cols(df, yn_cols):
    for column in yn_cols:
        yes_col = f'__{column}_yes'
        present_col = f'__{column}_present'
        df[column] = pd.NA
        df.loc[df[present_col], column] = 'N'
        df.loc[df[yes_col], column] = 'Y'
        df = df.drop(columns=[yes_col, present_col])
    return df


def normalize_component_label(series):
    return (
        series.astype('string')
        .str.upper()
        .str.replace(r'\s*;\s*', ':', regex=True)
        .str.replace(r'\s*:\s*', ':', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )


def map_component_group(label):
    if pd.isna(label):
        return pd.NA

    parent = str(label).split(':', 1)[0].strip()

    if label in COMPONENT_UNKNOWN_LABELS:
        return 'UNKNOWN OR OTHER'
    if parent in {'COMMUNICATION', 'COMMUNICATIONS'}:
        return 'COMMUNICATION'
    if parent in {'ENGINE', 'ENGINE AND ENGINE COOLING'}:
        return 'ENGINE / COOLING'
    if parent in {'VISIBILITY', 'VISIBILITY/WIPER'}:
        return 'VISIBILITY / WIPER'
    if parent in COMPONENT_FUEL_PARENTS:
        return 'FUEL / PROPULSION'
    if parent in COMPONENT_SERVICE_BRAKE_PARENTS:
        return 'SERVICE BRAKES'
    return parent


def add_source_era(df):
    source_era = pd.Series(pd.NA, index=df.index, dtype='string')
    source_era.loc[df['ldate'].notna() & df['ldate'].lt(SCHEMA_CHANGE_DATE)] = 'pre_2021_schema_change'
    source_era.loc[df['ldate'].notna() & df['ldate'].ge(SCHEMA_CHANGE_DATE)] = 'post_2021_schema_change'
    df['source_era'] = source_era
    return df


def apply_modeling_zero_rules(df):
    df['miles_zero_flag'] = df['miles'].eq(0)
    df['veh_speed_zero_flag'] = df['veh_speed'].eq(0)
    df.loc[df['miles_zero_flag'], 'miles'] = pd.NA
    df.loc[df['veh_speed_zero_flag'], 'veh_speed'] = pd.NA
    df['miles_missing_flag'] = df['miles'].isna()
    df['veh_speed_missing_flag'] = df['veh_speed'].isna()
    return df


def select_clean_columns(work):
    drop_cols = [column for column in AUDIT_VALUE_COLS if column in work.columns]
    return work.drop(columns=drop_cols).copy()


def build_cleaning_audit(work):
    keep_cols = filter_columns(AUDIT_KEY_COLS + AUDIT_VALUE_COLS, work)
    return work.loc[:, keep_cols].copy()


def merge_cleaning_audit(cleaned_df, audit_df):
    join_cols = [column for column in ['cmplid', 'odino'] if column in cleaned_df.columns and column in audit_df.columns]
    audit_keep_cols = [
        column
        for column in audit_df.columns
        if column not in join_cols and column not in cleaned_df.columns
    ]
    return cleaned_df.merge(
        audit_df[join_cols + audit_keep_cols],
        on=join_cols,
        how='left',
        validate='one_to_one'
    )


def build_source_era_drift(audit_df):
    era_values = ['overall'] + sorted(audit_df['source_era'].dropna().unique().tolist())
    rows = []

    for source_era in era_values:
        subset = audit_df if source_era == 'overall' else audit_df.loc[audit_df['source_era'].eq(source_era)]
        row_count = int(len(subset))
        if row_count == 0:
            continue

        for column in DRIFT_YN_COLS:
            if column not in subset.columns:
                continue
            series = subset[column]
            non_null = int(series.notna().sum())
            yes_count = int(series.eq('Y').sum())
            no_count = int(series.eq('N').sum())
            rows.extend(
                [
                    {
                        'source_era': source_era,
                        'field': column,
                        'field_type': 'yn',
                        'metric': 'row_count',
                        'value': row_count
                    },
                    {
                        'source_era': source_era,
                        'field': column,
                        'field_type': 'yn',
                        'metric': 'non_null_count',
                        'value': non_null
                    },
                    {
                        'source_era': source_era,
                        'field': column,
                        'field_type': 'yn',
                        'metric': 'non_null_rate',
                        'value': round(non_null / row_count, 4)
                    },
                    {
                        'source_era': source_era,
                        'field': column,
                        'field_type': 'yn',
                        'metric': 'y_count',
                        'value': yes_count
                    },
                    {
                        'source_era': source_era,
                        'field': column,
                        'field_type': 'yn',
                        'metric': 'y_rate',
                        'value': round(yes_count / row_count, 4)
                    },
                    {
                        'source_era': source_era,
                        'field': column,
                        'field_type': 'yn',
                        'metric': 'n_count',
                        'value': no_count
                    },
                    {
                        'source_era': source_era,
                        'field': column,
                        'field_type': 'yn',
                        'metric': 'n_rate',
                        'value': round(no_count / row_count, 4)
                    }
                ]
            )

        for column in DRIFT_NUM_COLS:
            missing_flag_col = f'{column}_missing_flag'
            zero_flag_col = f'{column}_zero_flag'
            if column not in subset.columns and missing_flag_col not in subset.columns and zero_flag_col not in subset.columns:
                continue

            if column in subset.columns:
                series = pd.to_numeric(subset[column], errors='coerce')
                non_null = int(series.notna().sum())
                missing_rate = round(series.isna().mean(), 4)
                mean_value = round(float(series.mean()), 4) if non_null else pd.NA
                median_value = round(float(series.median()), 4) if non_null else pd.NA
            else:
                series = pd.Series([pd.NA] * row_count, dtype='Float64')
                missing_count = int(subset[missing_flag_col].fillna(False).sum()) if missing_flag_col in subset.columns else row_count
                non_null = row_count - missing_count
                missing_rate = round(missing_count / row_count, 4)
                mean_value = pd.NA
                median_value = pd.NA

            if zero_flag_col in subset.columns:
                zero_count = int(subset[zero_flag_col].fillna(False).sum())
            else:
                zero_count = int(series.eq(0).sum())
            rows.extend(
                [
                    {
                        'source_era': source_era,
                        'field': column,
                        'field_type': 'numeric',
                        'metric': 'row_count',
                        'value': row_count
                    },
                    {
                        'source_era': source_era,
                        'field': column,
                        'field_type': 'numeric',
                        'metric': 'non_null_count',
                        'value': non_null
                    },
                    {
                        'source_era': source_era,
                        'field': column,
                        'field_type': 'numeric',
                        'metric': 'missing_rate',
                        'value': missing_rate
                    },
                    {
                        'source_era': source_era,
                        'field': column,
                        'field_type': 'numeric',
                        'metric': 'zero_count',
                        'value': zero_count
                    },
                    {
                        'source_era': source_era,
                        'field': column,
                        'field_type': 'numeric',
                        'metric': 'zero_rate',
                        'value': round(zero_count / row_count, 4)
                    },
                    {
                        'source_era': source_era,
                        'field': column,
                        'field_type': 'numeric',
                        'metric': 'mean',
                        'value': mean_value
                    },
                    {
                        'source_era': source_era,
                        'field': column,
                        'field_type': 'numeric',
                        'metric': 'median',
                        'value': median_value
                    }
                ]
            )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Shared cleaning
# -----------------------------------------------------------------------------
def build_cleaning_work(df):
    require_columns(df)
    work = df.copy()

    text_cols = work.select_dtypes(include=['object', 'string']).columns.tolist()
    for column in text_cols:
        work[column] = work[column].astype('string').str.strip()
        work[column] = work[column].replace({'': pd.NA})

    for column in UPPER_COLS + YN_COLS:
        if column in work.columns:
            work[column] = work[column].astype('string').str.upper()

    for column in DATE_COLS:
        if column in work.columns:
            work[column] = pd.to_datetime(work[column], errors='coerce')

    for column in INT_COLS:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors='coerce').astype('Int64')

    work = add_source_era(work)

    fail_year = work['faildate'].dt.year.astype('Int64')
    model_year = work['yeartxt']

    work['flag_prod_type_bad'] = work['prod_type'].notna() & ~work['prod_type'].isin(VALID_PROD_TYPES)
    work['flag_year_unknown'] = model_year.eq(9999)
    work['flag_year_out_of_range'] = model_year.notna() & ~model_year.between(1900, REFERENCE_MODEL_YEAR_MAX)
    valid_model_year = model_year.where(~(work['flag_year_unknown'] | work['flag_year_out_of_range']))
    work['flag_speed_999'] = work['veh_speed'].eq(999)
    work['flag_speed_high'] = work['veh_speed'].notna() & work['veh_speed'].gt(200)
    work['flag_miles_high'] = work['miles'].notna() & work['miles'].gt(500000)
    work['flag_injured_99'] = work['injured'].eq(99)
    work['flag_deaths_99'] = work['deaths'].eq(99)
    work['flag_state_bad'] = work['state'].notna() & ~work['state'].isin(POSTAL_CODES)
    work['flag_dealer_state_bad'] = work['dealer_state'].notna() & ~work['dealer_state'].isin(POSTAL_CODES)
    work['flag_vin_len_bad'] = work['vin'].notna() & work['vin'].str.len().ne(11)
    work['flag_fail_after_added'] = work['faildate'].notna() & work['datea'].notna() & (work['faildate'] > work['datea'])
    work['flag_fail_after_received'] = work['faildate'].notna() & work['ldate'].notna() & (work['faildate'] > work['ldate'])
    work['flag_added_before_received'] = work['datea'].notna() & work['ldate'].notna() & (work['datea'] < work['ldate'])
    work['flag_date_order_bad'] = (
        work['flag_fail_after_added']
        | work['flag_fail_after_received']
        | work['flag_added_before_received']
    )
    work['flag_fail_old_new_vehicle'] = fail_year.notna() & valid_model_year.notna() & valid_model_year.ge(1990) & fail_year.lt(1990)
    work['flag_fail_pre_model'] = fail_year.notna() & valid_model_year.notna() & fail_year.lt(valid_model_year - 1)
    work['flag_fail_pre_model_far'] = fail_year.notna() & valid_model_year.notna() & fail_year.lt(valid_model_year - 5)

    work.loc[work['flag_prod_type_bad'], 'prod_type'] = pd.NA
    work.loc[work['flag_year_unknown'] | work['flag_year_out_of_range'], 'yeartxt'] = pd.NA
    work.loc[work['flag_speed_999'], 'veh_speed'] = pd.NA
    work.loc[work['flag_injured_99'], 'injured'] = pd.NA
    work.loc[work['flag_deaths_99'], 'deaths'] = pd.NA
    work.loc[work['flag_state_bad'], 'state'] = pd.NA
    work.loc[work['flag_dealer_state_bad'], 'dealer_state'] = pd.NA

    work = apply_modeling_zero_rules(work)
    work = add_safe_lag_fields(work)
    return add_severity_flags(
        work,
        primary_name='severity_primary_row_flag',
        broad_name='severity_broad_row_flag'
    )
# -----------------------------------------------------------------------------
# Task tables
# -----------------------------------------------------------------------------
def build_severity_cases(cleaned_df, audit_df):
    vehicle_df = merge_cleaning_audit(cleaned_df, audit_df)
    vehicle_df = vehicle_df.loc[vehicle_df['prod_type'].eq(VEHICLE_TYPE) & vehicle_df['odino'].notna()].copy()
    if vehicle_df.empty:
        raise ValueError('No vehicle complaint rows found for the severity table')

    vehicle_df = vehicle_df.sort_values(['odino', 'cmplid'], na_position='last')
    grouped = vehicle_df.groupby('odino', sort=True)

    frames = [
        grouped.size().rename('row_count'),
        grouped['compdesc'].nunique(dropna=True).rename('component_count')
    ]

    first_cols = filter_columns(SEVERITY_FIRST_COLS, vehicle_df)
    if first_cols:
        frames.append(grouped[first_cols].first())

    max_cols = filter_columns(SEVERITY_MAX_COLS, vehicle_df)
    if max_cols:
        frames.append(grouped[max_cols].max())

    any_flag_cols = filter_columns(SEVERITY_ANY_FLAGS, vehicle_df)
    if any_flag_cols:
        frames.append(grouped[any_flag_cols].any())

    yn_cols = [column for column in SEVERITY_YN_COLS if column in vehicle_df.columns]
    temp_cols = []
    for column in yn_cols:
        yes_col = f'__{column}_yes'
        present_col = f'__{column}_present'
        vehicle_df[yes_col] = vehicle_df[column].fillna('N').eq('Y')
        vehicle_df[present_col] = vehicle_df[column].notna()
        temp_cols.extend([yes_col, present_col])

    if temp_cols:
        frames.append(vehicle_df.groupby('odino', sort=True)[temp_cols].max())

    case_df = pd.concat(frames, axis=1).reset_index()
    case_df = reconstruct_yn_cols(case_df, yn_cols)
    case_df['miles_missing_flag'] = case_df['miles'].isna()
    case_df['veh_speed_missing_flag'] = case_df['veh_speed'].isna()
    case_df = add_safe_lag_fields(case_df)
    return add_severity_flags(
        case_df,
        primary_name='severity_primary_flag',
        broad_name='severity_broad_flag'
    )


def build_component_rows(cleaned_df, audit_df):
    component_df = merge_cleaning_audit(cleaned_df, audit_df)
    component_df = component_df.loc[
        component_df['prod_type'].eq(VEHICLE_TYPE) & component_df['compdesc'].notna()
    ].copy()
    if component_df.empty:
        raise ValueError('No vehicle component rows found for the component table')

    component_df = component_df.rename(columns={'compdesc': 'component_raw'})
    component_df['component_raw_std'] = normalize_component_label(component_df['component_raw'])
    component_df['component_parent'] = (
        component_df['component_raw_std']
        .str.split(':')
        .str[0]
        .str.strip()
    )
    component_df['component_group'] = component_df['component_raw_std'].map(map_component_group)
    component_df['component_drop_reason'] = pd.NA

    component_df.loc[
        component_df['component_parent'].isin(COMPONENT_DROP_CHILD_PARENTS),
        'component_drop_reason'
    ] = 'DROP_CHILD_RESTRAINT'
    component_df.loc[
        component_df['component_group'].isin({'EQUIPMENT', 'EQUIPMENT ADAPTIVE/MOBILITY'}),
        'component_drop_reason'
    ] = 'DROP_EQUIPMENT'
    component_df.loc[
        component_df['component_group'].eq('UNKNOWN OR OTHER'),
        'component_drop_reason'
    ] = 'DROP_UNKNOWN_OTHER'

    component_group_rows = (
        component_df.loc[component_df['component_drop_reason'].isna(), 'component_group']
        .value_counts()
    )
    component_df['component_group_rows'] = (
        component_df['component_group']
        .map(component_group_rows)
        .astype('Int64')
    )
    component_df['component_keep_flag'] = component_df['component_drop_reason'].isna()

    keep_cols = [column for column in COMPONENT_COLS if column in component_df.columns]
    return component_df.loc[:, keep_cols].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Collapse helpers
# -----------------------------------------------------------------------------
def collapse_case_features(case_rows, target_mode):
    case_rows = case_rows.sort_values(['odino', 'cmplid'], na_position='last').copy()
    grouped = case_rows.groupby('odino', sort=True)

    frames = [grouped.size().rename('component_row_count')]
    if target_mode == 'single':
        frames.append(grouped['component_group'].first())
    elif target_mode == 'multi':
        frames.append(
            grouped['component_group']
            .agg(lambda s: '|'.join(sorted(pd.Series(s.dropna().astype(str).unique()).tolist())))
            .rename('component_groups')
        )
        frames.append(grouped['component_group'].nunique(dropna=True).rename('component_group_count'))
    elif target_mode != 'base':
        raise ValueError(f'Unknown target_mode: {target_mode}')

    first_cols = filter_columns(COLLAPSE_FIRST_COLS, case_rows)
    if first_cols:
        frames.append(grouped[first_cols].first())

    max_cols = filter_columns(COLLAPSE_MAX_COLS, case_rows)
    if max_cols:
        frames.append(grouped[max_cols].max())

    any_flag_cols = filter_columns(COLLAPSE_ANY_FLAG_COLS, case_rows)
    if any_flag_cols:
        frames.append(grouped[any_flag_cols].any())

    yn_cols = [column for column in COLLAPSE_YN_COLS if column in case_rows.columns]
    temp_cols = []
    for column in yn_cols:
        yes_col = f'__{column}_yes'
        present_col = f'__{column}_present'
        case_rows[yes_col] = case_rows[column].fillna('N').eq('Y')
        case_rows[present_col] = case_rows[column].notna()
        temp_cols.extend([yes_col, present_col])

    if temp_cols:
        frames.append(case_rows.groupby('odino', sort=True)[temp_cols].max())

    case_df = pd.concat(frames, axis=1).reset_index()
    case_df = reconstruct_yn_cols(case_df, yn_cols)
    case_df['miles_missing_flag'] = case_df['miles'].isna()
    case_df['veh_speed_missing_flag'] = case_df['veh_speed'].isna()
    case_df = add_safe_lag_fields(case_df)
    return add_severity_flags(
        case_df,
        primary_name='severity_primary_flag',
        broad_name='severity_broad_flag'
    )
def build_case_tables(component_df):
    keep_df = component_df.loc[
        component_df['component_keep_flag'].fillna(False) & component_df['odino'].notna()
    ].copy()
    if keep_df.empty:
        raise ValueError('No kept component rows found for case collapse')

    group_counts = keep_df.groupby('odino')['component_group'].nunique()
    single_ids = group_counts.loc[group_counts.eq(1)].index
    multi_ids = group_counts.loc[group_counts.gt(1)].index

    single_rows = keep_df.loc[keep_df['odino'].isin(single_ids)].copy()
    multi_rows = keep_df.loc[keep_df['odino'].isin(multi_ids)].copy()

    if single_rows.empty:
        raise ValueError('No single-label component cases found after filtering')

    base_case_df = collapse_case_features(keep_df, target_mode='base')
    base_keep_cols = filter_columns(BASE_CASE_COLS, base_case_df)
    base_case_df = base_case_df.loc[:, base_keep_cols].sort_values('odino').reset_index(drop=True)

    single_target_df = collapse_case_features(single_rows, target_mode='single')[['odino', 'component_group']]
    single_case_df = base_case_df.loc[base_case_df['odino'].isin(single_ids)].merge(
        single_target_df,
        on='odino',
        how='left',
        validate='one_to_one'
    )
    benchmark_policy = get_split_policy(BENCHMARK_SPLIT_MODE)
    fit_window_mask = pd.to_datetime(single_case_df['ldate'], errors='coerce').le(benchmark_policy['valid_end'])
    fit_counts = single_case_df.loc[fit_window_mask, 'component_group'].value_counts()
    single_case_df['component_group_fit_case_count'] = (
        single_case_df['component_group']
        .map(fit_counts)
        .astype('Int64')
    )
    single_case_df['single_label_keep_flag'] = (
        single_case_df['component_group_fit_case_count']
        .fillna(0)
        .ge(SINGLE_LABEL_MIN_CASES)
    )
    single_case_bench_df = single_case_df.loc[single_case_df['single_label_keep_flag']].copy()
    single_keep_cols = filter_columns(SINGLE_CASE_COLS, single_case_df)
    single_case_df = single_case_df.loc[:, single_keep_cols].sort_values('odino').reset_index(drop=True)
    single_case_bench_df = single_case_bench_df.loc[:, single_keep_cols].sort_values('odino').reset_index(drop=True)

    multi_target_df = collapse_case_features(keep_df, target_mode='multi')[['odino', 'component_groups', 'component_group_count']]
    multi_case_df = base_case_df.merge(
        multi_target_df,
        on='odino',
        how='left',
        validate='one_to_one'
    )
    multi_keep_cols = filter_columns(MULTI_CASE_COLS, multi_case_df)
    multi_case_df = multi_case_df.loc[:, multi_keep_cols].sort_values('odino').reset_index(drop=True)

    return keep_df, single_rows, multi_rows, base_case_df, single_case_df, single_case_bench_df, multi_case_df


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def normalize_text(value):
    if pd.isna(value):
        return ''
    return SPACE_RE.sub(' ', str(value).strip())


def is_placeholder_text(text):
    text = normalize_text(text)
    if not text:
        return False

    upper = text.upper()
    if upper in PLACEHOLDER_TEXTS:
        return True

    return len(text) <= 10 and not bool(ALPHA_RE.search(upper))


def build_base_text_rows(clean_df, odino_universe):
    work = clean_df.loc[
        clean_df['odino'].isin(odino_universe),
        ['odino', 'cmplid', 'cdescr', 'source_era', 'ldate']
    ].copy()
    work['ldate'] = pd.to_datetime(work['ldate'], errors='coerce')
    work['cmplid_num'] = pd.to_numeric(work['cmplid'], errors='coerce')
    work['cdescr_norm'] = work['cdescr'].map(normalize_text)
    work['cdescr_len'] = work['cdescr_norm'].str.len()
    work['has_text'] = work['cdescr_norm'].ne('')
    return work


def select_best_text_rows(clean_df, odino_universe):
    base_df = build_base_text_rows(clean_df, odino_universe)
    universe_df = pd.DataFrame({'odino': sorted(pd.Series(odino_universe, dtype='string').dropna().astype(str).unique())})

    fallback_df = (
        base_df
        .sort_values(['odino', 'ldate', 'cmplid_num'], ascending=[True, True, True], na_position='last')
        .groupby('odino', as_index=False)
        .first()
    )
    nonblank_df = (
        base_df.loc[base_df['has_text']].copy()
        .sort_values(
            ['odino', 'cdescr_len', 'ldate', 'cmplid_num'],
            ascending=[True, False, True, True],
            na_position='last'
        )
        .groupby('odino', as_index=False)
        .first()
    )

    chosen_df = universe_df.merge(fallback_df, on='odino', how='left', suffixes=('', '_fallback'))
    chosen_nonblank_df = nonblank_df.set_index('odino')
    fallback_lookup = fallback_df.set_index('odino')

    for column in ['cmplid', 'cmplid_num', 'cdescr', 'source_era', 'ldate', 'cdescr_norm', 'cdescr_len', 'has_text']:
        chosen_df[column] = chosen_df['odino'].map(chosen_nonblank_df[column]) if column in chosen_nonblank_df.columns else pd.NA
        if column in fallback_lookup.columns:
            chosen_df[column] = chosen_df[column].where(chosen_df[column].notna(), chosen_df['odino'].map(fallback_lookup[column]))

    chosen_df['cdescr'] = chosen_df['cdescr'].astype('string')
    chosen_df['cdescr_norm'] = chosen_df['cdescr_norm'].fillna('')
    chosen_df['cdescr_missing_flag'] = chosen_df['cdescr_norm'].eq('')
    chosen_df['cdescr_placeholder_flag'] = chosen_df['cdescr_norm'].map(is_placeholder_text)
    chosen_df.loc[chosen_df['cdescr_missing_flag'], 'cdescr_placeholder_flag'] = False
    chosen_df['cdescr_model_text'] = chosen_df['cdescr_norm']
    chosen_df.loc[chosen_df['cdescr_placeholder_flag'], 'cdescr_model_text'] = ''
    chosen_df['cdescr_char_len'] = chosen_df['cdescr_model_text'].str.len().astype('Int64')
    chosen_df['cdescr_word_count'] = (
        chosen_df['cdescr_model_text']
        .str.split()
        .map(lambda parts: len(parts) if isinstance(parts, list) else 0)
        .astype('Int64')
    )

    keep_cols = [
        'odino',
        'cdescr',
        'cdescr_model_text',
        'cdescr_missing_flag',
        'cdescr_placeholder_flag',
        'cdescr_char_len',
        'cdescr_word_count',
        'source_era',
        'ldate'
    ]
    return chosen_df.loc[:, keep_cols].sort_values('odino').reset_index(drop=True), base_df


# -----------------------------------------------------------------------------
# Summary output
# -----------------------------------------------------------------------------
def build_summary(cleaned_df, audit_df, severity_df, component_df):
    vehicle_df = cleaned_df.loc[cleaned_df['prod_type'].eq(VEHICLE_TYPE)]
    kept_component_df = component_df.loc[component_df['component_keep_flag']]

    summary_rows = [
        {
            'metric': 'cleaned_rows',
            'value': int(len(cleaned_df))
        },
        {
            'metric': 'vehicle_rows',
            'value': int(len(vehicle_df))
        },
        {
            'metric': 'severity_case_rows',
            'value': int(len(severity_df))
        },
        {
            'metric': 'component_rows',
            'value': int(len(component_df))
        },
        {
            'metric': 'component_model_rows',
            'value': int(len(kept_component_df))
        },
        {
            'metric': 'component_model_groups',
            'value': int(kept_component_df['component_group'].nunique())
        },
        {
            'metric': 'severity_primary_cases',
            'value': int(severity_df['severity_primary_flag'].sum())
        },
        {
            'metric': 'severity_broad_cases',
            'value': int(severity_df['severity_broad_flag'].sum())
        },
        {
            'metric': 'severity_primary_case_rate_pct',
            'value': round(float(severity_df['severity_primary_flag'].mean() * 100), 2)
        },
        {
            'metric': 'severity_broad_case_rate_pct',
            'value': round(float(severity_df['severity_broad_flag'].mean() * 100), 2)
        }
    ]

    for drop_reason in ['DROP_CHILD_RESTRAINT', 'DROP_EQUIPMENT', 'DROP_UNKNOWN_OTHER']:
        summary_rows.append(
            {
                'metric': f'component_{drop_reason.lower()}',
                'value': int(component_df['component_drop_reason'].eq(drop_reason).sum())
            }
        )

    for source_era, era_count in cleaned_df['source_era'].fillna('unknown').value_counts().sort_index().items():
        summary_rows.append(
            {
                'metric': f'source_era_{source_era}',
                'value': int(era_count)
            }
        )

    flag_cols = [
        'flag_year_unknown',
        'flag_year_out_of_range',
        'flag_speed_999',
        'flag_speed_high',
        'flag_miles_high',
        'flag_state_bad',
        'flag_dealer_state_bad',
        'flag_vin_len_bad',
        'flag_date_order_bad',
        'flag_fail_pre_model',
        'flag_fail_pre_model_far',
        'miles_zero_flag',
        'veh_speed_zero_flag',
        'faildate_untrusted_flag',
        'miles_missing_flag',
        'veh_speed_missing_flag'
    ]
    for column in flag_cols:
        summary_rows.append(
            {
                'metric': column,
                'value': int(audit_df[column].sum())
            }
        )

    return pd.DataFrame(summary_rows)

def build_conflict_summary(case_rows, scope_name):
    if case_rows.empty:
        return pd.DataFrame(columns=['scope', 'column', 'conflict_cases', 'conflict_rate_pct'])

    conflict_cols = [
        column
        for column in COLLAPSE_FIRST_COLS + COLLAPSE_MAX_COLS + COLLAPSE_YN_COLS
        if column in case_rows.columns
    ]
    distinct = case_rows.groupby('odino', sort=True)[conflict_cols].nunique(dropna=True)
    conflict_df = distinct.gt(1).sum().reset_index(name='conflict_cases')
    conflict_df = conflict_df.rename(columns={'index': 'column'})
    conflict_df['scope'] = scope_name
    conflict_df['conflict_rate_pct'] = (
        conflict_df['conflict_cases'] / max(case_rows['odino'].nunique(), 1) * 100
    ).round(4)
    return conflict_df[['scope', 'column', 'conflict_cases', 'conflict_rate_pct']].sort_values(
        ['conflict_cases', 'column'],
        ascending=[False, True]
    ).reset_index(drop=True)


def build_collapse_summary(component_df, keep_df, base_case_df, single_case_df, single_case_bench_df, multi_case_df):
    all_cases = int(component_df['odino'].nunique())
    kept_cases = int(keep_df['odino'].nunique())
    base_cases = int(base_case_df['odino'].nunique())
    single_cases = int(single_case_df['odino'].nunique())
    benchmark_cases = int(single_case_bench_df['odino'].nunique())
    multi_benchmark_cases = int(multi_case_df['odino'].nunique())
    multi_only_cases = kept_cases - single_cases
    rare_single_cases = single_cases - benchmark_cases

    summary_rows = [
        {
            'metric': 'component_rows_in',
            'value': int(len(component_df))
        },
        {
            'metric': 'kept_component_rows',
            'value': int(len(keep_df))
        },
        {
            'metric': 'all_component_cases',
            'value': all_cases
        },
        {
            'metric': 'kept_component_cases',
            'value': kept_cases
        },
        {
            'metric': 'component_case_base_cases',
            'value': base_cases
        },
        {
            'metric': 'single_label_cases_all',
            'value': single_cases
        },
        {
            'metric': 'single_label_benchmark_cases',
            'value': benchmark_cases
        },
        {
            'metric': 'single_label_rare_group_cases_dropped',
            'value': rare_single_cases
        },
        {
            'metric': 'multi_label_benchmark_cases',
            'value': multi_benchmark_cases
        },
        {
            'metric': 'multi_label_only_cases',
            'value': multi_only_cases
        },
        {
            'metric': 'single_label_case_share_pct',
            'value': round(single_cases / max(kept_cases, 1) * 100, 2)
        },
        {
            'metric': 'single_label_benchmark_share_pct',
            'value': round(benchmark_cases / max(kept_cases, 1) * 100, 2)
        },
        {
            'metric': 'multi_label_case_share_pct',
            'value': round(multi_only_cases / max(kept_cases, 1) * 100, 2)
        },
        {
            'metric': 'component_model_groups',
            'value': int(single_case_bench_df['component_group'].nunique())
        }
    ]
    return pd.DataFrame(summary_rows)


def build_target_scope_summary(base_case_df, single_case_df, single_case_bench_df, multi_case_df):
    rows = []
    kept_case_base = base_case_df[['odino', 'ldate', 'severity_broad_flag']].copy()
    multi_only_df = multi_case_df.loc[multi_case_df['component_group_count'].gt(1)].copy()
    scope_frames = {
        'kept_component_cases': kept_case_base,
        'multi_label_benchmark_cases': multi_case_df,
        'single_label_cases_all': single_case_df,
        'single_label_benchmark_cases': single_case_bench_df,
        'multi_label_only_cases': multi_only_df
    }

    for scope_name, frame in scope_frames.items():
        if frame.empty:
            continue
        rows.append(
            {
                'scope': 'overall',
                'segment': scope_name,
                'cases': int(len(frame)),
                'case_share': round(float(len(frame) / max(len(scope_frames['kept_component_cases']), 1)), 4),
                'severity_broad_rate': round(float(frame['severity_broad_flag'].mean()), 4)
                if 'severity_broad_flag' in frame.columns
                else pd.NA
            }
        )

    keep_years = scope_frames['kept_component_cases'][['odino', 'ldate', 'severity_broad_flag']].copy()
    keep_years['year'] = pd.to_datetime(keep_years['ldate']).dt.year.astype('Int64')
    single_years = single_case_df[['odino', 'ldate', 'severity_broad_flag']].copy()
    single_years['year'] = pd.to_datetime(single_years['ldate']).dt.year.astype('Int64')
    multi_years = multi_only_df[['odino', 'ldate', 'severity_broad_flag']].copy()
    multi_years['year'] = pd.to_datetime(multi_years['ldate']).dt.year.astype('Int64')

    for year_value in sorted(keep_years['year'].dropna().unique().tolist()):
        keep_count = int(keep_years.loc[keep_years['year'].eq(year_value), 'odino'].nunique())
        if keep_count == 0:
            continue
        single_slice = single_years.loc[single_years['year'].eq(year_value)]
        multi_slice = multi_years.loc[multi_years['year'].eq(year_value)]
        rows.extend(
            [
                {
                    'scope': 'by_year',
                    'segment': f'single_label_share_{year_value}',
                    'cases': int(len(single_slice)),
                    'case_share': round(float(len(single_slice) / keep_count), 4),
                    'severity_broad_rate': round(float(single_slice['severity_broad_flag'].mean()), 4)
                    if not single_slice.empty
                    else pd.NA
                },
                {
                    'scope': 'by_year',
                    'segment': f'multi_label_only_share_{year_value}',
                    'cases': int(len(multi_slice)),
                    'case_share': round(float(len(multi_slice) / keep_count), 4),
                    'severity_broad_rate': round(float(multi_slice['severity_broad_flag'].mean()), 4)
                    if not multi_slice.empty
                    else pd.NA
                }
            ]
        )

    return pd.DataFrame(rows)


def build_target_group_summary(keep_df, single_case_df, single_case_bench_df, multi_rows):
    kept_presence = (
        keep_df[['odino', 'component_group']]
        .drop_duplicates()
        ['component_group']
        .value_counts()
    )
    multi_presence = (
        multi_rows[['odino', 'component_group']]
        .drop_duplicates()
        ['component_group']
        .value_counts()
    ) if not multi_rows.empty else pd.Series(dtype='int64')
    single_presence = single_case_df['component_group'].value_counts()
    benchmark_presence = single_case_bench_df['component_group'].value_counts()
    rows = []
    all_groups = sorted(kept_presence.index.tolist())
    for group_name in all_groups:
        kept_count = int(kept_presence.get(group_name, 0))
        single_count = int(single_presence.get(group_name, 0))
        benchmark_count = int(benchmark_presence.get(group_name, 0))
        multi_count = int(multi_presence.get(group_name, 0))
        rows.append(
            {
                'component_group': group_name,
                'kept_case_presence': kept_count,
                'single_label_case_presence': single_count,
                'single_label_benchmark_case_presence': benchmark_count,
                'multi_label_case_presence': multi_count,
                'multi_label_presence_share': round(multi_count / max(kept_count, 1), 4),
                'single_label_presence_share': round(single_count / max(kept_count, 1), 4),
                'single_label_benchmark_presence_share': round(benchmark_count / max(kept_count, 1), 4)
            }
        )

    return pd.DataFrame(rows).sort_values('kept_case_presence', ascending=False).reset_index(drop=True)

def build_conflict_report(base_df, sidecar_df):
    nonblank_df = base_df.loc[base_df['has_text']].copy()
    if nonblank_df.empty:
        return pd.DataFrame(
            columns=[
                'odino',
                'candidate_rows',
                'distinct_nonblank_texts',
                'chosen_char_len',
                'earliest_ldate',
                'latest_ldate',
                'chosen_cdescr'
            ]
        )

    conflict_df = (
        nonblank_df.groupby('odino', as_index=False)
        .agg(
            candidate_rows=('odino', 'size'),
            distinct_nonblank_texts=('cdescr_norm', 'nunique'),
            earliest_ldate=('ldate', 'min'),
            latest_ldate=('ldate', 'max')
        )
    )
    conflict_df = conflict_df.loc[conflict_df['distinct_nonblank_texts'].gt(1)].copy()
    if conflict_df.empty:
        return pd.DataFrame(columns=[
            'odino',
            'candidate_rows',
            'distinct_nonblank_texts',
            'chosen_char_len',
            'earliest_ldate',
            'latest_ldate',
            'chosen_cdescr'
        ])

    chosen_df = sidecar_df[['odino', 'cdescr_model_text']].copy()
    chosen_df = chosen_df.rename(columns={'cdescr_model_text': 'chosen_cdescr'})
    chosen_df['chosen_char_len'] = chosen_df['chosen_cdescr'].str.len().astype('Int64')
    conflict_df = conflict_df.merge(chosen_df, on='odino', how='left', validate='one_to_one')
    return conflict_df.sort_values(['distinct_nonblank_texts', 'candidate_rows', 'odino'], ascending=[False, False, True]).reset_index(drop=True)


def build_overlap_row(prior_name, later_name, prior_df, later_df):
    prior_text = prior_df.loc[prior_df['cdescr_model_text'].ne(''), 'cdescr_model_text']
    later_text = later_df.loc[later_df['cdescr_model_text'].ne(''), 'cdescr_model_text']
    prior_set = set(prior_text.tolist())
    later_overlap_mask = later_text.isin(prior_set)
    later_unique = later_text.nunique()
    overlap_unique = later_text.loc[later_overlap_mask].nunique()

    return {
        'prior_split': prior_name,
        'later_split': later_name,
        'prior_rows': int(len(prior_df)),
        'later_rows': int(len(later_df)),
        'prior_nonempty_rows': int(prior_text.shape[0]),
        'later_nonempty_rows': int(later_text.shape[0]),
        'prior_unique_texts': int(len(prior_set)),
        'later_unique_texts': int(later_unique),
        'overlap_unique_texts': int(overlap_unique),
        'overlap_unique_pct': round(float(overlap_unique / max(later_unique, 1) * 100), 4),
        'overlap_rows': int(later_overlap_mask.sum()),
        'overlap_row_pct': round(float(later_overlap_mask.mean() * 100), 4) if len(later_overlap_mask) else 0.0
    }


def build_overlap_report(sidecar_df):
    policy = get_split_policy(FEATURE_WAVE1_SPLIT_MODE)
    work = sidecar_df.copy()
    work['ldate'] = pd.to_datetime(work['ldate'], errors='coerce')

    train_df = work.loc[work['ldate'] <= policy['train_core_end']].copy()
    screen_df = work.loc[(work['ldate'] > policy['train_core_end']) & (work['ldate'] <= policy['screen_end'])].copy()
    select_df = work.loc[(work['ldate'] > policy['screen_end']) & (work['ldate'] <= policy['select_end'])].copy()
    holdout_df = work.loc[work['ldate'] > policy['select_end']].copy()
    dev_screen_df = pd.concat([train_df, screen_df], ignore_index=True)
    dev_select_df = pd.concat([dev_screen_df, select_df], ignore_index=True)

    rows = [
        build_overlap_row('train_core', 'screen_2024', train_df, screen_df),
        build_overlap_row('dev_2020_2024', 'select_2025', dev_screen_df, select_df),
        build_overlap_row('dev_2020_2025', 'holdout_2026', dev_select_df, holdout_df)
    ]
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Clean ODI complaints and build audited structured task tables'
    )
    parser.add_argument(
        '--input-path',
        default=None,
        help='Optional path to the combined complaint parquet or csv file'
    )
    parser.add_argument(
        '--output-format',
        choices=['parquet', 'csv'],
        default=settings.OUTPUT_FORMAT if settings.OUTPUT_FORMAT in {'parquet', 'csv'} else 'parquet',
        help='Preferred output format for cleaned tables'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Whether to build and output summary tables for cleaning and case collapse steps'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    raw_df, _ = load_frame(INPUT_STEM, args.input_path)
    work_df = build_cleaning_work(raw_df)
    cleaned_df = select_clean_columns(work_df)
    audit_df = build_cleaning_audit(work_df)
    severity_df = build_severity_cases(cleaned_df, audit_df)
    component_df = build_component_rows(cleaned_df, audit_df)

    clean_path = write_dataframe(
        cleaned_df,
        PROCESSED_DATA_DIR / CLEAN_STEM,
        prefer_parquet=args.output_format == 'parquet'
    )
    severity_path = write_dataframe(
        severity_df,
        PROCESSED_DATA_DIR / SEVERITY_STEM,
        prefer_parquet=args.output_format == 'parquet'
    )

    if args.summary:
        cleaning_summary_df = build_summary(cleaned_df, audit_df, severity_df, component_df)
        drift_df = build_source_era_drift(audit_df)
        cleaning_summary_path = OUTPUTS_DIR / CLEANING_SUMMARY_NAME
        drift_path = OUTPUTS_DIR / DRIFT_NAME
        cleaning_summary_df.to_csv(cleaning_summary_path, index=False)
        drift_df.to_csv(drift_path, index=False)
        print(f"[write] {cleaning_summary_path}")
        print(f'[write] {drift_path}')

    print(f"[write] {clean_path}")
    print(f"[write] {severity_path}")
    print("")
    print("[done 1/3] Complaint preprocessing finished")

    keep_df, single_rows, multi_rows, base_case_df, single_case_df, single_case_bench_df, multi_case_df = build_case_tables(component_df)

    single_case_path = write_dataframe(
        single_case_df,
        PROCESSED_DATA_DIR / SINGLE_CASE_STEM,
        prefer_parquet=args.output_format == 'parquet'
    )
    multi_case_path = write_dataframe(
        multi_case_df,
        PROCESSED_DATA_DIR / MULTI_CASE_STEM,
        prefer_parquet=args.output_format == 'parquet'
    )

    if args.summary:
        component_summary_df = build_collapse_summary(component_df, keep_df, base_case_df, single_case_df, single_case_bench_df, multi_case_df)
        conflict_df = pd.concat(
            [
                build_conflict_summary(keep_df, 'all_kept_cases'),
                build_conflict_summary(single_rows, 'single_label_cases'),
                build_conflict_summary(multi_rows, 'multi_label_cases')
            ],
            ignore_index=True
        )
        target_scope_df = build_target_scope_summary(base_case_df, single_case_df, single_case_bench_df, multi_case_df)
        target_group_df = build_target_group_summary(keep_df, single_case_df, single_case_bench_df, multi_rows)
        component_summary_path = OUTPUTS_DIR / COMPONENT_SUMMARY_NAME
        conflict_path = OUTPUTS_DIR / CONFLICT_NAME
        target_scope_path = OUTPUTS_DIR / TARGET_SCOPE_NAME
        target_group_path = OUTPUTS_DIR / TARGET_GROUP_NAME
        component_summary_df.to_csv(component_summary_path, index=False)
        conflict_df.to_csv(conflict_path, index=False)
        target_scope_df.to_csv(target_scope_path, index=False)
        target_group_df.to_csv(target_group_path, index=False)

        print(f"[write] {component_summary_path}")
        print(f"[write] {conflict_path}")
        print(f"[write] {target_scope_path}")
        print(f"[write] {target_group_path}")

    print(f"[write] {single_case_path}")
    print(f"[write] {multi_case_path}")
    print("")
    print("[done 2/3] Component case collapse finished")

    odino_universe = base_case_df['odino'].dropna().astype(str).unique().tolist()
    sidecar_df, text_base_df = select_best_text_rows(cleaned_df, odino_universe)

    sidecar_path = write_dataframe(
        sidecar_df,
        PROCESSED_DATA_DIR / SIDECAR_STEM,
        prefer_parquet=args.output_format == 'parquet'
    )

    if args.summary:
        text_conflict_df = build_conflict_report(text_base_df, sidecar_df)
        overlap_df = build_overlap_report(sidecar_df)
        text_conflict_path = OUTPUTS_DIR / TEXT_CONFLICT_NAME
        overlap_path = OUTPUTS_DIR / OVERLAP_NAME
        text_conflict_df.to_csv(text_conflict_path, index=False)
        overlap_df.to_csv(overlap_path, index=False)
        print(f"[write] {text_conflict_path}")
        print(f"[write] {overlap_path}")

    print(f"[write] {sidecar_path}")
    print("")
    print("[done 3/3] Component text sidecar finished")
    print("All preprocessing steps completed successfully")
    return 0


if __name__ == '__main__':
    sys.exit(main())
