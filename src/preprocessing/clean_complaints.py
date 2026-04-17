import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import settings
from src.config.contracts import (
    CLEANED_COMPLAINTS_STEM,
    CLEANING_AUDIT_STEM,
    CLEANING_DRIFT_NAME,
    CLEANING_SUMMARY_NAME,
    COMBINED_COMPLAINTS_STEM,
    COMPONENT_ROWS_STEM,
    SEVERITY_CASES_STEM,
)
from src.config.paths import OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_project_directories
from src.config.split_policy import REFERENCE_MODEL_YEAR_MAX
from src.data.io_utils import write_dataframe

# -----------------------------------------------------------------------------
# Output names
# -----------------------------------------------------------------------------
INPUT_STEM = COMBINED_COMPLAINTS_STEM
CLEAN_STEM = CLEANED_COMPLAINTS_STEM
AUDIT_STEM = CLEANING_AUDIT_STEM
SEVERITY_STEM = SEVERITY_CASES_STEM
COMPONENT_STEM = COMPONENT_ROWS_STEM
SUMMARY_NAME = CLEANING_SUMMARY_NAME
DRIFT_NAME = CLEANING_DRIFT_NAME


# -----------------------------------------------------------------------------
# Domain constants
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

CASE_FIRST_COLS = [
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

CASE_MAX_COLS = [
    'injured',
    'deaths',
    'miles',
    'veh_speed'
]

CASE_ANY_FLAGS = [
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
# Small helpers
# -----------------------------------------------------------------------------
def require_columns(df):
    missing = [column for column in REQ_COLS if column not in df.columns]
    if missing:
        missing_text = ', '.join(missing)
        raise ValueError(f'Missing required complaint columns: {missing_text}')


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
        'No combined complaints file found under data/processed. Run the ODI ingest first'
    )


def load_complaints(input_path=None):
    path = resolve_input_path(input_path)
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


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
    keep_cols = [column for column in AUDIT_KEY_COLS + AUDIT_VALUE_COLS if column in work.columns]
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


def clean_complaints(df):
    work = build_cleaning_work(df)
    return select_clean_columns(work)


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

    first_cols = [column for column in CASE_FIRST_COLS if column in vehicle_df.columns]
    if first_cols:
        frames.append(grouped[first_cols].first())

    max_cols = [column for column in CASE_MAX_COLS if column in vehicle_df.columns]
    if max_cols:
        frames.append(grouped[max_cols].max())

    any_flag_cols = [column for column in CASE_ANY_FLAGS if column in vehicle_df.columns]
    if any_flag_cols:
        frames.append(grouped[any_flag_cols].any())

    yn_cols = [column for column in ['fire', 'crash', 'medical_attn', 'vehicles_towed_yn'] if column in vehicle_df.columns]
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

    for column in yn_cols:
        yes_col = f'__{column}_yes'
        present_col = f'__{column}_present'
        case_df[column] = pd.NA
        case_df.loc[case_df[present_col], column] = 'N'
        case_df.loc[case_df[yes_col], column] = 'Y'
        case_df = case_df.drop(columns=[yes_col, present_col])

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
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    raw_df = load_complaints(args.input_path)
    work_df = build_cleaning_work(raw_df)
    cleaned_df = select_clean_columns(work_df)
    audit_df = build_cleaning_audit(work_df)
    severity_df = build_severity_cases(cleaned_df, audit_df)
    component_df = build_component_rows(cleaned_df, audit_df)
    summary_df = build_summary(cleaned_df, audit_df, severity_df, component_df)
    drift_df = build_source_era_drift(audit_df)

    clean_path = write_dataframe(
        cleaned_df,
        PROCESSED_DATA_DIR / CLEAN_STEM,
        prefer_parquet=args.output_format == 'parquet'
    )
    audit_path = write_dataframe(
        audit_df,
        PROCESSED_DATA_DIR / AUDIT_STEM,
        prefer_parquet=args.output_format == 'parquet'
    )
    severity_path = write_dataframe(
        severity_df,
        PROCESSED_DATA_DIR / SEVERITY_STEM,
        prefer_parquet=args.output_format == 'parquet'
    )
    component_path = write_dataframe(
        component_df,
        PROCESSED_DATA_DIR / COMPONENT_STEM,
        prefer_parquet=args.output_format == 'parquet'
    )
    summary_path = OUTPUTS_DIR / SUMMARY_NAME
    drift_path = OUTPUTS_DIR / DRIFT_NAME
    summary_df.to_csv(summary_path, index=False)
    drift_df.to_csv(drift_path, index=False)

    print(f'[write] {clean_path}')
    print(f'[write] {audit_path}')
    print(f'[write] {severity_path}')
    print(f'[write] {component_path}')
    print(f'[write] {summary_path}')
    print(f'[write] {drift_path}')
    print('')
    print('[done] Complaint preprocessing finished')
    return 0


if __name__ == '__main__':
    sys.exit(main())
