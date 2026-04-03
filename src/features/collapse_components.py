import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import settings
from src.config.paths import OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_project_directories
from src.data.io_utils import write_dataframe
from src.preprocessing.clean_complaints import add_safe_lag_fields, add_severity_flags

# -----------------------------------------------------------------------------
# Output names
# -----------------------------------------------------------------------------
INPUT_STEM = 'odi_component_rows'
CASE_STEM = 'odi_component_model_cases'
MULTI_CASE_STEM = 'odi_component_multilabel_cases'
SUMMARY_NAME = 'collapse_components_summary.csv'
CONFLICT_NAME = 'collapse_components_conflicts.csv'
TARGET_SCOPE_NAME = 'component_target_scope_summary.csv'
TARGET_GROUP_NAME = 'component_target_scope_groups.csv'


# -----------------------------------------------------------------------------
# Collapse rules
# -----------------------------------------------------------------------------
SINGLE_LABEL_MIN_CASES = 250
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

CASE_FIRST_COLS = [
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

CASE_MAX_COLS = [
    'component_group_rows',
    'injured',
    'deaths',
    'miles',
    'veh_speed'
]

CASE_YN_COLS = [
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

CASE_ANY_FLAG_COLS = [
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

SINGLE_MODEL_COLS = [
    'odino',
    'source_era',
    'source_zip',
    'source_file',
    'component_group',
    'component_row_count',
    'component_group_rows',
    'component_group_case_count',
    'mfr_name',
    'maketxt',
    'modeltxt',
    'yeartxt',
    'state',
    'ldate',
    'faildate',
    'lag_days_safe',
    'complaint_year',
    'complaint_month',
    'complaint_quarter',
    'vehicle_age_years',
    'vehicle_age_bucket',
    'state_region',
    'prior_cmpl_mfr_all',
    'prior_cmpl_make_model_all',
    'prior_cmpl_make_model_year_all',
    'prior_severity_share_mfr_all',
    'prior_severity_share_make_model_all',
    'prior_severity_share_make_model_year_all',
    'cmpl_type',
    'drive_train',
    'fuel_sys',
    'fuel_type',
    'trans_type',
    'num_cyls',
    'miles',
    'veh_speed',
    'injured',
    'deaths',
    'fire',
    'crash',
    'medical_attn',
    'vehicles_towed_yn',
    'police_rpt_yn',
    'orig_owner_yn',
    'anti_brakes_yn',
    'cruise_cont_yn',
    'repaired_yn',
    'miles_missing_flag',
    'veh_speed_missing_flag',
    'miles_zero_flag',
    'veh_speed_zero_flag',
    'faildate_trusted_flag',
    'faildate_untrusted_flag',
    'severity_primary_flag',
    'severity_broad_flag',
    'flag_year_unknown',
    'flag_year_out_of_range',
    'flag_speed_high',
    'flag_miles_high',
    'flag_date_order_bad',
    'flag_fail_pre_model',
    'flag_fail_pre_model_far'
]

MULTI_MODEL_COLS = [
    'odino',
    'source_era',
    'source_zip',
    'source_file',
    'component_groups',
    'component_group_count',
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
    'complaint_year',
    'complaint_month',
    'complaint_quarter',
    'vehicle_age_years',
    'vehicle_age_bucket',
    'state_region',
    'prior_cmpl_mfr_all',
    'prior_cmpl_make_model_all',
    'prior_cmpl_make_model_year_all',
    'prior_severity_share_mfr_all',
    'prior_severity_share_make_model_all',
    'prior_severity_share_make_model_year_all',
    'cmpl_type',
    'drive_train',
    'fuel_sys',
    'fuel_type',
    'trans_type',
    'num_cyls',
    'miles',
    'veh_speed',
    'injured',
    'deaths',
    'fire',
    'crash',
    'medical_attn',
    'vehicles_towed_yn',
    'police_rpt_yn',
    'orig_owner_yn',
    'anti_brakes_yn',
    'cruise_cont_yn',
    'repaired_yn',
    'miles_missing_flag',
    'veh_speed_missing_flag',
    'miles_zero_flag',
    'veh_speed_zero_flag',
    'faildate_trusted_flag',
    'faildate_untrusted_flag',
    'severity_primary_flag',
    'severity_broad_flag',
    'flag_year_unknown',
    'flag_year_out_of_range',
    'flag_speed_high',
    'flag_miles_high',
    'flag_date_order_bad',
    'flag_fail_pre_model',
    'flag_fail_pre_model_far'
]


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
        'No component row file found under data/processed. Run complaint preprocessing first'
    )


def load_component_rows(input_path=None):
    path = resolve_input_path(input_path)
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


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

    first_cols = [column for column in CASE_FIRST_COLS if column in case_rows.columns]
    if first_cols:
        frames.append(grouped[first_cols].first())

    max_cols = [column for column in CASE_MAX_COLS if column in case_rows.columns]
    if max_cols:
        frames.append(grouped[max_cols].max())

    any_flag_cols = [column for column in CASE_ANY_FLAG_COLS if column in case_rows.columns]
    if any_flag_cols:
        frames.append(grouped[any_flag_cols].any())

    yn_cols = [column for column in CASE_YN_COLS if column in case_rows.columns]
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


def build_sort_keys(case_df):
    work = case_df.copy()
    work['__odino_num'] = pd.to_numeric(work['odino'], errors='coerce')
    return work.sort_values(
        ['ldate', '__odino_num', 'odino'],
        na_position='last'
    ).reset_index(drop=True)


def add_vehicle_age_fields(case_df):
    case_df = case_df.copy()
    case_df['complaint_year'] = pd.to_datetime(case_df['ldate'], errors='coerce').dt.year.astype('Int64')
    case_df['complaint_month'] = pd.to_datetime(case_df['ldate'], errors='coerce').dt.month.astype('Int64')
    case_df['complaint_quarter'] = pd.to_datetime(case_df['ldate'], errors='coerce').dt.quarter.astype('Int64')

    model_year = pd.to_numeric(case_df['yeartxt'], errors='coerce')
    vehicle_age = case_df['complaint_year'].astype('Float64') - model_year.astype('Float64')
    vehicle_age = vehicle_age.where(vehicle_age.ge(0))
    case_df['vehicle_age_years'] = vehicle_age
    age_bucket = pd.cut(
        vehicle_age,
        bins=VEHICLE_AGE_BUCKETS,
        labels=VEHICLE_AGE_LABELS,
        include_lowest=True
    )
    case_df['vehicle_age_bucket'] = age_bucket.astype('string')
    return case_df


def add_state_region(case_df):
    case_df = case_df.copy()
    state = case_df['state'].astype('string').str.upper()
    case_df['state_region'] = state.map(STATE_REGION_MAP).fillna('UNKNOWN').astype(str)
    return case_df


def add_prior_history_features(case_df):
    work = build_sort_keys(case_df)
    severity = work['severity_broad_flag'].fillna(False).astype(int)  # noqa: F841

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
        grouped = work.groupby(key_cols, sort=False, dropna=False)
        prior_count = grouped.cumcount()
        prior_severity_count = grouped['severity_broad_flag'].transform(
            lambda s: s.fillna(False).astype(int).cumsum() - s.fillna(False).astype(int)
        )
        work[count_name] = prior_count.astype('Int64')
        prior_share = prior_severity_count.astype(float) / prior_count.replace(0, pd.NA)
        work[share_name] = pd.Series(prior_share, index=work.index).astype('Float64')

    return work.drop(columns=['__odino_num'])


def add_wave1_case_features(case_df):
    case_df = add_vehicle_age_fields(case_df)
    case_df = add_state_region(case_df)
    return add_prior_history_features(case_df)


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

    base_case_df = add_wave1_case_features(collapse_case_features(keep_df, target_mode='base'))
    single_target_df = collapse_case_features(single_rows, target_mode='single')[['odino', 'component_group']]
    single_case_df = base_case_df.loc[base_case_df['odino'].isin(single_ids)].merge(
        single_target_df,
        on='odino',
        how='left',
        validate='one_to_one'
    )
    single_case_counts = single_case_df['component_group'].value_counts()
    single_case_df['component_group_case_count'] = (
        single_case_df['component_group']
        .map(single_case_counts)
        .astype('Int64')
    )
    single_case_df['single_label_keep_flag'] = single_case_df['component_group_case_count'].ge(SINGLE_LABEL_MIN_CASES)
    single_case_bench_df = single_case_df.loc[single_case_df['single_label_keep_flag']].copy()
    keep_cols = [column for column in SINGLE_MODEL_COLS if column in single_case_bench_df.columns]
    single_case_bench_df = single_case_bench_df.loc[:, keep_cols].sort_values('odino').reset_index(drop=True)

    multi_target_df = collapse_case_features(keep_df, target_mode='multi')[['odino', 'component_groups', 'component_group_count']]
    multi_case_df = base_case_df.merge(
        multi_target_df,
        on='odino',
        how='left',
        validate='one_to_one'
    )
    multi_keep_cols = [column for column in MULTI_MODEL_COLS if column in multi_case_df.columns]
    multi_case_df = multi_case_df.loc[:, multi_keep_cols].sort_values('odino').reset_index(drop=True)

    return keep_df, single_rows, multi_rows, single_case_df, single_case_bench_df, multi_case_df


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
def build_conflict_summary(case_rows, scope_name):
    if case_rows.empty:
        return pd.DataFrame(columns=['scope', 'column', 'conflict_cases', 'conflict_rate_pct'])

    conflict_cols = [
        column
        for column in CASE_FIRST_COLS + CASE_MAX_COLS + CASE_YN_COLS
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


def build_summary(component_df, keep_df, single_case_df, single_case_bench_df, multi_case_df):
    all_cases = int(component_df['odino'].nunique())
    kept_cases = int(keep_df['odino'].nunique())
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


def build_target_scope_summary(keep_df, single_case_df, single_case_bench_df, multi_case_df):
    rows = []
    kept_case_base = (
        keep_df.groupby('odino', sort=True)
        .agg(
            ldate=('ldate', 'first'),
            severity_broad_flag=('severity_broad_row_flag', 'max')
        )
        .reset_index()
    )
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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Collapse kept component rows into single-label and multi-label case tables'
    )
    parser.add_argument(
        '--input-path',
        default=None,
        help='Optional path to the component row parquet or csv file'
    )
    parser.add_argument(
        '--output-format',
        choices=['parquet', 'csv'],
        default=settings.OUTPUT_FORMAT if settings.OUTPUT_FORMAT in {'parquet', 'csv'} else 'parquet',
        help='Preferred output format for the collapsed case tables'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    component_df = load_component_rows(args.input_path)
    keep_df, single_rows, multi_rows, single_case_df, single_case_bench_df, multi_case_df = build_case_tables(component_df)

    summary_df = build_summary(component_df, keep_df, single_case_df, single_case_bench_df, multi_case_df)
    conflict_df = pd.concat(
        [
            build_conflict_summary(keep_df, 'all_kept_cases'),
            build_conflict_summary(single_rows, 'single_label_cases'),
            build_conflict_summary(multi_rows, 'multi_label_cases')
        ],
        ignore_index=True
    )
    target_scope_df = build_target_scope_summary(keep_df, single_case_df, single_case_bench_df, multi_case_df)
    target_group_df = build_target_group_summary(keep_df, single_case_df, single_case_bench_df, multi_rows)

    case_path = write_dataframe(
        single_case_bench_df,
        PROCESSED_DATA_DIR / CASE_STEM,
        prefer_parquet=args.output_format == 'parquet'
    )
    multi_case_path = write_dataframe(
        multi_case_df,
        PROCESSED_DATA_DIR / MULTI_CASE_STEM,
        prefer_parquet=args.output_format == 'parquet'
    )
    summary_path = OUTPUTS_DIR / SUMMARY_NAME
    conflict_path = OUTPUTS_DIR / CONFLICT_NAME
    target_scope_path = OUTPUTS_DIR / TARGET_SCOPE_NAME
    target_group_path = OUTPUTS_DIR / TARGET_GROUP_NAME
    summary_df.to_csv(summary_path, index=False)
    conflict_df.to_csv(conflict_path, index=False)
    target_scope_df.to_csv(target_scope_path, index=False)
    target_group_df.to_csv(target_group_path, index=False)

    print(f'[write] {case_path}')
    print(f'[write] {multi_case_path}')
    print(f'[write] {summary_path}')
    print(f'[write] {conflict_path}')
    print(f'[write] {target_scope_path}')
    print(f'[write] {target_group_path}')
    print('')
    print('[done] Component case collapse finished')
    return 0


if __name__ == '__main__':
    sys.exit(main())
