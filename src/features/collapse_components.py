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
SUMMARY_NAME = 'collapse_components_summary.csv'
CONFLICT_NAME = 'collapse_components_conflicts.csv'


# -----------------------------------------------------------------------------
# Collapse rules
# -----------------------------------------------------------------------------
CASE_FIRST_COLS = [
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
    'num_cyls'
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
    'cruise_cont_yn'
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
    'flag_fail_pre_model_far'
]

MODEL_COLS = [
    'odino',
    'component_group',
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
    'miles_missing_flag',
    'veh_speed_missing_flag',
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
# Make sure collapsed processed files exist
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


# Load data in depending on file type
def load_component_rows(input_path=None):
    path = resolve_input_path(input_path)
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


# -----------------------------------------------------------------------------
# Case collapse
# -----------------------------------------------------------------------------
# Robust function to collapse multi-component complaints to a single case
def collapse_component_cases(component_df):
    keep_df = component_df.loc[
        component_df['component_keep_flag'].fillna(False) & component_df['odino'].notna()
    ].copy()
    if keep_df.empty:
        raise ValueError('No kept component rows found for case collapse')

    group_counts = keep_df.groupby('odino')['component_group'].nunique()
    single_ids = group_counts.loc[group_counts.eq(1)].index
    single_df = keep_df.loc[keep_df['odino'].isin(single_ids)].copy()
    if single_df.empty:
        raise ValueError('No single-label component cases found after filtering')

    single_df = single_df.sort_values(['odino', 'cmplid'], na_position='last')
    grouped = single_df.groupby('odino', sort=True)

    frames = [
        grouped.size().rename('component_row_count'),
        grouped['component_group'].first()
    ]

    first_cols = [column for column in CASE_FIRST_COLS if column in single_df.columns]
    if first_cols:
        frames.append(grouped[first_cols].first())

    max_cols = [column for column in CASE_MAX_COLS if column in single_df.columns]
    if max_cols:
        frames.append(grouped[max_cols].max())

    any_flag_cols = [column for column in CASE_ANY_FLAG_COLS if column in single_df.columns]
    if any_flag_cols:
        frames.append(grouped[any_flag_cols].any())

    yn_cols = [column for column in CASE_YN_COLS if column in single_df.columns]
    temp_cols = []
    for column in yn_cols:
        yes_col = f'__{column}_yes'
        present_col = f'__{column}_present'
        single_df[yes_col] = single_df[column].fillna('N').eq('Y')
        single_df[present_col] = single_df[column].notna()
        temp_cols.extend([yes_col, present_col])

    if temp_cols:
        frames.append(single_df.groupby('odino', sort=True)[temp_cols].max())

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
    case_df = add_severity_flags(
        case_df,
        primary_name='severity_primary_flag',
        broad_name='severity_broad_flag'
    )

    keep_cols = [column for column in MODEL_COLS if column in case_df.columns]
    case_df = case_df.loc[:, keep_cols].sort_values('odino').reset_index(drop=True)
    return keep_df, single_df, case_df


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
# Summary of any collapse conflicts that emerged from the collapse function
def build_conflict_summary(single_df):
    conflict_cols = [
        column
        for column in CASE_FIRST_COLS + CASE_MAX_COLS + CASE_YN_COLS
        if column in single_df.columns
    ]
    distinct = single_df.groupby('odino', sort=True)[conflict_cols].nunique(dropna=True)
    conflict_df = distinct.gt(1).sum().reset_index(name='conflict_cases')
    conflict_df = conflict_df.rename(columns={'index': 'column'})
    conflict_df['conflict_rate_pct'] = (
        conflict_df['conflict_cases'] / single_df['odino'].nunique() * 100
    ).round(4)
    return conflict_df.sort_values(['conflict_cases', 'column'], ascending=[False, True]).reset_index(drop=True)


# Summary of operations, original input, and final output
def build_summary(component_df, keep_df, single_df, case_df):
    all_cases = int(component_df['odino'].nunique())
    kept_cases = int(keep_df['odino'].nunique())
    single_cases = int(case_df['odino'].nunique())
    multi_cases = kept_cases - single_cases

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
            'metric': 'single_label_cases',
            'value': single_cases
        },
        {
            'metric': 'multi_label_cases',
            'value': multi_cases
        },
        {
            'metric': 'single_label_case_share_pct',
            'value': round(single_cases / kept_cases * 100, 2)
        },
        {
            'metric': 'component_model_groups',
            'value': int(case_df['component_group'].nunique())
        }
    ]
    return pd.DataFrame(summary_rows)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
# Optional arguments to add in command line
def parse_args():
    parser = argparse.ArgumentParser(
        description='Collapse kept component rows to a single-label case table for structured modeling'
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
        help='Preferred output format for the collapsed case table'
    )
    return parser.parse_args()

# The real meat and potatoes
def main():
    args = parse_args()
    ensure_project_directories()

    component_df = load_component_rows(args.input_path)
    keep_df, single_df, case_df = collapse_component_cases(component_df)
    summary_df = build_summary(component_df, keep_df, single_df, case_df)
    conflict_df = build_conflict_summary(single_df)

    case_path = write_dataframe(
        case_df,
        PROCESSED_DATA_DIR / CASE_STEM,
        prefer_parquet=args.output_format == 'parquet'
    )
    summary_path = OUTPUTS_DIR / SUMMARY_NAME
    conflict_path = OUTPUTS_DIR / CONFLICT_NAME
    summary_df.to_csv(summary_path, index=False)
    conflict_df.to_csv(conflict_path, index=False)

    print(f'[write] {case_path}')
    print(f'[write] {summary_path}')
    print(f'[write] {conflict_path}')
    print('')
    print('[done] Component case collapse finished')
    return 0


if __name__ == '__main__':
    sys.exit(main())
