import pandas as pd

from src.features.collapse_components import build_case_tables


def make_component_row(case_id, cmplid, component_group, ldate='2024-01-01'):
    return {
        'cmplid': str(cmplid),
        'odino': str(case_id),
        'component_group': component_group,
        'component_group_rows': 999,
        'component_keep_flag': True,
        'source_era': 'post_2021_schema_change',
        'source_zip': 'zip',
        'source_file': 'file',
        'mfr_name': 'MFR',
        'maketxt': 'MAKE',
        'modeltxt': 'MODEL',
        'yeartxt': 2020,
        'state': 'NC',
        'ldate': pd.Timestamp(ldate),
        'faildate': pd.Timestamp('2023-12-20'),
        'cmpl_type': 'C',
        'drive_train': 'AWD',
        'fuel_sys': 'FI',
        'fuel_type': 'GAS',
        'trans_type': 'AUTO',
        'num_cyls': 4,
        'miles': 1000,
        'veh_speed': 50,
        'injured': 0,
        'deaths': 0,
        'fire': 'N',
        'crash': 'N',
        'medical_attn': 'N',
        'vehicles_towed_yn': 'N',
        'police_rpt_yn': 'N',
        'orig_owner_yn': 'N',
        'anti_brakes_yn': 'N',
        'cruise_cont_yn': 'N',
        'repaired_yn': 'N',
        'miles_missing_flag': False,
        'veh_speed_missing_flag': False,
        'miles_zero_flag': False,
        'veh_speed_zero_flag': False,
        'faildate_trusted_flag': True,
        'faildate_untrusted_flag': False,
        'severity_primary_row_flag': False,
        'severity_broad_row_flag': False,
        'flag_year_unknown': False,
        'flag_year_out_of_range': False,
        'flag_speed_999': False,
        'flag_speed_high': False,
        'flag_miles_high': False,
        'flag_state_bad': False,
        'flag_date_order_bad': False,
        'flag_fail_pre_model': False,
        'flag_fail_pre_model_far': False
    }


def test_case_tables_split_base_single_and_multi_contracts():
    rows = []
    for case_id in range(1, 251):
        rows.append(make_component_row(case_id, case_id, 'ENGINE / COOLING', ldate='2024-01-01'))

    rows.append(make_component_row(999, 9991, 'PARKING BRAKE', ldate='2024-01-01'))
    rows.append(make_component_row(1000, 10001, 'LANE DEPARTURE', ldate='2025-01-01'))
    rows.append(make_component_row(1000, 10002, 'ELECTRONIC STABILITY CONTROL (ESC)', ldate='2025-01-01'))

    component_df = pd.DataFrame(rows)
    keep_df, single_rows, multi_rows, base_case_df, single_case_df, single_case_bench_df, multi_case_df = build_case_tables(component_df)

    assert len(keep_df) == len(component_df)
    assert len(base_case_df) == 252
    assert len(single_case_df) == 251
    assert len(single_case_bench_df) == 250
    assert len(multi_rows['odino'].unique()) == 1
    assert len(multi_case_df) == 252

    multi_case = multi_case_df.loc[multi_case_df['odino'].eq('1000'), 'component_groups'].iloc[0]
    assert multi_case == 'ELECTRONIC STABILITY CONTROL (ESC)|LANE DEPARTURE'

    for frame in [base_case_df, single_case_df, multi_case_df]:
        assert 'complaint_year' not in frame.columns
        assert 'vehicle_age_years' not in frame.columns
        assert 'state_region' not in frame.columns
        assert 'prior_cmpl_mfr_all' not in frame.columns

    assert 'component_group_fit_case_count' in single_case_df.columns
    assert 'single_label_keep_flag' in single_case_df.columns
    assert 'component_group_count' in multi_case_df.columns


def test_single_label_keep_flag_uses_preholdout_fit_window_only():
    rows = []
    for case_id in range(1, 251):
        rows.append(make_component_row(case_id, case_id, 'ENGINE / COOLING', ldate='2025-06-01'))

    for offset in range(251, 401):
        rows.append(make_component_row(offset, offset, 'PARKING BRAKE', ldate='2026-02-01'))

    component_df = pd.DataFrame(rows)
    _, _, _, _, single_case_df, single_case_bench_df, _ = build_case_tables(component_df)

    engine_row = single_case_df.loc[single_case_df['component_group'].eq('ENGINE / COOLING')].iloc[0]
    parking_row = single_case_df.loc[single_case_df['component_group'].eq('PARKING BRAKE')].iloc[0]

    assert int(engine_row['component_group_fit_case_count']) == 250
    assert bool(engine_row['single_label_keep_flag']) is True
    assert pd.isna(parking_row['component_group_fit_case_count']) or int(parking_row['component_group_fit_case_count']) == 0
    assert bool(parking_row['single_label_keep_flag']) is False
    assert 'PARKING BRAKE' not in single_case_bench_df['component_group'].tolist()
