import pandas as pd

from src.preprocessing.clean_complaints import (
    build_case_tables,
    build_cleaning_audit,
    build_cleaning_work,
    build_conflict_report,
    build_overlap_report,
    build_source_era_drift,
    is_placeholder_text,
    select_best_text_rows,
    select_clean_columns,
)


def base_row():
    return {
        'cmplid': '1',
        'odino': '100',
        'mfr_name': 'maker',
        'maketxt': 'make',
        'modeltxt': 'model',
        'yeartxt': '2020',
        'crash': 'N',
        'faildate': '2020-01-01',
        'fire': 'N',
        'injured': '0',
        'deaths': '0',
        'compdesc': 'ENGINE',
        'city': 'charlotte',
        'state': 'nc',
        'vin': '12345678901',
        'datea': '2020-01-10',
        'ldate': '2020-01-10',
        'miles': '100',
        'cdescr': 'test',
        'veh_speed': '55',
        'dealer_state': 'nc',
        'prod_type': 'V',
        'medical_attn': 'N',
        'vehicles_towed_yn': 'N',
        'police_rpt_yn': 'N',
        'orig_owner_yn': 'N',
        'anti_brakes_yn': 'N',
        'cruise_cont_yn': 'N',
        'orig_equip_yn': 'N',
        'repaired_yn': 'N',
        'source_zip': 'COMPLAINTS_RECEIVED_2020-2024.zip',
        'source_file': 'CMPL.txt'
    }


def test_clean_complaints_keeps_shared_columns_but_moves_audit_flags_out():
    row_a = base_row()
    row_a['miles'] = '0'
    row_a['veh_speed'] = '0'
    row_a['ldate'] = '2020-01-10'

    row_b = base_row()
    row_b['cmplid'] = '2'
    row_b['odino'] = '101'
    row_b['ldate'] = '2025-01-10'
    row_b['datea'] = '2025-01-10'
    row_b['faildate'] = '2024-12-25'

    work_df = build_cleaning_work(pd.DataFrame([row_a, row_b]))
    cleaned = select_clean_columns(work_df)
    audit_df = build_cleaning_audit(work_df)

    assert 'source_zip' in cleaned.columns
    assert 'source_file' in cleaned.columns
    assert cleaned.loc[0, 'source_era'] == 'pre_2021_schema_change'
    assert cleaned.loc[1, 'source_era'] == 'post_2021_schema_change'

    assert 'miles_zero_flag' not in cleaned.columns
    assert 'veh_speed_zero_flag' not in cleaned.columns
    assert 'lag_days_safe' not in cleaned.columns
    assert 'miles_zero_flag' in audit_df.columns
    assert 'veh_speed_zero_flag' in audit_df.columns

    audit_row = audit_df.loc[audit_df['odino'].eq('100')].iloc[0]
    assert bool(audit_row['miles_zero_flag']) is True
    assert bool(audit_row['veh_speed_zero_flag']) is True


def test_source_era_drift_reads_from_cleaning_audit():
    row_a = base_row()
    row_a['miles'] = '0'
    row_a['veh_speed'] = '0'
    row_a['ldate'] = '2020-01-10'

    row_b = base_row()
    row_b['cmplid'] = '2'
    row_b['odino'] = '101'
    row_b['miles'] = '10'
    row_b['veh_speed'] = '20'
    row_b['ldate'] = '2025-01-10'
    row_b['datea'] = '2025-01-10'
    row_b['faildate'] = '2024-12-25'

    work_df = build_cleaning_work(pd.DataFrame([row_a, row_b]))
    audit_df = build_cleaning_audit(work_df)
    drift = build_source_era_drift(audit_df)

    miles_zero = drift.loc[
        drift['source_era'].eq('pre_2021_schema_change')
        & drift['field'].eq('miles')
        & drift['metric'].eq('zero_count'),
        'value'
    ].iloc[0]
    speed_zero = drift.loc[
        drift['source_era'].eq('pre_2021_schema_change')
        & drift['field'].eq('veh_speed')
        & drift['metric'].eq('zero_count'),
        'value'
    ].iloc[0]

    assert int(miles_zero) == 1
    assert int(speed_zero) == 1


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


def build_clean_row(odino, cmplid, cdescr, ldate, source_era='post_2021'):
    return {
        'odino': odino,
        'cmplid': cmplid,
        'cdescr': cdescr,
        'source_era': source_era,
        'ldate': ldate
    }


def test_select_best_text_rows_prefers_longest_then_earliest_then_cmplid():
    clean_df = pd.DataFrame(
        [
            build_clean_row('1', '11', 'short', '2024-01-02'),
            build_clean_row('1', '12', 'longer complaint narrative', '2024-01-02'),
            build_clean_row('1', '10', 'longer complaint narrative', '2024-01-01'),
            build_clean_row('2', '20', '   ', '2024-02-01'),
            build_clean_row('2', '21', None, '2024-02-02')
        ]
    )

    sidecar_df, base_df = select_best_text_rows(clean_df, ['1', '2'])

    chosen_one = sidecar_df.loc[sidecar_df['odino'].eq('1')].iloc[0]
    chosen_two = sidecar_df.loc[sidecar_df['odino'].eq('2')].iloc[0]

    assert chosen_one['cdescr'] == 'longer complaint narrative'
    assert chosen_one['cdescr_model_text'] == 'longer complaint narrative'
    assert bool(chosen_one['cdescr_missing_flag']) is False
    assert bool(chosen_two['cdescr_missing_flag']) is True
    assert chosen_two['cdescr_model_text'] == ''
    assert len(base_df) == 5


def test_placeholder_text_rules_cover_named_and_short_nonalpha_values():
    assert is_placeholder_text('N/A') is True
    assert is_placeholder_text('12345') is True
    assert is_placeholder_text('TEST TEST') is True
    assert is_placeholder_text('engine stall on highway') is False


def test_conflict_and_overlap_reports_capture_expected_rows():
    clean_df = pd.DataFrame(
        [
            build_clean_row('1', '1', 'engine failed', '2023-01-01'),
            build_clean_row('1', '2', 'engine failed badly', '2023-01-02'),
            build_clean_row('2', '3', 'same text', '2024-02-01'),
            build_clean_row('3', '4', 'same text', '2025-03-01'),
            build_clean_row('4', '5', 'same text', '2026-04-01')
        ]
    )

    sidecar_df, base_df = select_best_text_rows(clean_df, ['1', '2', '3', '4'])
    conflict_df = build_conflict_report(base_df, sidecar_df)
    overlap_df = build_overlap_report(sidecar_df)

    assert conflict_df['odino'].tolist() == ['1']

    select_overlap = overlap_df.loc[
        overlap_df['prior_split'].eq('dev_2020_2024')
        & overlap_df['later_split'].eq('select_2025')
    ].iloc[0]
    holdout_overlap = overlap_df.loc[
        overlap_df['prior_split'].eq('dev_2020_2025')
        & overlap_df['later_split'].eq('holdout_2026')
    ].iloc[0]

    assert int(select_overlap['overlap_unique_texts']) == 1
    assert int(holdout_overlap['overlap_unique_texts']) == 1
