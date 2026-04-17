import pandas as pd

from src.preprocessing.clean_complaints import (
    build_cleaning_audit,
    build_cleaning_work,
    build_source_era_drift,
    clean_complaints,
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
    cleaned = clean_complaints(pd.DataFrame([row_a, row_b]))
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
