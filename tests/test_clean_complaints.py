import pandas as pd

from src.preprocessing.clean_complaints import build_source_era_drift, clean_complaints


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


def test_clean_complaints_keeps_provenance_and_zero_flags():
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

    cleaned = clean_complaints(pd.DataFrame([row_a, row_b]))

    assert 'source_zip' in cleaned.columns
    assert 'source_file' in cleaned.columns
    assert cleaned.loc[0, 'source_era'] == 'pre_2021_schema_change'
    assert cleaned.loc[1, 'source_era'] == 'post_2021_schema_change'
    assert bool(cleaned.loc[0, 'miles_zero_flag']) is True
    assert bool(cleaned.loc[0, 'veh_speed_zero_flag']) is True
    assert pd.isna(cleaned.loc[0, 'miles'])
    assert pd.isna(cleaned.loc[0, 'veh_speed'])
    assert bool(cleaned.loc[0, 'miles_missing_flag']) is True
    assert bool(cleaned.loc[0, 'veh_speed_missing_flag']) is True


def test_source_era_drift_keeps_zero_behavior_visible():
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

    cleaned = clean_complaints(pd.DataFrame([row_a, row_b]))
    drift = build_source_era_drift(cleaned)

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
