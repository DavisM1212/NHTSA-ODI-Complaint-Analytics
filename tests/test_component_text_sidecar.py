import pandas as pd

from src.features.component_text_sidecar import (
    build_conflict_report,
    build_overlap_report,
    is_placeholder_text,
    select_best_text_rows,
)


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
