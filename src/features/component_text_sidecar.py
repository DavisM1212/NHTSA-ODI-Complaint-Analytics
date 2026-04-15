import argparse
import re
import sys

import pandas as pd

from src.config import settings
from src.config.paths import OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_project_directories
from src.data.io_utils import load_frame, write_dataframe
from src.modeling.component_common import (
    FEATURE_WAVE1_SPLIT_MODE,
    MULTI_INPUT_STEM,
    get_split_policy,
)

# -----------------------------------------------------------------------------
# Output names
# -----------------------------------------------------------------------------
CLEAN_STEM = 'odi_complaints_cleaned'
SIDECAR_STEM = 'odi_component_text_sidecar'
CONFLICT_NAME = 'component_text_sidecar_conflicts.csv'
OVERLAP_NAME = 'component_text_overlap_report.csv'


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
        description='Build the shared complaint narrative sidecar for component text modeling'
    )
    parser.add_argument(
        '--cleaned-input-path',
        default=None,
        help='Optional path to the cleaned complaints parquet or csv file'
    )
    parser.add_argument(
        '--multi-input-path',
        default=None,
        help='Optional path to the component multi-label case parquet or csv file'
    )
    parser.add_argument(
        '--output-format',
        choices=['parquet', 'csv'],
        default=settings.OUTPUT_FORMAT if settings.OUTPUT_FORMAT in {'parquet', 'csv'} else 'parquet',
        help='Preferred output format for the processed sidecar table'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    clean_df, clean_path = load_frame(CLEAN_STEM, input_path=args.cleaned_input_path)
    multi_df, multi_path = load_frame(MULTI_INPUT_STEM, input_path=args.multi_input_path)
    odino_universe = multi_df['odino'].dropna().astype(str).unique().tolist()

    sidecar_df, base_df = select_best_text_rows(clean_df, odino_universe)
    conflict_df = build_conflict_report(base_df, sidecar_df)
    overlap_df = build_overlap_report(sidecar_df)

    sidecar_path = write_dataframe(
        sidecar_df,
        PROCESSED_DATA_DIR / SIDECAR_STEM,
        prefer_parquet=args.output_format == 'parquet'
    )
    conflict_path = OUTPUTS_DIR / CONFLICT_NAME
    overlap_path = OUTPUTS_DIR / OVERLAP_NAME
    conflict_df.to_csv(conflict_path, index=False)
    overlap_df.to_csv(overlap_path, index=False)

    print(f'[input] cleaned={clean_path}')
    print(f'[input] multi={multi_path}')
    print(f'[write] {sidecar_path}')
    print(f'[write] {conflict_path}')
    print(f'[write] {overlap_path}')
    print('')
    print('[done] Component text sidecar finished')
    return 0


if __name__ == '__main__':
    sys.exit(main())
