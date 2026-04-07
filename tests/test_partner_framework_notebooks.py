import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_notebook(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def notebook_text(nb):
    return '\n'.join(
        ''.join(cell.get('source', []))
        for cell in nb.get('cells', [])
    )


def test_partner_framework_notebooks_are_valid_json():
    paths = [
        PROJECT_ROOT / 'notebooks' / 'Severity_Ranking_Framework.ipynb',
        PROJECT_ROOT / 'notebooks' / 'NLP_Early_Warning_Framework.ipynb'
    ]
    for path in paths:
        nb = load_notebook(path)
        assert nb['nbformat'] == 4
        assert nb.get('cells')
        assert all(not cell.get('outputs') for cell in nb['cells'] if cell.get('cell_type') == 'code')


def test_severity_framework_contract():
    nb = load_notebook(PROJECT_ROOT / 'notebooks' / 'Severity_Ranking_Framework.ipynb')
    text = notebook_text(nb)
    required = [
        'Severity Ranking Framework',
        'odi_severity_cases.parquet',
        'severity_primary_flag',
        'TARGET_COL =',
        'valid_2025',
        'holdout_2026',
        'DummyClassifier',
        'SGDClassifier',
        'prepare_structured_features',
        'FeatureUnion',
        'TfidfVectorizer',
        'WORD_NGRAM_RANGE =',
        'TEXT_MIN_DF =',
        'text_tfidf_word_char_sgd',
        'severity_partner_results.csv',
        'Next TODOs / Extension Ideas'
    ]
    for item in required:
        assert item in text


def test_early_warning_framework_contract():
    nb = load_notebook(PROJECT_ROOT / 'notebooks' / 'NLP_Early_Warning_Framework.ipynb')
    text = notebook_text(nb)
    required = [
        'NLP Early Warning',
        'odi_component_multilabel_cases.parquet',
        'odi_component_text_sidecar.parquet',
        'component_groups',
        'cohort_key_cols =',
        'SHORT_WINDOW_MONTHS =',
        'LONG_WINDOW_MONTHS =',
        "cohort_grouped = feature_df.groupby(cohort_key_cols, sort=False, dropna=False)",
        "feature_df['months_with_history'] = cohort_grouped.cumcount()",
        "feature_df[col] = cohort_grouped['complaint_count'].shift(lag)",
        'prior_3m_mean_count',
        'prior_6m_mean_count',
        'WATCHLIST_WEIGHTS =',
        'watchlist_score',
        'nlp_early_warning_watchlist.csv',
        'nlp_early_warning_terms.csv',
        'Next TODOs / Extension Ideas'
    ]
    for item in required:
        assert item in text
