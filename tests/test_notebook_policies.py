from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_notebooks_are_not_output_stripped():
    gitattributes = (PROJECT_ROOT / '.gitattributes').read_text(encoding='utf-8')

    assert 'notebooks/EDA.ipynb filter=notebookstrip' not in gitattributes
    assert 'notebooks/Cleaning.ipynb filter=notebookstrip' not in gitattributes
    assert 'notebooks/Severity_Ranking_Framework.ipynb filter=notebookstrip' not in gitattributes
    assert 'notebooks/NLP_Early_Warning_Framework.ipynb filter=notebookstrip' not in gitattributes


def test_severity_and_nlp_notebooks_do_not_import_src_modules():
    severity_text = (PROJECT_ROOT / 'notebooks' / 'Severity_Ranking_Framework.ipynb').read_text(encoding='utf-8')
    nlp_text = (PROJECT_ROOT / 'notebooks' / 'NLP_Early_Warning_Framework.ipynb').read_text(encoding='utf-8')

    assert 'from src.' not in severity_text
    assert 'import src.' not in severity_text
    assert 'from src.' not in nlp_text
    assert 'import src.' not in nlp_text


def test_cleaning_notebook_keeps_the_allowed_local_import_exception():
    cleaning_text = (PROJECT_ROOT / 'notebooks' / 'Cleaning.ipynb').read_text(encoding='utf-8')

    assert 'from src.data.schema_checks import get_schema_spec' in cleaning_text


def test_severity_holdout_stays_off_by_default():
    severity_text = (PROJECT_ROOT / 'notebooks' / 'Severity_Ranking_Framework.ipynb').read_text(encoding='utf-8')

    assert 'RUN_HOLDOUT = False' in severity_text
