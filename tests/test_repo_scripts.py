from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_official_and_ingest_runner_scripts_exist():
    assert (PROJECT_ROOT / 'scripts' / 'run_ingest_windows.ps1').exists()
    assert (PROJECT_ROOT / 'scripts' / 'run_ingest_mac_linux.sh').exists()
    assert (PROJECT_ROOT / 'scripts' / 'run_component_official_windows.ps1').exists()
    assert (PROJECT_ROOT / 'scripts' / 'run_component_official_mac_linux.sh').exists()


def test_official_runner_scripts_call_the_new_pipeline_modules():
    windows_text = (PROJECT_ROOT / 'scripts' / 'run_component_official_windows.ps1').read_text(encoding='utf-8')
    mac_text = (PROJECT_ROOT / 'scripts' / 'run_component_official_mac_linux.sh').read_text(encoding='utf-8')

    for text in [windows_text, mac_text]:
        assert 'src.preprocessing.clean_complaints' in text
        assert 'src.modeling.component_single_text_calibrated' in text
        assert 'src.modeling.component_multi_routing' in text
        assert 'src.reporting.update_component_readme' in text
        assert 'src.reporting.component_visuals' in text
