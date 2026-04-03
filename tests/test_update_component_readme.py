import json
import shutil
import uuid
from pathlib import Path

from src.reporting.update_component_readme import update_component_readme


def test_update_component_readme_replaces_marker_block():
    tmp_path = Path('data/outputs') / f'test_readme_{uuid.uuid4().hex}'
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        readme_path = tmp_path / 'README.md'
        readme_path.write_text(
            '\n'.join(
                [
                    '# Title',
                    '<!-- COMPONENT_BENCHMARK_START -->',
                    'old block',
                    '<!-- COMPONENT_BENCHMARK_END -->',
                    'tail'
                ]
            ),
            encoding='utf-8'
        )

        single_manifest = tmp_path / 'single.json'
        single_manifest.write_text(
            json.dumps(
                {
                    'selected_feature_set': 'core_structured',
                    'selected_iteration': 123,
                    'official_holdout_metrics': {
                        'macro_f1': 0.2222,
                        'top_1_accuracy': 0.3333,
                        'top_3_accuracy': 0.5555
                    }
                }
            ),
            encoding='utf-8'
        )

        multi_manifest = tmp_path / 'multi.json'
        multi_manifest.write_text(
            json.dumps(
                {
                    'selected_model': 'CatBoost MultiLabel',
                    'selected_feature_set': 'core_structured',
                    'selected_threshold': 0.3,
                    'selected_iteration': 250,
                    'official_holdout_metrics': {
                        'macro_f1': 0.1111,
                        'micro_f1': 0.4444,
                        'recall_at_3': 0.6666,
                        'precision_at_3': 0.2222
                    }
                }
            ),
            encoding='utf-8'
        )

        update_component_readme(
            single_manifest_path=single_manifest,
            multi_manifest_path=multi_manifest,
            readme_path=readme_path
        )

        updated = readme_path.read_text(encoding='utf-8')
        assert '0.2222' in updated
        assert '0.4444' in updated
        assert 'CatBoost MultiLabel' in updated
        assert '250' in updated
        assert 'old block' not in updated
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
