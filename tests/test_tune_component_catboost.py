from argparse import Namespace

from src.modeling.tune_component_catboost import (
    DEF_SELECTION_EVAL_PERIOD,
    DEFAULT_SEED_TEXT,
    QUICK_FEATURE_SET,
    QUICK_SELECTION_EVAL_PERIOD,
    QUICK_TRIALS,
    build_fit_plan,
    resolve_run_config,
)


def build_args(**overrides):
    values = {
        'input_path': None,
        'task_type': 'CPU',
        'devices': '0',
        'n_trials': 40,
        'seed_list': DEFAULT_SEED_TEXT,
        'feature_selection_seed_list': None,
        'feature_set': None,
        'selection_eval_period': DEF_SELECTION_EVAL_PERIOD,
        'random_seed': 42,
        'verbose': 0,
        'quick': False
    }
    values.update(overrides)
    return Namespace(**values)


def test_default_run_config_keeps_feature_sweep():
    config = resolve_run_config(build_args())

    assert config['run_feature_selection'] is True
    assert config['manual_feature_set'] is None
    assert config['feature_set_names'] == [
        'core_structured',
        'core_plus_quality',
        'core_plus_stable_incident'
    ]
    assert config['seed_list'] == [42, 43, 44, 45, 46]
    assert config['selection_seed_list'] == [42, 43, 44, 45, 46]
    assert config['selection_eval_period'] == DEF_SELECTION_EVAL_PERIOD


def test_quick_run_config_uses_lighter_defaults():
    config = resolve_run_config(build_args(task_type='GPU', quick=True))
    fit_plan = build_fit_plan(config)

    assert config['manual_feature_set'] == QUICK_FEATURE_SET
    assert config['run_feature_selection'] is False
    assert config['n_trials'] == QUICK_TRIALS
    assert config['seed_list'] == [42]
    assert config['selection_seed_list'] == [42]
    assert config['selection_eval_period'] == QUICK_SELECTION_EVAL_PERIOD
    assert fit_plan['feature_selection_fits'] == 0
    assert fit_plan['optuna_fits'] == QUICK_TRIALS
    assert fit_plan['best_trial_rescore_fits'] == 1


def test_manual_feature_set_preserves_explicit_overrides():
    config = resolve_run_config(
        build_args(
            feature_set='core_plus_quality',
            n_trials=12,
            seed_list='42,43',
            feature_selection_seed_list='99',
            selection_eval_period=7,
            quick=True
        )
    )

    assert config['manual_feature_set'] == 'core_plus_quality'
    assert config['run_feature_selection'] is False
    assert config['n_trials'] == 12
    assert config['seed_list'] == [42, 43]
    assert config['selection_seed_list'] == [99]
    assert config['selection_eval_period'] == 7
