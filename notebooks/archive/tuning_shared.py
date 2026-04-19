import pandas as pd

from src.config.paths import OUTPUTS_DIR
from src.modeling.common.helpers import fit_catboost_with_external_selection


def log_line(message=''):
    print(message, flush=True)


def evaluate_params_across_seeds(train_df, valid_df, feature_info, params, task_type, devices, seed_list, verbose, selection_eval_period=1, progress_label=None):
    run_rows = []
    total_seeds = len(seed_list)
    for seed_idx, seed in enumerate(seed_list, start=1):
        result = fit_catboost_with_external_selection(
            train_df,
            valid_df,
            feature_info,
            params,
            task_type=task_type,
            devices=devices,
            random_seed=seed,
            verbose=verbose,
            selection_eval_period=selection_eval_period,
            include_train_outputs=False,
            include_valid_outputs=False
        )
        row = {
            'seed': seed,
            'fit_seconds': result['fit_seconds'],
            'selected_iteration': result['selected_iteration'],
            'top_1_accuracy': result['valid_metrics']['top_1_accuracy'],
            'macro_f1': result['valid_metrics']['macro_f1'],
            'top_3_accuracy': result['valid_metrics']['top_3_accuracy']
        }
        run_rows.append(row)

        if progress_label:
            log_line(
                f'[{progress_label}] seed {seed_idx}/{total_seeds} ({seed}) '
                f'fit={row["fit_seconds"]:.2f}s iter={row["selected_iteration"]} '
                f'macro_f1={row["macro_f1"]:.4f} top1={row["top_1_accuracy"]:.4f} '
                f'top3={row["top_3_accuracy"]:.4f}'
            )
    return pd.DataFrame(run_rows)


def summarize_seed_metrics(run_df):
    return {
        'fit_seconds_mean': round(float(run_df['fit_seconds'].mean()), 2),
        'fit_seconds_std': round(float(run_df['fit_seconds'].std(ddof=0)), 2),
        'selected_iteration_mean': round(float(run_df['selected_iteration'].mean()), 2),
        'selected_iteration_median': int(run_df['selected_iteration'].median()),
        'top_1_accuracy_mean': round(float(run_df['top_1_accuracy'].mean()), 4),
        'top_1_accuracy_std': round(float(run_df['top_1_accuracy'].std(ddof=0)), 4),
        'macro_f1_mean': round(float(run_df['macro_f1'].mean()), 4),
        'macro_f1_std': round(float(run_df['macro_f1'].std(ddof=0)), 4),
        'top_3_accuracy_mean': round(float(run_df['top_3_accuracy'].mean()), 4),
        'top_3_accuracy_std': round(float(run_df['top_3_accuracy'].std(ddof=0)), 4)
    }


# -----------------------------------------------------------------------------
# Archived Wave 2 manifest helpers
# -----------------------------------------------------------------------------
LOCKED_SINGLE_METRICS = OUTPUTS_DIR / 'component_single_label_benchmark_metrics.csv'
LOCKED_SINGLE_MANIFEST = OUTPUTS_DIR / 'component_single_label_benchmark_manifest.json'
LOCKED_SINGLE_SELECTION = OUTPUTS_DIR / 'component_single_label_selection_manifest.json'
LOCKED_SINGLE_CALIBRATION = OUTPUTS_DIR / 'component_single_label_holdout_calibration.csv'
LOCKED_MULTI_METRICS = OUTPUTS_DIR / 'component_multilabel_metrics.csv'
LOCKED_MULTI_MANIFEST = OUTPUTS_DIR / 'component_multilabel_manifest.json'

GLOBAL_MANIFEST_NAME = 'component_textwave2_manifest.json'
FEATUREWAVE_TASK = 'text_wave2'

TEXT_CONFIG = {
    'word_tfidf': {
        'analyzer': 'word',
        'ngram_range': [1, 2],
        'min_df': 5,
        'max_df': 0.995,
        'sublinear_tf': True,
        'lowercase': True,
        'strip_accents': 'unicode',
        'dtype': 'float32'
    },
    'char_tfidf': {
        'analyzer': 'char_wb',
        'ngram_range': [3, 5],
        'min_df': 5,
        'sublinear_tf': True,
        'lowercase': True,
        'strip_accents': 'unicode',
        'dtype': 'float32'
    },
    'fusion_text_weights': [0.25, 0.50, 0.75],
    'multi_threshold_grid': [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.30],
    'structured_feature_set': 'wave1_incident_cohort_history',
    'final_linear_model_default': 'sgd'
}


def read_locked_single_select_baseline():
    metric_df = pd.read_csv(LOCKED_SINGLE_METRICS)
    row = metric_df.loc[
        metric_df['model'].eq('CatBoost')
        & metric_df['stage'].eq('selection_train_valid')
        & metric_df['split'].eq('valid_2025')
    ]
    if row.empty:
        raise ValueError('Locked single-label validation baseline row is missing')
    return row.iloc[0].to_dict()


def read_locked_multi_select_baseline():
    metric_df = pd.read_csv(LOCKED_MULTI_METRICS)
    row = metric_df.loc[
        metric_df['model'].eq('CatBoost MultiLabel')
        & metric_df['stage'].eq('selection_train_valid')
        & metric_df['split'].eq('valid_2025')
    ]
    if row.empty:
        raise ValueError('Locked multi-label validation baseline row is missing')
    return row.iloc[0].to_dict()


def read_locked_single_ece():
    calib_df = pd.read_csv(LOCKED_SINGLE_CALIBRATION)
    overall = calib_df.loc[calib_df['section'].eq('overall')]
    if overall.empty:
        raise ValueError('Locked single-label calibration file is missing its overall row')
    return float(overall['ece'].iloc[0])


def empty_single_holdout_df():
    return pd.DataFrame(
        columns=[
            'task',
            'family_name',
            'family_kind',
            'structured_feature_set',
            'input_path',
            'text_sidecar_path',
            'split_mode',
            'text_enabled',
            'model',
            'stage',
            'split',
            'rows',
            'fit_seconds',
            'selected_iteration',
            'top_1_accuracy',
            'macro_f1',
            'top_3_accuracy',
            'selected_text_weight',
            'prior_text_overlap'
        ]
    )


def empty_single_calibration_df():
    return pd.DataFrame(
        columns=[
            'section',
            'bin',
            'count',
            'share',
            'accuracy',
            'avg_confidence',
            'gap',
            'ece',
            'multiclass_brier'
        ]
    )


def empty_single_confusion_df():
    return pd.DataFrame(
        columns=[
            'true_group',
            'pred_group',
            'count',
            'row_share'
        ]
    )


def empty_multi_holdout_df():
    return pd.DataFrame(
        columns=[
            'task',
            'family_name',
            'family_kind',
            'structured_feature_set',
            'input_path',
            'text_sidecar_path',
            'split_mode',
            'text_enabled',
            'model',
            'stage',
            'split',
            'rows',
            'fit_seconds',
            'selected_iteration',
            'threshold',
            'micro_f1',
            'macro_f1',
            'recall_at_3',
            'precision_at_3',
            'label_coverage',
            'selected_text_weight',
            'prior_text_overlap'
        ]
    )


def build_single_manifest_entry(result, locked_select_baseline, locked_holdout_baseline, locked_holdout_ece):
    return {
        'input_path': result.get('input_path'),
        'text_sidecar_path': result.get('text_sidecar_path'),
        'selected_family': result.get('selected_family'),
        'final_linear_model': result.get('final_linear_model'),
        'screen_fusion_weight': result.get('screen_fusion_weight'),
        'select_metrics': result.get('select_metrics'),
        'locked_select_baseline': locked_select_baseline,
        'locked_holdout_baseline': locked_holdout_baseline,
        'locked_holdout_ece': locked_holdout_ece,
        'select_gate_pass': result.get('select_gate_pass'),
        'promotion_status': result.get('promotion_status'),
        'holdout_overlap_slices': result.get('overlap_metrics', []),
        'checkpoint_stage': result.get('checkpoint_stage'),
        'completed_stages': result.get('completed_stages', []),
        'artifacts': {}
    }


def build_multi_manifest_entry(result, locked_select_baseline, locked_holdout_baseline):
    return {
        'input_path': result.get('input_path'),
        'text_sidecar_path': result.get('text_sidecar_path'),
        'selected_family': result.get('selected_family'),
        'final_linear_model': result.get('final_linear_model'),
        'screen_fusion_weight': result.get('screen_fusion_weight'),
        'select_metrics': result.get('select_metrics'),
        'locked_select_baseline': locked_select_baseline,
        'locked_holdout_baseline': locked_holdout_baseline,
        'select_gate_pass': result.get('select_gate_pass'),
        'promotion_status': result.get('promotion_status'),
        'holdout_overlap_slices': result.get('overlap_metrics', []),
        'checkpoint_stage': result.get('checkpoint_stage'),
        'completed_stages': result.get('completed_stages', []),
        'artifacts': {}
    }
