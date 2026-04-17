import pandas as pd

from src.modeling.common.helpers import fit_catboost_with_external_selection

# Shared CatBoost tuning helpers reused by entrypoints
# Read src/modeling/README.md or tune_component_catboost.py first for workflow order


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
