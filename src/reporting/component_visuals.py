import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.paths import OUTPUTS_DIR, PROJECT_ROOT

# -----------------------------------------------------------------------------
# Paths and artifact names
# -----------------------------------------------------------------------------
DEFAULT_FIGURE_DIR = PROJECT_ROOT / 'docs' / 'figures' / 'component_models'
OFFICIAL_SUMMARY_CSV = 'component_official_benchmark_summary.csv'
SINGLE_TEXT_MANIFEST = 'component_textwave2b_calibration_manifest.json'
SINGLE_CLASS_METRICS = 'component_single_label_textwave2b_class_metrics.csv'
SINGLE_CONFUSION = 'component_single_label_textwave2b_confusion_major.csv'
SINGLE_CALIBRATION = 'component_single_label_textwave2b_calibration.csv'
MULTI_LABEL_METRICS = 'component_multilabel_label_metrics.csv'
TARGET_SCOPE_SUMMARY = 'component_target_scope_summary.csv'

FIGURE_INDEX = 'component_model_figure_index.csv'

FIGURE_STYLE = {
    'single': '#1f77b4',
    'multi': '#ff7f0e',
    'baseline': '#9aa0a6',
    'promoted': '#2ca02c',
    'accent': '#d62728',
    'grid': '#e8e8e8'
}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def read_csv(outputs_dir, name):
    path = Path(outputs_dir) / name
    if not path.exists():
        raise FileNotFoundError(f'Missing required component artifact: {path}')
    return pd.read_csv(path)


def read_json(outputs_dir, name):
    path = Path(outputs_dir) / name
    if not path.exists():
        raise FileNotFoundError(f'Missing required component artifact: {path}')
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def clean_label(value):
    text = '' if pd.isna(value) else str(value).strip()
    return text if text else 'Other / tail'


def figure_path(output_dir, name):
    return Path(output_dir) / f'{name}.png'


def display_path(path):
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def save_figure(fig, output_dir, name, title, description):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = figure_path(output_dir, name)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return {
        'figure': name,
        'path': display_path(path),
        'title': title,
        'description': description
    }


def setup_axes(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc='left', fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis='x', color=FIGURE_STYLE['grid'], linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_horizontal_bars(ax, labels, values, color, label_values=True):
    positions = np.arange(len(labels))
    ax.barh(positions, values, color=color)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.0, float(np.nanmax(values)) * 1.08 if len(values) else 1.0))

    if label_values:
        for idx, value in enumerate(values):
            ax.text(
                value + 0.01,
                idx,
                f'{value:.3f}',
                va='center',
                fontsize=8
            )


def write_index(rows, output_dir):
    output_dir = Path(output_dir)
    index_path = output_dir / FIGURE_INDEX
    pd.DataFrame(rows).to_csv(index_path, index=False)
    return index_path


# -----------------------------------------------------------------------------
# Figure builders
# -----------------------------------------------------------------------------
def plot_official_summary(summary_df, output_dir):
    single = summary_df.loc[summary_df['task'].eq('single_label_component')].iloc[0]
    multi = summary_df.loc[summary_df['task'].eq('multi_label_component_routing')].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    single_metrics = pd.Series(
        {
            'Macro F1': float(single['macro_f1']),
            'Top-1 Accuracy': float(single['top_1_accuracy']),
            'Top-3 Accuracy': float(single['top_3_accuracy']),
            'Calibration ECE': float(single['ece'])
        }
    )
    multi_metrics = pd.Series(
        {
            'Macro F1': float(multi['macro_f1']),
            'Micro F1': float(multi['micro_f1']),
            'Recall@3': float(multi['recall_at_3']),
            'Precision@3': float(multi['precision_at_3'])
        }
    )

    plot_horizontal_bars(
        axes[0],
        single_metrics.index.to_list(),
        single_metrics.to_numpy(),
        FIGURE_STYLE['single']
    )
    setup_axes(
        axes[0],
        'Official Single-Label Component Benchmark',
        xlabel='Holdout metric'
    )

    plot_horizontal_bars(
        axes[1],
        multi_metrics.index.to_list(),
        multi_metrics.to_numpy(),
        FIGURE_STYLE['multi']
    )
    setup_axes(
        axes[1],
        'Official Multi-Label Routing Benchmark',
        xlabel='Holdout metric'
    )

    fig.suptitle('Final Component Model Benchmarks', fontsize=16, fontweight='bold', x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save_figure(
        fig,
        output_dir,
        'component_official_benchmark_summary',
        'Final Component Model Benchmarks',
        'Official holdout metrics for the locked single-label and multi-label component models'
    )


def plot_single_lift(manifest, output_dir):
    baseline = manifest['locked_holdout_baseline']
    promoted = manifest['calibrated_holdout_metrics']
    metrics = ['macro_f1', 'top_1_accuracy', 'top_3_accuracy']
    labels = ['Macro F1', 'Top-1 Accuracy', 'Top-3 Accuracy']

    baseline_values = [float(baseline[metric]) for metric in metrics]
    promoted_values = [float(promoted[metric]) for metric in metrics]
    positions = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        positions - width / 2,
        baseline_values,
        width,
        label='Structured CatBoost baseline',
        color=FIGURE_STYLE['baseline']
    )
    ax.bar(
        positions + width / 2,
        promoted_values,
        width,
        label='Calibrated text + structured fusion',
        color=FIGURE_STYLE['promoted']
    )

    for idx, value in enumerate(promoted_values):
        lift = value - baseline_values[idx]
        ax.text(
            idx + width / 2,
            value + 0.02,
            f'+{lift:.3f}',
            ha='center',
            fontsize=9,
            fontweight='bold'
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc='upper left')
    setup_axes(
        ax,
        'Single-Label Model Lift From Complaint Narrative Text',
        xlabel=None,
        ylabel='Holdout metric'
    )
    return save_figure(
        fig,
        output_dir,
        'component_single_label_model_lift',
        'Single-Label Model Lift From Complaint Narrative Text',
        'Holdout metric lift from structured CatBoost to calibrated text plus structured late fusion'
    )


def plot_single_class_f1(class_df, output_dir):
    class_df = class_df.copy()
    class_df['component_group'] = class_df['component_group'].map(clean_label)
    class_df = class_df.sort_values(['f1', 'support'], ascending=[True, True])

    fig_height = max(6, len(class_df) * 0.34)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    plot_horizontal_bars(
        ax,
        class_df['component_group'].to_list(),
        class_df['f1'].astype(float).to_numpy(),
        FIGURE_STYLE['single']
    )
    setup_axes(
        ax,
        'Single-Label Holdout F1 By Component Group',
        xlabel='F1 score'
    )
    return save_figure(
        fig,
        output_dir,
        'component_single_label_class_f1',
        'Single-Label Holdout F1 By Component Group',
        'Per-class F1 for the promoted calibrated single-label component model'
    )


def plot_multilabel_class_f1(label_df, output_dir):
    label_df = label_df.copy()
    if 'model' in label_df.columns:
        label_df = label_df.loc[label_df['model'].eq('CatBoost MultiLabel')]
    label_df['component_group'] = label_df['component_group'].map(clean_label)
    label_df = label_df.sort_values(['f1', 'support'], ascending=[True, True])

    fig_height = max(7, len(label_df) * 0.30)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    plot_horizontal_bars(
        ax,
        label_df['component_group'].to_list(),
        label_df['f1'].astype(float).to_numpy(),
        FIGURE_STYLE['multi']
    )
    setup_axes(
        ax,
        'Multi-Label Routing Holdout F1 By Component Group',
        xlabel='F1 score'
    )
    return save_figure(
        fig,
        output_dir,
        'component_multilabel_class_f1',
        'Multi-Label Routing Holdout F1 By Component Group',
        'Per-label F1 for the official structured CatBoost multi-label routing model'
    )


def plot_single_calibration(calibration_df, output_dir):
    bins = calibration_df.loc[calibration_df['section'].eq('bin')].copy()
    bins = bins.loc[bins['count'].fillna(0).astype(float).gt(0)]
    bins['accuracy'] = bins['accuracy'].astype(float)
    bins['avg_confidence'] = bins['avg_confidence'].astype(float)
    bins['share'] = bins['share'].astype(float)

    overall = calibration_df.loc[calibration_df['section'].eq('overall')].iloc[0]

    fig, ax = plt.subplots(figsize=(7.5, 7))
    sizes = 4000 * bins['share'].to_numpy()
    ax.scatter(
        bins['avg_confidence'],
        bins['accuracy'],
        s=np.maximum(sizes, 35),
        alpha=0.72,
        color=FIGURE_STYLE['promoted'],
        edgecolor='white',
        linewidth=0.7
    )
    ax.plot([0, 1], [0, 1], linestyle='--', color='#555555', linewidth=1)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.text(
        0.03,
        0.92,
        f"ECE = {float(overall['ece']):.4f}\nBrier = {float(overall['multiclass_brier']):.4f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'edgecolor': '#cccccc'}
    )
    setup_axes(
        ax,
        'Single-Label Calibration After Power Scaling',
        xlabel='Average predicted confidence',
        ylabel='Observed accuracy'
    )
    return save_figure(
        fig,
        output_dir,
        'component_single_label_calibration',
        'Single-Label Calibration After Power Scaling',
        'Reliability plot for the calibrated single-label text plus structured fusion model'
    )


def plot_single_confusion(confusion_df, output_dir):
    confusion_df = confusion_df.copy()
    confusion_df['true_group'] = confusion_df['true_group'].map(clean_label)
    confusion_df['pred_group'] = confusion_df['pred_group'].map(clean_label)
    pivot = confusion_df.pivot_table(
        index='true_group',
        columns='pred_group',
        values='row_share',
        aggfunc='sum',
        fill_value=0.0
    )

    totals = confusion_df.groupby('true_group')['count'].sum().sort_values(ascending=False)
    ordered = totals.index.to_list()
    pivot = pivot.reindex(index=ordered)
    pivot = pivot.reindex(columns=[col for col in ordered if col in pivot.columns])

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel('Predicted component group')
    ax.set_ylabel('True component group')
    ax.set_title('Single-Label Major-Group Confusion Pattern', loc='left', fontsize=14, fontweight='bold')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label='Row share')
    fig.tight_layout()
    return save_figure(
        fig,
        output_dir,
        'component_single_label_confusion_major',
        'Single-Label Major-Group Confusion Pattern',
        'Row-normalized confusion heatmap for major single-label component groups'
    )


def plot_target_scope(scope_df, output_dir):
    overall = scope_df.loc[scope_df['scope'].eq('overall')].copy()
    keep_segments = [
        'single_label_benchmark_cases',
        'multi_label_only_cases'
    ]
    overall = overall.loc[overall['segment'].isin(keep_segments)]
    labels = ['Single-label benchmark', 'Multi-label only']

    fig, ax1 = plt.subplots(figsize=(9, 5))
    bars = ax1.bar(
        labels,
        overall['cases'].astype(int),
        color=[FIGURE_STYLE['single'], FIGURE_STYLE['multi']]
    )
    ax1.set_ylabel('Cases')
    ax1.set_title('Component Target Scope And Excluded Multi-Label Cases', loc='left', fontsize=14, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for bar, share in zip(bars, overall['case_share'].astype(float), strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{int(bar.get_height()):,}\n{share:.1%}',
            ha='center',
            va='bottom',
            fontsize=9
        )

    ax2 = ax1.twinx()
    ax2.plot(
        labels,
        overall['severity_broad_rate'].astype(float),
        color=FIGURE_STYLE['accent'],
        marker='o',
        linewidth=2,
        label='Broad severity rate'
    )
    ax2.set_ylabel('Broad severity rate')
    ax2.set_ylim(0, max(0.15, float(overall['severity_broad_rate'].max()) * 1.5))
    ax2.spines['top'].set_visible(False)

    fig.tight_layout()
    return save_figure(
        fig,
        output_dir,
        'component_target_scope',
        'Component Target Scope And Excluded Multi-Label Cases',
        'Single-label benchmark coverage versus multi-label-only cases and severity rate'
    )


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def generate_component_visuals(outputs_dir=OUTPUTS_DIR, output_dir=DEFAULT_FIGURE_DIR):
    outputs_dir = Path(outputs_dir)
    output_dir = Path(output_dir)

    summary_df = read_csv(outputs_dir, OFFICIAL_SUMMARY_CSV)
    single_manifest = read_json(outputs_dir, SINGLE_TEXT_MANIFEST)
    single_class_df = read_csv(outputs_dir, SINGLE_CLASS_METRICS)
    single_confusion_df = read_csv(outputs_dir, SINGLE_CONFUSION)
    single_calibration_df = read_csv(outputs_dir, SINGLE_CALIBRATION)
    multilabel_df = read_csv(outputs_dir, MULTI_LABEL_METRICS)
    target_scope_df = read_csv(outputs_dir, TARGET_SCOPE_SUMMARY)

    rows = [
        plot_official_summary(summary_df, output_dir),
        plot_single_lift(single_manifest, output_dir),
        plot_single_class_f1(single_class_df, output_dir),
        plot_single_calibration(single_calibration_df, output_dir),
        plot_single_confusion(single_confusion_df, output_dir),
        plot_multilabel_class_f1(multilabel_df, output_dir),
        plot_target_scope(target_scope_df, output_dir)
    ]
    index_path = write_index(rows, output_dir)
    return {
        'output_dir': output_dir,
        'index_path': index_path,
        'figures': rows
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate presentation-ready component model figures from saved artifacts'
    )
    parser.add_argument('--outputs-dir', default=str(OUTPUTS_DIR))
    parser.add_argument('--output-dir', default=str(DEFAULT_FIGURE_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    result = generate_component_visuals(
        outputs_dir=args.outputs_dir,
        output_dir=args.output_dir
    )

    print(f"[write] {result['output_dir']}")
    for row in result['figures']:
        print(f"[figure] {row['path']}")
    print(f"[index] {result['index_path']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
