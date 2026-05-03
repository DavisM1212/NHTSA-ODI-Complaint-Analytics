import argparse
import json
import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from matplotlib.patches import Patch

from src.config.contracts import (
    NLP_EARLY_WARNING_OFFICIAL_MANIFEST,
    NLP_EARLY_WARNING_RECURRING_SIGNALS,
    NLP_EARLY_WARNING_RISK_MONITOR,
    NLP_EARLY_WARNING_TOPIC_LIBRARY,
    NLP_EARLY_WARNING_WATCHLIST,
    NLP_EARLY_WARNING_WATCHLIST_SUMMARY,
)
from src.config.paths import OUTPUTS_DIR, PROJECT_ROOT

# -----------------------------------------------------------------------------
# Artifact names
# -----------------------------------------------------------------------------
DEFAULT_FIGURE_DIR = PROJECT_ROOT / 'docs' / 'figures' / 'nlp_early_warning'
OFFICIAL_MANIFEST = NLP_EARLY_WARNING_OFFICIAL_MANIFEST
TOPIC_LIBRARY = NLP_EARLY_WARNING_TOPIC_LIBRARY
WATCHLIST = NLP_EARLY_WARNING_WATCHLIST
WATCHLIST_SUMMARY = NLP_EARLY_WARNING_WATCHLIST_SUMMARY
RISK_MONITOR = NLP_EARLY_WARNING_RISK_MONITOR
RECURRING_SIGNALS = NLP_EARLY_WARNING_RECURRING_SIGNALS

FIGURE_INDEX = 'nlp_early_warning_figure_index.csv'

FIGURE_STYLE = {
    'high': '#c43c39',
    'moderate': '#f28e2b',
    'early': '#4e79a7',
    'risk': '#7b8ea3',
    'development': '#9aa0a6',
    'forward': '#1f77b4',
    'accent': '#1b9e77',
    'grid': '#e8e8e8',
    'line': '#3f3f46'
}

TIER_COLORS = {
    'High-confidence signal': FIGURE_STYLE['high'],
    'Moderate signal': FIGURE_STYLE['moderate'],
    'Early signal': FIGURE_STYLE['early']
}

TIER_ORDER = [
    'High-confidence signal',
    'Moderate signal',
    'Early signal'
]

COMPONENT_GROUP_LABELS = {
    'ELECTRICAL SYSTEM': 'Electrical',
    'POWER TRAIN': 'Powertrain',
    'ENGINE / COOLING': 'Engine/Cooling',
    'VEHICLE SPEED CONTROL': 'Speed Control',
    'FUEL / PROPULSION': 'Fuel/Propulsion',
    'SERVICE BRAKES': 'Brakes',
    'STEERING': 'Steering',
    'BACK OVER PREVENTION': 'Backup Safety',
    'STRUCTURE': 'Structure',
    'VISIBILITY / WIPER': 'Visibility/Wiper',
    'FORWARD COLLISION AVOIDANCE': 'Fwd Collision Avoid',
    'TIRES': 'Tires'
}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def read_csv(outputs_dir, name, parse_dates=None):
    path = Path(outputs_dir) / name
    if not path.exists():
        raise FileNotFoundError(f'Missing required NLP artifact: {path}')
    return pd.read_csv(path, parse_dates=parse_dates)


def read_json(outputs_dir, name):
    path = Path(outputs_dir) / name
    if not path.exists():
        raise FileNotFoundError(f'Missing required NLP artifact: {path}')
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


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


def setup_axes(ax, title, xlabel=None, ylabel=None, grid_axis='x'):
    ax.set_title(title, loc='left', fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis=grid_axis, color=FIGURE_STYLE['grid'], linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def write_index(rows, output_dir):
    output_dir = Path(output_dir)
    index_path = output_dir / FIGURE_INDEX
    pd.DataFrame(rows).to_csv(index_path, index=False)
    return index_path


def wrap_text(text, width=42):
    text = '' if pd.isna(text) else str(text).strip()
    if not text:
        return ''
    return '\n'.join(
        textwrap.wrap(
            text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False
        )
    )


def wrap_text_limited(text, width=42, max_lines=2):
    text = '' if pd.isna(text) else str(text).strip()
    if not text:
        return ''
    lines = textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False
    )
    if len(lines) <= max_lines:
        return '\n'.join(lines)
    kept = lines[:max_lines]
    final_line = kept[-1].rstrip()
    if len(final_line) > 3:
        final_line = final_line[:-3].rstrip()
    kept[-1] = f'{final_line}...'
    return '\n'.join(kept)


def short_text(text, limit=40):
    text = '' if pd.isna(text) else str(text).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3].rstrip()}..."


def month_label(value):
    return pd.Timestamp(value).strftime('%Y-%m')


def pct_text(value, decimals=1):
    return f'{100 * float(value):.{decimals}f}%'


def uniform_bar_positions(count, step=2.18):
    if count <= 0:
        return np.array([], dtype=float)
    return np.arange(count, dtype=float) * step


def compact_component_groups(text):
    text = '' if pd.isna(text) else str(text).strip()
    if not text:
        return ''
    groups = [part.strip() for part in text.split('|') if str(part).strip()]
    display_groups = [COMPONENT_GROUP_LABELS.get(group, group.title()) for group in groups]
    return ' + '.join(display_groups)


def compact_topic_label(text):
    text = '' if pd.isna(text) else str(text).strip()
    replacements = {
        ' issue': '',
        ' complaints': '',
        ' cluster': ''
    }
    for suffix, repl in replacements.items():
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)] + repl
            break
    return text


def build_summary_label_parts(row):
    header = f"{row['maketxt']} {row['modeltxt']} {row['yeartxt']}"
    component_text = compact_component_groups(row.get('component_groups'))
    topic_text = compact_topic_label(row['topic_label'])
    return {
        'header': wrap_text_limited(header, width=22, max_lines=1),
        'component': wrap_text_limited(component_text, width=35, max_lines=2) if component_text else '',
        'topic': wrap_text_limited(topic_text, width=35, max_lines=2)
    }


def build_recurring_callout(row):
    header = f"{row['maketxt']} {row['modeltxt']} {row['yeartxt']}"
    topic_text = compact_topic_label(row['topic_label'])
    return f"{header}\n{wrap_text_limited(topic_text, width=28, max_lines=2)}"


def jitter_by_group(values, spread=0.16):
    series = pd.Series(values).reset_index(drop=True)
    offsets = np.zeros(len(series), dtype=float)
    for value in sorted(series.unique()):
        idx = np.flatnonzero(series.to_numpy() == value)
        if len(idx) == 1:
            continue
        offsets[idx] = np.linspace(-spread, spread, len(idx))
    return offsets


def spread_label_positions(values, min_gap=1.7):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    adjusted = np.sort(values.copy())
    for idx in range(1, len(adjusted)):
        if adjusted[idx] - adjusted[idx - 1] < min_gap:
            adjusted[idx] = adjusted[idx - 1] + min_gap
    overflow = adjusted[-1] - values.max()
    if overflow > 0:
        adjusted -= overflow * 0.55
    floor = max(0.8, values.min() - 0.4)
    if adjusted[0] < floor:
        adjusted += floor - adjusted[0]
    return adjusted


def build_tier_legend():
    return [
        Patch(facecolor=TIER_COLORS[tier], label=tier)
        for tier in TIER_ORDER
    ]


def development_forward_labels(manifest):
    time_windows = manifest.get('time_windows', {})
    development_end = pd.Timestamp(time_windows.get('development_end', '2024-12-31'))
    forward_start = pd.Timestamp(time_windows.get('forward_start', '2025-01-01'))
    return (
        f"Development share through {development_end.year}",
        f"Forward share from {forward_start.year}"
    )


def add_leaderboard_label(ax, y_value, label_parts):
    children = [
        TextArea(
            label_parts['header'],
            textprops={
                'fontsize': 8.6,
                'fontweight': 'bold',
                'color': '#222222',
                'ha': 'right',
                'va': 'center'
            }
        )
    ]
    if label_parts['component']:
        children.append(
            TextArea(
                label_parts['component'],
                textprops={
                    'fontsize': 7.7,
                    'color': '#4b5563',
                    'ha': 'right',
                    'va': 'center'
                }
            )
        )
    children.append(
        TextArea(
            label_parts['topic'],
            textprops={
                'fontsize': 7.9,
                'color': '#222222',
                'ha': 'right',
                'va': 'center'
            }
        )
    )
    label_box = VPacker(children=children, align='right', pad=0, sep=0.6)
    annotation = AnnotationBbox(
        label_box,
        (0, y_value),
        xybox=(-4, 0),
        xycoords='data',
        boxcoords='offset points',
        box_alignment=(1, 0.5),
        frameon=False,
        pad=0,
        annotation_clip=False
    )
    ax.add_artist(annotation)


# -----------------------------------------------------------------------------
# Figure builders
# -----------------------------------------------------------------------------
def plot_latest_leaderboard(summary_df, watchlist_df, output_dir):
    latest_month = summary_df['month'].max()
    latest_summary = summary_df.loc[summary_df['month'].eq(latest_month)].copy()
    latest_watchlist = watchlist_df.loc[watchlist_df['month'].eq(latest_month)].copy()

    plot_df = latest_summary.sort_values(
        ['max_component_watchlist_score', 'complaints'],
        ascending=[False, False]
    ).head(12).copy()
    plot_df = plot_df.sort_values('complaints', ascending=True)
    plot_df['label_parts'] = plot_df.apply(build_summary_label_parts, axis=1)
    plot_df['color'] = plot_df['best_signal_tier'].map(TIER_COLORS).fillna(FIGURE_STYLE['early'])
    y_positions = uniform_bar_positions(len(plot_df), step=1.75)
    fig_height = max(9.4, float(y_positions[-1]) * 0.52 + 2.9 if len(y_positions) else 9.4)

    fig, ax = plt.subplots(figsize=(15.0, fig_height))
    bars = ax.barh(
        y_positions,
        plot_df['complaints'].astype(float),
        height=0.92,
        color=plot_df['color']
    )
    ax.set_yticks([])
    ax.tick_params(axis='y', length=0)

    xmax = float(plot_df['complaints'].max()) * 1.24 if not plot_df.empty else 1.0
    ax.set_xlim(0, max(1.0, xmax))
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        x_value = float(row['complaints'])
        label = f"{int(row['complaints'])} complaints | score {float(row['max_component_watchlist_score']):.1f} | {int(row['unique_states'])} states"
        ax.text(x_value + 0.4, bar.get_y() + bar.get_height() / 2, label, va='center', fontsize=8.5)
        add_leaderboard_label(ax, bar.get_y() + bar.get_height() / 2, row['label_parts'])
    if len(y_positions):
        ax.set_ylim(-1.0, y_positions[-1] + 1.1)

    setup_axes(
        ax,
        f'Latest month: {month_label(latest_month)}',
        xlabel='Rolled-up complaints',
        grid_axis='x'
    )
    ax.legend(
        handles=build_tier_legend(),
        loc='lower right',
        frameon=False,
        fontsize=9
    )

    fig.suptitle('Current Emerging Signal Leaderboard', fontsize=16, fontweight='bold', x=0.02, ha='left')
    fig.text(
        0.02,
        0.95,
        f"Top 12 cohorts out of {len(latest_summary):,} rolled-up cohort-topic signals and {len(latest_watchlist):,} monthly watchlist rows",
        fontsize=10,
        color=FIGURE_STYLE['line']
    )
    fig.subplots_adjust(left=0.14, right=0.985, top=0.92, bottom=0.06)
    return save_figure(
        fig,
        output_dir,
        'nlp_watchlist_latest_leaderboard',
        'Current Emerging Signal Leaderboard',
        'Top latest-month cohort-topic watchlist signals ranked by rolled-up complaints and colored by signal tier'
    )


def plot_latest_topic_mix(summary_df, output_dir):
    latest_month = summary_df['month'].max()
    latest_summary = summary_df.loc[summary_df['month'].eq(latest_month)].copy()

    plot_df = latest_summary.groupby('topic_label', as_index=False).agg(
        complaints=('complaints', 'sum'),
        flagged_cohorts=('topic_id', 'size')
    )
    plot_df = plot_df.sort_values('complaints', ascending=False).head(10).copy()
    plot_df = plot_df.sort_values('complaints', ascending=True)
    plot_df['label'] = plot_df['topic_label'].map(lambda value: wrap_text(value, width=34))

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    bars = ax.barh(plot_df['label'], plot_df['complaints'].astype(float), color=FIGURE_STYLE['accent'])

    xmax = float(plot_df['complaints'].max()) * 1.18 if not plot_df.empty else 1.0
    ax.set_xlim(0, max(1.0, xmax))
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        text = f"{int(row['flagged_cohorts'])} flagged cohorts"
        ax.text(float(row['complaints']) + 0.6, bar.get_y() + bar.get_height() / 2, text, va='center', fontsize=8.5)

    setup_axes(
        ax,
        f'Latest month: {month_label(latest_month)}',
        xlabel='Rolled-up complaints',
        grid_axis='x'
    )
    fig.suptitle('Latest Watchlist Topic Mix', fontsize=16, fontweight='bold', x=0.02, ha='left')
    fig.text(
        0.02,
        0.93,
        'Top issue themes by latest-month complaint volume',
        fontsize=10,
        color=FIGURE_STYLE['line']
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_figure(
        fig,
        output_dir,
        'nlp_watchlist_latest_topic_mix',
        'Latest Watchlist Topic Mix',
        'Top latest-month defect themes by rolled-up complaints, annotated with the number of flagged cohorts they drive'
    )


def plot_topic_share_shift(topic_library_df, manifest, output_dir):
    plot_df = topic_library_df.loc[
        topic_library_df['watchlist_group'].eq('defect_watchlist')
    ].copy()
    plot_df = plot_df.sort_values('forward_share', ascending=False).head(10).copy()
    plot_df = plot_df.sort_values('forward_share', ascending=True)
    plot_df['development_pct'] = 100 * plot_df['development_share'].astype(float)
    plot_df['forward_pct'] = 100 * plot_df['forward_share'].astype(float)
    plot_df['label'] = plot_df['topic_label'].map(lambda value: wrap_text(value, width=34))

    development_label, forward_label = development_forward_labels(manifest)
    y_positions = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        ax.plot(
            [row['development_pct'], row['forward_pct']],
            [idx, idx],
            color=FIGURE_STYLE['grid'],
            linewidth=2.0,
            zorder=1
        )

    ax.scatter(plot_df['development_pct'], y_positions, s=85, color=FIGURE_STYLE['development'], label=development_label, zorder=3)
    ax.scatter(plot_df['forward_pct'], y_positions, s=85, color=FIGURE_STYLE['forward'], label=forward_label, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df['label'])

    xmax = max(float(plot_df['development_pct'].max()), float(plot_df['forward_pct'].max())) * 1.23 if not plot_df.empty else 1.0
    ax.set_xlim(0, max(1.0, xmax))
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        change_text = f"{float(row['share_percent_change']):+,.1f}%"
        anchor = max(float(row['development_pct']), float(row['forward_pct']))
        ax.text(anchor + 0.15, idx, change_text, va='center', fontsize=8.5)

    setup_axes(
        ax,
        'Top Defect Watchlist Topics by Complaint Share Shift',
        xlabel='Complaint share (%)',
        grid_axis='x'
    )
    ax.legend(loc='lower right', frameon=False, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return save_figure(
        fig,
        output_dir,
        'nlp_watchlist_topic_share_shift',
        'Topic Share Shift Into The Forward Period',
        'Development-versus-forward complaint share shift for the top defect-watchlist topics in the locked NMF library'
    )


def plot_recurring_large_signals(recurring_df, output_dir):
    plot_df = recurring_df.sort_values(
        ['months_flagged', 'max_complaints', 'max_watchlist_score'],
        ascending=[False, False, False]
    ).head(10).copy()
    plot_df['color'] = plot_df['best_signal_tier'].map(TIER_COLORS).fillna(FIGURE_STYLE['early'])
    plot_df['bubble_size'] = 110 + 48 * plot_df['max_watchlist_score'].astype(float)
    plot_df['x_plot'] = plot_df['months_flagged'].astype(float) + jitter_by_group(plot_df['months_flagged'].astype(int), spread=0.18)

    fig, ax = plt.subplots(figsize=(13.8, 7.6))
    ax.scatter(
        plot_df['x_plot'].astype(float),
        plot_df['max_complaints'].astype(float),
        s=plot_df['bubble_size'].astype(float),
        c=plot_df['color'],
        alpha=0.80,
        edgecolors='white',
        linewidths=1.3,
        zorder=3
    )

    for months_flagged, group_df in plot_df.groupby('months_flagged', sort=True):
        group_df = group_df.sort_values('max_complaints').copy()
        y_targets = spread_label_positions(group_df['max_complaints'].astype(float).to_numpy(), min_gap=5)
        x_target = float(group_df['x_plot'].max()) + (0.12 if float(months_flagged) >= 16 else 0.18)
        for y_text, (_, row) in zip(y_targets, group_df.iterrows()):
            ax.annotate(
                build_recurring_callout(row),
                xy=(float(row['x_plot']), float(row['max_complaints'])),
                xytext=(x_target, float(y_text)),
                textcoords='data',
                ha='left',
                va='center',
                fontsize=7.9,
                arrowprops={
                    'arrowstyle': '-',
                    'color': '#cbd5e1',
                    'linewidth': 0.9,
                    'shrinkA': 5,
                    'shrinkB': 5,
                    'alpha': 0.95
                }
            )

    months_min = float(plot_df['months_flagged'].min()) if not plot_df.empty else 0.0
    months_max = float(plot_df['months_flagged'].max()) if not plot_df.empty else 1.0
    complaints_max = float(plot_df['max_complaints'].max()) if not plot_df.empty else 1.0
    ax.set_xlim(months_min - 0.5, months_max + 0.95)
    ax.set_xticks(sorted(plot_df['months_flagged'].astype(int).unique()))
    ax.set_ylim(0, complaints_max * 1.18)
    setup_axes(
        ax,
        'Top Recurring Cohort-Topic Signals',
        xlabel='Months flagged',
        ylabel='Maximum complaints in a flagged month',
        grid_axis='both'
    )
    bubble_handles = build_tier_legend() + [
        Line2D([0], [0], marker='o', linestyle='', color='w', markerfacecolor='#6b7280', markeredgecolor='white', markersize=8, label='Bubble size = max watchlist score')
    ]
    ax.legend(
        handles=bubble_handles,
        loc='upper right',
        frameon=False,
        fontsize=9
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return save_figure(
        fig,
        output_dir,
        'nlp_watchlist_recurring_large_signals',
        'Recurring Large Signals Across Many Months',
        'Top recurring cohort-topic signals shown as a bubble chart where x is months flagged, y is peak flagged-month complaints, and bubble size is max watchlist score'
    )


def plot_recent_signal_flow(summary_df, risk_df, output_dir):
    latest_month = summary_df['month'].max()
    month_range = pd.date_range(latest_month - pd.DateOffset(months=13), latest_month, freq='MS')

    recent_summary = summary_df.loc[summary_df['month'].isin(month_range)].copy()
    recent_risk = risk_df.loc[risk_df['month'].isin(month_range)].copy()

    tier_counts = recent_summary.groupby(['month', 'best_signal_tier']).size().unstack(fill_value=0)
    tier_counts = tier_counts.reindex(month_range, fill_value=0)
    for tier in TIER_ORDER:
        if tier not in tier_counts.columns:
            tier_counts[tier] = 0
    tier_counts = tier_counts[TIER_ORDER]

    summary_complaints = recent_summary.groupby('month')['complaints'].sum().reindex(month_range, fill_value=0)
    risk_complaints = recent_risk.groupby('month')['complaints'].sum().reindex(month_range, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))
    positions = np.arange(len(month_range))

    bottom = np.zeros(len(month_range))
    for tier in reversed(TIER_ORDER):
        values = tier_counts[tier].to_numpy(dtype=float)
        axes[0].bar(
            positions,
            values,
            bottom=bottom,
            color=TIER_COLORS[tier],
            label=tier
        )
        bottom += values

    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([month_label(value) for value in month_range], rotation=45, ha='right')
    setup_axes(
        axes[0],
        'Recent flagged summary rows by signal tier',
        ylabel='Flagged summary rows',
        grid_axis='y'
    )
    axes[0].legend(loc='upper right', frameon=False, fontsize=8.5)

    axes[1].plot(
        positions,
        summary_complaints.to_numpy(dtype=float),
        color=FIGURE_STYLE['accent'],
        linewidth=2.2,
        marker='o',
        label='Emerging watchlist summary'
    )
    axes[1].plot(
        positions,
        risk_complaints.to_numpy(dtype=float),
        color=FIGURE_STYLE['risk'],
        linewidth=2.2,
        marker='o',
        label='Risk monitor'
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([month_label(value) for value in month_range], rotation=45, ha='right')
    setup_axes(
        axes[1],
        'Recent complaint volume in emerging vs risk outputs',
        ylabel='Rolled-up complaints',
        grid_axis='y'
    )
    axes[1].legend(loc='upper right', frameon=False, fontsize=8.5)

    fig.suptitle('Recent Signal Trend In The Official NLP Watchlist', fontsize=16, fontweight='bold', x=0.02, ha='left')
    fig.text(
        0.02,
        0.92,
        f"Previous 14 months through {month_label(latest_month)} showing how the main emerging watchlist and the persistent risk monitor move over time",
        fontsize=10,
        color=FIGURE_STYLE['line']
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_figure(
        fig,
        output_dir,
        'nlp_watchlist_recent_signal_flow',
        'Recent Signal Trend In The Official NLP Watchlist',
        'Recent 14-month cadence of emerging watchlist rows by tier alongside complaint volume in the summary watchlist and risk monitor'
    )


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def generate_watchlist_visuals(outputs_dir=OUTPUTS_DIR, output_dir=DEFAULT_FIGURE_DIR):
    manifest = read_json(outputs_dir, OFFICIAL_MANIFEST)
    topic_library_df = read_csv(outputs_dir, TOPIC_LIBRARY)
    watchlist_df = read_csv(outputs_dir, WATCHLIST, parse_dates=['month'])
    summary_df = read_csv(outputs_dir, WATCHLIST_SUMMARY, parse_dates=['month'])
    risk_df = read_csv(outputs_dir, RISK_MONITOR, parse_dates=['month'])
    recurring_df = read_csv(outputs_dir, RECURRING_SIGNALS, parse_dates=['first_month', 'latest_month'])

    rows = [
        plot_latest_leaderboard(summary_df, watchlist_df, output_dir),
        plot_latest_topic_mix(summary_df, output_dir),
        plot_topic_share_shift(topic_library_df, manifest, output_dir),
        plot_recurring_large_signals(recurring_df, output_dir),
        plot_recent_signal_flow(summary_df, risk_df, output_dir)
    ]

    index_path = write_index(rows, output_dir)
    return {
        'output_dir': Path(output_dir),
        'index_path': index_path,
        'figures': rows
    }


def main():
    parser = argparse.ArgumentParser(description="Generate official NLP early-warning watchlist figures")
    parser.add_argument('--outputs-dir', default=str(OUTPUTS_DIR), help='Directory with official NLP output artifacts')
    parser.add_argument('--output-dir', default=str(DEFAULT_FIGURE_DIR), help='Directory where figure PNGs should be written')
    args = parser.parse_args()

    result = generate_watchlist_visuals(
        outputs_dir=args.outputs_dir,
        output_dir=args.output_dir
    )

    print(f"[write] {display_path(result['output_dir'])}")
    for row in result['figures']:
        print(f"[figure] {row['figure']} -> {row['path']}")
    print(f"[index] {display_path(result['index_path'])}")


if __name__ == '__main__':
    main()
