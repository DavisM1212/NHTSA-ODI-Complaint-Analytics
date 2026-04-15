import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.config import settings
from src.config.paths import OUTPUTS_DIR, ensure_project_directories
from src.data.io_utils import load_frame, write_json
from src.features.component_text_sidecar import SIDECAR_STEM
from src.modeling.component_common import (
    FEATURE_WAVE1_SPLIT_MODE,
    SINGLE_INPUT_STEM,
    TARGET_COL,
    build_multiclass_calibration_df,
    build_multiclass_class_df,
    build_multiclass_confusion_df,
    feature_manifest,
    prep_single_label_cases,
    score_multiclass_from_proba,
    split_single_label_cases_by_mode,
)
from src.modeling.component_text_shared import (
    FINAL_LINEAR_MODEL_CHOICES,
    FINAL_LINEAR_MODEL_DEFAULT,
    LATE_FUSION_FAMILY,
    LOCKED_SINGLE_MANIFEST,
    LOCKED_SINGLE_SELECTION,
    SINGLE_ECE_WORSE_LIMIT,
    SINGLE_PROMOTE_HOLDOUT_DELTA,
    SINGLE_TOP3_DROP_LIMIT,
    STRUCTURED_FEATURE_SET,
    TEXT_ONLY_FAMILY,
    apply_single_fusion_weight,
    build_single_overlap_rows,
    build_single_row,
    fit_single_structured_family,
    fit_single_structured_holdout,
    fit_single_text_family,
    load_json,
    log_line,
    merge_text_sidecar,
    read_locked_single_ece,
)

# Workflow owner for post-Wave 2 single-label calibration
# Runs after component_text_wave2.py and writes the calibrated holdout artifacts

WAVE2_MANIFEST = OUTPUTS_DIR / 'component_textwave2_manifest.json'
GLOBAL_MANIFEST_NAME = 'component_textwave2b_calibration_manifest.json'
SELECT_GRID_NAME = 'component_single_label_textwave2b_calibration_select_grid.csv'
HOLDOUT_NAME = 'component_single_label_textwave2b_calibrated_holdout.csv'
CALIBRATION_NAME = 'component_single_label_textwave2b_calibration.csv'
CLASS_NAME = 'component_single_label_textwave2b_class_metrics.csv'
CONFUSION_NAME = 'component_single_label_textwave2b_confusion_major.csv'

DEFAULT_ALPHA_GRID = [
    0.50,
    0.75,
    1.00,
    1.10,
    1.20,
    1.35,
    1.50,
    1.75,
    2.00,
    2.50,
    3.00,
    4.00,
    5.00,
]


def parse_float_list(value):
    values = []
    for part in str(value).split(','):
        stripped = part.strip()
        if not stripped:
            continue
        parsed = float(stripped)
        if parsed <= 0:
            raise argparse.ArgumentTypeError('Calibration alpha values must be positive')
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError('At least one calibration alpha is required')
    return values


def apply_power_calibration(proba, alpha):
    proba = np.asarray(proba, dtype=np.float64)
    proba = np.clip(proba, 1e-15, 1.0)
    powered = np.power(proba, float(alpha))
    row_sums = powered.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze(axis=1) <= 0
    if np.any(zero_rows):
        powered[zero_rows] = 1.0
        row_sums = powered.sum(axis=1, keepdims=True)
    return powered / row_sums


def calibration_overall(calibration_df):
    overall = calibration_df.loc[calibration_df['section'].eq('overall')]
    if overall.empty:
        raise ValueError('Calibration dataframe is missing its overall row')
    return overall.iloc[0].to_dict()


def build_calibration_candidate_row(y_true, proba, classes, alpha):
    calibrated = apply_power_calibration(proba, alpha)
    _, metrics = score_multiclass_from_proba(y_true, calibrated, classes)
    calibration_df = build_multiclass_calibration_df(y_true, calibrated, classes)
    overall = calibration_overall(calibration_df)
    return {
        'calibration_method': 'power',
        'calibration_alpha': float(alpha),
        'rows': int(len(y_true)),
        'top_1_accuracy': metrics['top_1_accuracy'],
        'macro_f1': metrics['macro_f1'],
        'top_3_accuracy': metrics['top_3_accuracy'],
        'ece': float(overall['ece']),
        'avg_confidence': float(overall['avg_confidence']),
        'accuracy': float(overall['accuracy']),
        'confidence_gap': float(overall['gap']),
        'multiclass_brier': float(overall['multiclass_brier']),
        'log_loss': float(log_loss(y_true, np.clip(calibrated, 1e-15, 1.0), labels=classes)),
    }


def select_calibration_alpha(y_true, proba, classes, alpha_grid):
    rows = [
        build_calibration_candidate_row(y_true, proba, classes, alpha)
        for alpha in alpha_grid
    ]
    candidate_df = pd.DataFrame(rows)
    candidate_df = candidate_df.sort_values(
        ['ece', 'multiclass_brier', 'log_loss', 'calibration_alpha'],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return candidate_df.iloc[0].to_dict(), candidate_df


def add_calibration_columns(row, method, alpha, source):
    row = dict(row)
    row['calibration_method'] = method
    row['calibration_alpha'] = alpha
    row['calibration_source'] = source
    return row


def build_holdout_row(input_path, text_sidecar_path, holdout_df, proba, classes, fit_seconds, selected_iteration, text_weight, method, alpha, source):
    row = build_single_row(
        'single_label',
        LATE_FUSION_FAMILY,
        input_path,
        text_sidecar_path,
        'final_holdout',
        'holdout_2026',
        holdout_df[TARGET_COL].astype(str),
        proba,
        classes,
        fit_seconds=fit_seconds,
        selected_iteration=selected_iteration,
        selected_text_weight=text_weight,
    )
    return add_calibration_columns(row, method, alpha, source)


def build_holdout_overlap_rows(base_row, y_true, proba, classes, overlap_mask):
    rows = build_single_overlap_rows(base_row, y_true, proba, classes, overlap_mask)
    return [
        add_calibration_columns(
            row,
            base_row['calibration_method'],
            base_row['calibration_alpha'],
            base_row['calibration_source'],
        )
        for row in rows
    ]


def load_wave2_single_config(fusion_weight_override=None):
    if not WAVE2_MANIFEST.exists():
        raise FileNotFoundError(f'Missing Wave 2 manifest: {WAVE2_MANIFEST}')
    manifest = load_json(WAVE2_MANIFEST)
    single_entry = manifest.get('tasks', {}).get('single_label', {})
    selected_family = single_entry.get('selected_family')
    if selected_family != LATE_FUSION_FAMILY:
        raise ValueError(
            f'Wave 2b calibration expects {LATE_FUSION_FAMILY}, '
            f'but Wave 2 selected {selected_family!r}'
        )

    fusion_weight = fusion_weight_override
    if fusion_weight is None:
        fusion_weight = single_entry.get('screen_fusion_weight')
    if fusion_weight is None:
        raise ValueError('Wave 2 manifest does not record screen_fusion_weight')

    return manifest, float(fusion_weight)


def load_locked_single_holdout():
    manifest = load_json(LOCKED_SINGLE_MANIFEST)
    return manifest['official_holdout_metrics']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calibrate Wave 2 single-label text_structured_late_fusion probabilities with select_2025'
    )
    parser.add_argument(
        '--task-type',
        choices=['CPU', 'GPU', 'cpu', 'gpu'],
        default='CPU',
        help='CatBoost processing target for the structured carry-forward branch',
    )
    parser.add_argument('--devices', default='0')
    parser.add_argument('--random-seed', type=int, default=settings.RANDOM_SEED)
    parser.add_argument('--single-input-path', default=None)
    parser.add_argument('--text-sidecar-path', default=None)
    parser.add_argument(
        '--final-linear-model',
        choices=FINAL_LINEAR_MODEL_CHOICES,
        default=FINAL_LINEAR_MODEL_DEFAULT,
    )
    parser.add_argument(
        '--alpha-grid',
        type=parse_float_list,
        default=DEFAULT_ALPHA_GRID,
        help='Comma-separated positive power-calibration alphas',
    )
    parser.add_argument(
        '--fusion-weight',
        type=float,
        default=None,
        help='Override the Wave 2 selected late-fusion text weight',
    )
    parser.add_argument('--skip-readme-update', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    wave2_manifest, text_weight = load_wave2_single_config(args.fusion_weight)
    structured_feature_info = feature_manifest(STRUCTURED_FEATURE_SET)
    locked_selection = load_json(LOCKED_SINGLE_SELECTION)
    locked_holdout = load_locked_single_holdout()
    locked_ece = read_locked_single_ece()

    raw_df, input_path = load_frame(SINGLE_INPUT_STEM, input_path=args.single_input_path)
    sidecar_df, text_sidecar_path = load_frame(SIDECAR_STEM, input_path=args.text_sidecar_path)
    case_df = prep_single_label_cases(raw_df, structured_feature_info['feature_cols'])
    case_df = merge_text_sidecar(case_df, sidecar_df)
    split_parts = split_single_label_cases_by_mode(case_df, split_mode=FEATURE_WAVE1_SPLIT_MODE)

    dev_screen_df = split_parts['dev_2020_2024']
    select_df = split_parts['select_2025']
    dev_select_df = split_parts['dev_2020_2025']
    holdout_df = split_parts['holdout_2026']

    log_line(
        f'[wave2b] rows | dev_screen={len(dev_screen_df):,} '
        f'select_2025={len(select_df):,} dev_select={len(dev_select_df):,} '
        f'holdout_2026={len(holdout_df):,}'
    )
    log_line(
        f'[wave2b] family={LATE_FUSION_FAMILY} text_weight={text_weight:.2f} '
        f'final_linear_model={args.final_linear_model}'
    )

    start_total = perf_counter()
    log_line('[wave2b] fitting select structured branch')
    structured_select = fit_single_structured_family(
        dev_screen_df,
        select_df,
        structured_feature_info,
        locked_selection['best_params'],
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed,
    )
    log_line(
        f'[wave2b] select structured fit_seconds={float(structured_select["fit_seconds"]):.2f} '
        f'iteration={int(structured_select["selected_iteration"])}'
    )

    log_line('[wave2b] fitting select text branch')
    text_select = fit_single_text_family(
        dev_screen_df,
        select_df,
        structured_feature_info,
        TEXT_ONLY_FAMILY,
        final_model=False,
    )
    log_line(f'[wave2b] select text fit_seconds={float(text_select["fit_seconds"]):.2f}')

    select_fusion = apply_single_fusion_weight(
        select_df[TARGET_COL].astype(str),
        text_select['eval_proba'],
        text_select['classes'],
        structured_select['valid_proba'],
        structured_select['classes'],
        text_weight=text_weight,
    )

    selected_alpha, select_grid_df = select_calibration_alpha(
        select_df[TARGET_COL].astype(str),
        select_fusion['proba'],
        select_fusion['classes'],
        args.alpha_grid,
    )
    select_grid_df.insert(0, 'stage', 'select_2025')
    select_grid_df.insert(1, 'split', 'select_2025')
    select_grid_df.insert(2, 'family_name', LATE_FUSION_FAMILY)
    select_grid_df.insert(3, 'selected_text_weight', text_weight)
    select_grid_df.to_csv(OUTPUTS_DIR / SELECT_GRID_NAME, index=False)
    log_line(
        f'[wave2b] selected alpha={float(selected_alpha["calibration_alpha"]):.3f} '
        f'select_ece={float(selected_alpha["ece"]):.4f}'
    )

    log_line('[wave2b] fitting holdout structured branch')
    structured_holdout = fit_single_structured_holdout(
        dev_select_df,
        holdout_df,
        structured_feature_info,
        locked_selection['best_params'],
        structured_select['selected_iteration'],
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed,
    )
    log_line(f'[wave2b] holdout structured fit_seconds={float(structured_holdout["fit_seconds"]):.2f}')

    log_line('[wave2b] fitting holdout text branch')
    text_holdout = fit_single_text_family(
        dev_select_df,
        holdout_df,
        structured_feature_info,
        TEXT_ONLY_FAMILY,
        final_model=True,
        final_model_kind=args.final_linear_model,
    )
    log_line(f'[wave2b] holdout text fit_seconds={float(text_holdout["fit_seconds"]):.2f}')

    holdout_fusion = apply_single_fusion_weight(
        holdout_df[TARGET_COL].astype(str),
        text_holdout['eval_proba'],
        text_holdout['classes'],
        structured_holdout['proba'],
        structured_holdout['classes'],
        text_weight=text_weight,
    )
    uncalibrated_proba = holdout_fusion['proba']
    calibrated_proba = apply_power_calibration(
        uncalibrated_proba,
        selected_alpha['calibration_alpha'],
    )
    fit_seconds = round(float(structured_holdout['fit_seconds']) + float(text_holdout['fit_seconds']), 2)
    uncalibrated_row = build_holdout_row(
        input_path,
        text_sidecar_path,
        holdout_df,
        uncalibrated_proba,
        holdout_fusion['classes'],
        fit_seconds,
        structured_select['selected_iteration'],
        text_weight,
        'none',
        1.0,
        'none',
    )
    calibrated_row = build_holdout_row(
        input_path,
        text_sidecar_path,
        holdout_df,
        calibrated_proba,
        holdout_fusion['classes'],
        fit_seconds,
        structured_select['selected_iteration'],
        text_weight,
        'power',
        selected_alpha['calibration_alpha'],
        'select_2025',
    )

    overlap_mask = pd.Series(holdout_df['cdescr_model_text']).fillna('').astype(str).isin(
        set(pd.Series(dev_select_df['cdescr_model_text']).fillna('').astype(str)) - {''}
    ).to_numpy()
    holdout_rows = [uncalibrated_row]
    holdout_rows.extend(
        build_holdout_overlap_rows(
            uncalibrated_row,
            holdout_df[TARGET_COL].astype(str),
            uncalibrated_proba,
            holdout_fusion['classes'],
            overlap_mask,
        )
    )
    holdout_rows.append(calibrated_row)
    holdout_rows.extend(
        build_holdout_overlap_rows(
            calibrated_row,
            holdout_df[TARGET_COL].astype(str),
            calibrated_proba,
            holdout_fusion['classes'],
            overlap_mask,
        )
    )
    holdout_df_out = pd.DataFrame(holdout_rows)
    holdout_df_out.to_csv(OUTPUTS_DIR / HOLDOUT_NAME, index=False)

    # score_multiclass_from_proba returns the prediction separately, rebuild it explicitly for clarity
    calibrated_pred, _ = score_multiclass_from_proba(
        holdout_df[TARGET_COL].astype(str),
        calibrated_proba,
        holdout_fusion['classes'],
    )
    class_df = build_multiclass_class_df(
        holdout_df[TARGET_COL].astype(str),
        calibrated_pred,
        holdout_fusion['classes'],
    )
    class_df.to_csv(OUTPUTS_DIR / CLASS_NAME, index=False)

    focus_groups = dev_select_df[TARGET_COL].value_counts().head(12).index.tolist()
    confusion_df = build_multiclass_confusion_df(
        holdout_df[TARGET_COL].astype(str),
        calibrated_pred,
        focus_groups,
    )
    confusion_df.to_csv(OUTPUTS_DIR / CONFUSION_NAME, index=False)

    calibration_df = build_multiclass_calibration_df(
        holdout_df[TARGET_COL].astype(str),
        calibrated_proba,
        holdout_fusion['classes'],
    )
    calibration_df.insert(0, 'calibration_method', 'power')
    calibration_df.insert(1, 'calibration_alpha', float(selected_alpha['calibration_alpha']))
    calibration_df.to_csv(OUTPUTS_DIR / CALIBRATION_NAME, index=False)

    holdout_cal_overall = calibration_overall(calibration_df)
    holdout_macro_gain = float(calibrated_row['macro_f1'] - locked_holdout['macro_f1'])
    holdout_top3_ok = calibrated_row['top_3_accuracy'] >= (
        locked_holdout['top_3_accuracy'] - SINGLE_TOP3_DROP_LIMIT
    )
    holdout_ece = float(holdout_cal_overall['ece'])
    holdout_ece_ok = (holdout_ece - locked_ece) <= SINGLE_ECE_WORSE_LIMIT
    promotion_status = 'promoted' if (
        holdout_macro_gain >= SINGLE_PROMOTE_HOLDOUT_DELTA
        and holdout_top3_ok
        and holdout_ece_ok
    ) else 'rejected_holdout'

    manifest = {
        'artifact_role': 'text_wave2b_calibration',
        'task': 'single_label',
        'family_name': LATE_FUSION_FAMILY,
        'split_mode': FEATURE_WAVE1_SPLIT_MODE,
        'public_benchmark_locked': True,
        'calibration_method': 'power',
        'calibration_source': 'select_2025',
        'selected_alpha': float(selected_alpha['calibration_alpha']),
        'alpha_grid': [float(alpha) for alpha in args.alpha_grid],
        'text_weight': text_weight,
        'structured_feature_set': STRUCTURED_FEATURE_SET,
        'final_linear_model': args.final_linear_model,
        'selected_iteration': int(structured_select['selected_iteration']),
        'locked_holdout_baseline': locked_holdout,
        'locked_holdout_ece': locked_ece,
        'uncalibrated_holdout_metrics': uncalibrated_row,
        'calibrated_holdout_metrics': calibrated_row,
        'holdout_ece': holdout_ece,
        'holdout_macro_gain': holdout_macro_gain,
        'holdout_top3_ok': bool(holdout_top3_ok),
        'holdout_ece_ok': bool(holdout_ece_ok),
        'promotion_status': promotion_status,
        'input_path': str(input_path),
        'text_sidecar_path': str(text_sidecar_path),
        'runtime_seconds': round(perf_counter() - start_total, 2),
        'wave2_manifest': str(WAVE2_MANIFEST),
        'wave2_manifest_selected_family': wave2_manifest.get('tasks', {}).get('single_label', {}).get('selected_family'),
        'artifacts': {
            'select_grid': str(OUTPUTS_DIR / SELECT_GRID_NAME),
            'holdout': str(OUTPUTS_DIR / HOLDOUT_NAME),
            'calibration': str(OUTPUTS_DIR / CALIBRATION_NAME),
            'class_metrics': str(OUTPUTS_DIR / CLASS_NAME),
            'confusion_major': str(OUTPUTS_DIR / CONFUSION_NAME),
        },
    }
    write_json(manifest, OUTPUTS_DIR / GLOBAL_MANIFEST_NAME)

    if not args.skip_readme_update:
        try:
            from src.reporting.update_component_readme import update_component_readme

            update_component_readme(single_manifest_path=OUTPUTS_DIR / GLOBAL_MANIFEST_NAME)
        except Exception as exc:
            print(f'[warn] README benchmark update skipped: {exc}')

    log_line(
        f'[wave2b] holdout calibrated macro_f1={float(calibrated_row["macro_f1"]):.4f} '
        f'top1={float(calibrated_row["top_1_accuracy"]):.4f} '
        f'top3={float(calibrated_row["top_3_accuracy"]):.4f} '
        f'ece={holdout_ece:.4f} '
        f'promotion_status={promotion_status}'
    )
    print(f'[write] {OUTPUTS_DIR / GLOBAL_MANIFEST_NAME}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
