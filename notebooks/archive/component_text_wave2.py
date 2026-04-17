import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import src.modeling.common.text_fusion as txt_fus
from src.config import settings
from src.config.paths import OUTPUTS_DIR, ensure_project_directories
from src.data.io_utils import load_frame, write_json
from src.features.component_text_sidecar import SIDECAR_STEM
from src.modeling.common.helpers import (
    FEATURE_WAVE1_SPLIT_MODE,
    MULTI_INPUT_STEM,
    SINGLE_INPUT_STEM,
    TARGET_COL,
    apply_multilabel_threshold,
    build_multiclass_calibration_df,
    build_multiclass_class_df,
    build_multiclass_confusion_df,
    feature_manifest,
    prep_multi_label_cases,
    prep_single_label_cases,
    split_multi_label_cases_by_mode,
    split_single_label_cases_by_mode,
)


def log_single_family(stage_name, family_name, row):
    txt_fus.log_line(
        f'[single] {stage_name} {family_name} '
        f'macro_f1={float(row["macro_f1"]):.4f} '
        f'top1={float(row["top_1_accuracy"]):.4f} '
        f'top3={float(row["top_3_accuracy"]):.4f}'
    )


def log_multi_family(stage_name, family_name, row):
    txt_fus.log_line(
        f'[multi] {stage_name} {family_name} '
        f'macro_f1={float(row["macro_f1"]):.4f} '
        f'micro_f1={float(row["micro_f1"]):.4f} '
        f'recall@3={float(row["recall_at_3"]):.4f}'
    )


# -----------------------------------------------------------------------------
# Single-label wave
# -----------------------------------------------------------------------------
def run_single_wave(args, structured_feature_info, locked_single_select_row, locked_single_manifest, locked_single_ece, locked_single_selection, checkpoint_fn=None):
    screen_rows = []
    select_rows = []
    holdout_rows = []
    class_df = pd.DataFrame()
    confusion_df = txt_fus.empty_single_confusion_df()
    calibration_df = txt_fus.empty_single_calibration_df()
    overlap_metrics = []
    completed_stages = []

    raw_df, input_path = load_frame(SINGLE_INPUT_STEM, input_path=args.single_input_path)
    sidecar_df, text_sidecar_path = load_frame(SIDECAR_STEM, input_path=args.text_sidecar_path)
    case_df = prep_single_label_cases(raw_df, structured_feature_info['feature_cols'])
    case_df = txt_fus.merge_text_sidecar(case_df, sidecar_df)
    split_parts = split_single_label_cases_by_mode(case_df, split_mode=FEATURE_WAVE1_SPLIT_MODE)

    train_core_df = split_parts['train_core']
    screen_df = split_parts['screen_2024']
    dev_screen_df = split_parts['dev_2020_2024']
    select_df = split_parts['select_2025']
    dev_select_df = split_parts['dev_2020_2025']
    holdout_df = split_parts['holdout_2026']
    locked_params = locked_single_selection['best_params']

    txt_fus.log_line(
        f'[single] Split rows | train_core={len(train_core_df):,} screen_2024={len(screen_df):,} '
        f'select_2025={len(select_df):,} holdout_2026={len(holdout_df):,} '
        f'final_linear_model={args.final_linear_model}'
    )

    def emit_checkpoint(checkpoint_stage, selected_family=None, select_metrics=None, select_gate_pass=None, promotion_status='running'):
        if checkpoint_fn is None:
            return
        checkpoint_fn(
            checkpoint_stage,
            {
                'input_path': str(input_path),
                'text_sidecar_path': str(text_sidecar_path),
                'split_df': split_parts['split_df'],
                'screen_df': pd.DataFrame(screen_rows),
                'select_df': pd.DataFrame(select_rows),
                'holdout_df': pd.DataFrame(holdout_rows) if holdout_rows else txt_fus.empty_single_holdout_df(),
                'class_df': class_df,
                'confusion_df': confusion_df,
                'calibration_df': calibration_df,
                'selected_family': selected_family,
                'final_linear_model': args.final_linear_model,
                'screen_fusion_weight': late_screen['selected_text_weight'] if 'late_screen' in locals() else None,
                'select_metrics': select_metrics,
                'select_gate_pass': select_gate_pass,
                'promotion_status': promotion_status,
                'overlap_metrics': overlap_metrics,
                'checkpoint_stage': checkpoint_stage,
                'completed_stages': list(completed_stages)
            }
        )

    structured_screen = txt_fus.fit_single_structured_family(
        train_core_df,
        screen_df,
        structured_feature_info,
        locked_params,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed
    )
    structured_screen_row = txt_fus.build_single_row(
        'single_label',
        txt_fus.STRUCTURED_FAMILY,
        input_path,
        text_sidecar_path,
        'screen_2024',
        'screen_2024',
        screen_df[TARGET_COL].astype(str),
        structured_screen['valid_proba'],
        structured_screen['classes'],
        fit_seconds=structured_screen['fit_seconds'],
        selected_iteration=structured_screen['selected_iteration']
    )
    screen_rows.append(structured_screen_row)
    log_single_family('screen_2024', txt_fus.STRUCTURED_FAMILY, structured_screen_row)

    text_only_screen = txt_fus.fit_single_text_family(
        train_core_df,
        screen_df,
        structured_feature_info,
        txt_fus.TEXT_ONLY_FAMILY,
        final_model=False
    )
    text_only_screen_row = txt_fus.build_single_row(
        'single_label',
        txt_fus.TEXT_ONLY_FAMILY,
        input_path,
        text_sidecar_path,
        'screen_2024',
        'screen_2024',
        screen_df[TARGET_COL].astype(str),
        text_only_screen['eval_proba'],
        text_only_screen['classes'],
        fit_seconds=text_only_screen['fit_seconds']
    )
    screen_rows.append(text_only_screen_row)
    log_single_family('screen_2024', txt_fus.TEXT_ONLY_FAMILY, text_only_screen_row)

    if args.skip_text_plus:
        txt_fus.log_line('[single] screen_2024 text_plus_structured_linear skipped by flag')
    else:
        text_plus_screen = txt_fus.fit_single_text_family(
            train_core_df,
            screen_df,
            structured_feature_info,
            txt_fus.TEXT_PLUS_STRUCTURED_FAMILY,
            final_model=False
        )
        text_plus_screen_row = txt_fus.build_single_row(
            'single_label',
            txt_fus.TEXT_PLUS_STRUCTURED_FAMILY,
            input_path,
            text_sidecar_path,
            'screen_2024',
            'screen_2024',
            screen_df[TARGET_COL].astype(str),
            text_plus_screen['eval_proba'],
            text_plus_screen['classes'],
            fit_seconds=text_plus_screen['fit_seconds']
        )
        screen_rows.append(text_plus_screen_row)
        log_single_family('screen_2024', txt_fus.TEXT_PLUS_STRUCTURED_FAMILY, text_plus_screen_row)

    late_screen = txt_fus.select_single_fusion_weight(
        screen_df[TARGET_COL].astype(str),
        text_only_screen['eval_proba'],
        text_only_screen['classes'],
        structured_screen['valid_proba'],
        structured_screen['classes']
    )
    late_screen_row = txt_fus.build_single_row(
        'single_label',
        txt_fus.LATE_FUSION_FAMILY,
        input_path,
        text_sidecar_path,
        'screen_2024',
        'screen_2024',
        screen_df[TARGET_COL].astype(str),
        late_screen['proba'],
        late_screen['classes'],
        fit_seconds=round(float(text_only_screen['fit_seconds']) + float(structured_screen['fit_seconds']), 2),
        selected_iteration=structured_screen['selected_iteration'],
        selected_text_weight=late_screen['selected_text_weight']
    )
    screen_rows.append(late_screen_row)
    log_single_family('screen_2024', txt_fus.LATE_FUSION_FAMILY, late_screen_row)
    completed_stages.append('screen_2024')
    emit_checkpoint('screen_2024_complete')

    structured_select = txt_fus.fit_single_structured_family(
        dev_screen_df,
        select_df,
        structured_feature_info,
        locked_params,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed
    )
    structured_select_row = txt_fus.build_single_row(
        'single_label',
        txt_fus.STRUCTURED_FAMILY,
        input_path,
        text_sidecar_path,
        'select_2025',
        'select_2025',
        select_df[TARGET_COL].astype(str),
        structured_select['valid_proba'],
        structured_select['classes'],
        fit_seconds=structured_select['fit_seconds'],
        selected_iteration=structured_select['selected_iteration']
    )
    select_rows.append(structured_select_row)
    log_single_family('select_2025', txt_fus.STRUCTURED_FAMILY, structured_select_row)

    text_only_select = txt_fus.fit_single_text_family(
        dev_screen_df,
        select_df,
        structured_feature_info,
        txt_fus.TEXT_ONLY_FAMILY,
        final_model=False
    )
    text_only_select_row = txt_fus.build_single_row(
        'single_label',
        txt_fus.TEXT_ONLY_FAMILY,
        input_path,
        text_sidecar_path,
        'select_2025',
        'select_2025',
        select_df[TARGET_COL].astype(str),
        text_only_select['eval_proba'],
        text_only_select['classes'],
        fit_seconds=text_only_select['fit_seconds']
    )
    select_rows.append(text_only_select_row)
    log_single_family('select_2025', txt_fus.TEXT_ONLY_FAMILY, text_only_select_row)

    if args.skip_text_plus:
        txt_fus.log_line('[single] select_2025 text_plus_structured_linear skipped by flag')
    else:
        text_plus_select = txt_fus.fit_single_text_family(
            dev_screen_df,
            select_df,
            structured_feature_info,
            txt_fus.TEXT_PLUS_STRUCTURED_FAMILY,
            final_model=False
        )
        text_plus_select_row = txt_fus.build_single_row(
            'single_label',
            txt_fus.TEXT_PLUS_STRUCTURED_FAMILY,
            input_path,
            text_sidecar_path,
            'select_2025',
            'select_2025',
            select_df[TARGET_COL].astype(str),
            text_plus_select['eval_proba'],
            text_plus_select['classes'],
            fit_seconds=text_plus_select['fit_seconds']
        )
        select_rows.append(text_plus_select_row)
        log_single_family('select_2025', txt_fus.TEXT_PLUS_STRUCTURED_FAMILY, text_plus_select_row)

    late_select = txt_fus.apply_single_fusion_weight(
        select_df[TARGET_COL].astype(str),
        text_only_select['eval_proba'],
        text_only_select['classes'],
        structured_select['valid_proba'],
        structured_select['classes'],
        text_weight=late_screen['selected_text_weight']
    )
    late_select_row = txt_fus.build_single_row(
        'single_label',
        txt_fus.LATE_FUSION_FAMILY,
        input_path,
        text_sidecar_path,
        'select_2025',
        'select_2025',
        select_df[TARGET_COL].astype(str),
        late_select['proba'],
        late_select['classes'],
        fit_seconds=round(float(text_only_select['fit_seconds']) + float(structured_select['fit_seconds']), 2),
        selected_iteration=structured_select['selected_iteration'],
        selected_text_weight=late_screen['selected_text_weight']
    )
    select_rows.append(late_select_row)
    log_single_family('select_2025', txt_fus.LATE_FUSION_FAMILY, late_select_row)

    select_df_all = pd.DataFrame(select_rows)
    current_best_row = txt_fus.select_best_row(
        select_df_all,
        ['macro_f1', 'top_1_accuracy', 'top_3_accuracy']
    )
    selected_family = current_best_row['family_name']
    select_improvement = float(current_best_row['macro_f1'] - locked_single_select_row['macro_f1'])
    select_gate_pass = select_improvement >= txt_fus.SINGLE_PROMOTE_SELECT_DELTA
    promotion_status = 'rejected_select'
    txt_fus.log_line(
        f'[single] select_2025 best={selected_family} '
        f'macro_f1={float(current_best_row["macro_f1"]):.4f} '
        f'delta_vs_locked={select_improvement:+.4f} '
        f'gate_pass={str(select_gate_pass).lower()}'
    )
    completed_stages.append('select_2025')
    emit_checkpoint(
        'select_2025_complete',
        selected_family=selected_family,
        select_metrics=current_best_row,
        select_gate_pass=select_gate_pass,
        promotion_status='running'
    )

    if select_gate_pass:
        txt_fus.log_line(f'[single] holdout_2026 start family={selected_family}')
        emit_checkpoint(
            'holdout_2026_started',
            selected_family=selected_family,
            select_metrics=current_best_row,
            select_gate_pass=select_gate_pass,
            promotion_status='running'
        )
        if selected_family == txt_fus.STRUCTURED_FAMILY:
            txt_fus.log_line(
                f'[single] holdout_2026 fitting structured carry-forward '
                f'iteration={int(structured_select["selected_iteration"])}'
            )
            holdout_result = txt_fus.fit_single_structured_holdout(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                locked_params,
                structured_select['selected_iteration'],
                task_type=str(args.task_type).upper(),
                devices=args.devices,
                random_seed=args.random_seed
            )
            txt_fus.log_line(
                f'[single] holdout_2026 structured fit complete '
                f'fit_seconds={float(holdout_result["fit_seconds"]):.2f}'
            )
            holdout_row = txt_fus.build_single_row(
                'single_label',
                txt_fus.STRUCTURED_FAMILY,
                input_path,
                text_sidecar_path,
                'final_holdout',
                'holdout_2026',
                holdout_df[TARGET_COL].astype(str),
                holdout_result['proba'],
                holdout_result['classes'],
                fit_seconds=holdout_result['fit_seconds'],
                selected_iteration=structured_select['selected_iteration']
            )
        elif selected_family == txt_fus.TEXT_ONLY_FAMILY:
            txt_fus.log_line(
                f'[single] holdout_2026 fitting final text-only linear model '
                f'dev_rows={len(dev_select_df):,} '
                f'final_model={args.final_linear_model}'
            )
            holdout_result = txt_fus.fit_single_text_family(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                txt_fus.TEXT_ONLY_FAMILY,
                final_model=True,
                final_model_kind=args.final_linear_model
            )
            txt_fus.log_line(
                f'[single] holdout_2026 text-only fit complete '
                f'fit_seconds={float(holdout_result["fit_seconds"]):.2f}'
            )
            holdout_result = {
                'fit_seconds': holdout_result['fit_seconds'],
                'proba': holdout_result['eval_proba'],
                'pred': holdout_result['pred'],
                'classes': holdout_result['classes']
            }
            holdout_row = txt_fus.build_single_row(
                'single_label',
                txt_fus.TEXT_ONLY_FAMILY,
                input_path,
                text_sidecar_path,
                'final_holdout',
                'holdout_2026',
                holdout_df[TARGET_COL].astype(str),
                holdout_result['proba'],
                holdout_result['classes'],
                fit_seconds=holdout_result['fit_seconds']
            )
        elif selected_family == txt_fus.TEXT_PLUS_STRUCTURED_FAMILY:
            txt_fus.log_line(
                f'[single] holdout_2026 fitting final text+structured linear model '
                f'dev_rows={len(dev_select_df):,} '
                f'final_model={args.final_linear_model}'
            )
            holdout_result = txt_fus.fit_single_text_family(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                txt_fus.TEXT_PLUS_STRUCTURED_FAMILY,
                final_model=True,
                final_model_kind=args.final_linear_model
            )
            txt_fus.log_line(
                f'[single] holdout_2026 text+structured fit complete '
                f'fit_seconds={float(holdout_result["fit_seconds"]):.2f}'
            )
            holdout_result = {
                'fit_seconds': holdout_result['fit_seconds'],
                'proba': holdout_result['eval_proba'],
                'pred': holdout_result['pred'],
                'classes': holdout_result['classes']
            }
            holdout_row = txt_fus.build_single_row(
                'single_label',
                txt_fus.TEXT_PLUS_STRUCTURED_FAMILY,
                input_path,
                text_sidecar_path,
                'final_holdout',
                'holdout_2026',
                holdout_df[TARGET_COL].astype(str),
                holdout_result['proba'],
                holdout_result['classes'],
                fit_seconds=holdout_result['fit_seconds']
            )
        else:
            txt_fus.log_line(
                f'[single] holdout_2026 fitting late-fusion structured branch '
                f'iteration={int(structured_select["selected_iteration"])}'
            )
            structured_holdout = txt_fus.fit_single_structured_holdout(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                locked_params,
                structured_select['selected_iteration'],
                task_type=str(args.task_type).upper(),
                devices=args.devices,
                random_seed=args.random_seed
            )
            txt_fus.log_line(
                f'[single] holdout_2026 structured branch complete '
                f'fit_seconds={float(structured_holdout["fit_seconds"]):.2f}'
            )
            txt_fus.log_line(
                f'[single] holdout_2026 fitting late-fusion text branch '
                f'text_weight={float(late_screen["selected_text_weight"]):.2f} '
                f'dev_rows={len(dev_select_df):,} '
                f'final_model={args.final_linear_model}'
            )
            text_holdout = txt_fus.fit_single_text_family(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                txt_fus.TEXT_ONLY_FAMILY,
                final_model=True,
                final_model_kind=args.final_linear_model
            )
            txt_fus.log_line(
                f'[single] holdout_2026 text branch complete '
                f'fit_seconds={float(text_holdout["fit_seconds"]):.2f}'
            )
            late_holdout = txt_fus.apply_single_fusion_weight(
                holdout_df[TARGET_COL].astype(str),
                text_holdout['eval_proba'],
                text_holdout['classes'],
                structured_holdout['proba'],
                structured_holdout['classes'],
                text_weight=late_screen['selected_text_weight']
            )
            holdout_result = {
                'fit_seconds': round(float(text_holdout['fit_seconds']) + float(structured_holdout['fit_seconds']), 2),
                'proba': late_holdout['proba'],
                'pred': late_holdout['pred'],
                'classes': late_holdout['classes']
            }
            holdout_row = txt_fus.build_single_row(
                'single_label',
                txt_fus.LATE_FUSION_FAMILY,
                input_path,
                text_sidecar_path,
                'final_holdout',
                'holdout_2026',
                holdout_df[TARGET_COL].astype(str),
                holdout_result['proba'],
                holdout_result['classes'],
                fit_seconds=holdout_result['fit_seconds'],
                selected_iteration=structured_select['selected_iteration'],
                selected_text_weight=late_screen['selected_text_weight']
            )

        holdout_rows.append(holdout_row)
        log_single_family('holdout_2026', selected_family, holdout_row)
        class_df = build_multiclass_class_df(
            holdout_df[TARGET_COL].astype(str),
            holdout_result['pred'],
            holdout_result['classes']
        )
        calibration_df = build_multiclass_calibration_df(
            holdout_df[TARGET_COL].astype(str),
            holdout_result['proba'],
            holdout_result['classes']
        )
        focus_groups = (
            dev_select_df[TARGET_COL]
            .value_counts()
            .head(12)
            .index
            .tolist()
        )
        confusion_df = build_multiclass_confusion_df(
            holdout_df[TARGET_COL].astype(str),
            holdout_result['pred'],
            focus_groups
        )

        if selected_family in txt_fus.TEXT_FAMILIES:
            overlap_mask = txt_fus.build_overlap_mask(
                dev_select_df['cdescr_model_text'],
                holdout_df['cdescr_model_text']
            )
            slice_rows = txt_fus.build_single_overlap_rows(
                holdout_row,
                holdout_df[TARGET_COL].astype(str),
                holdout_result['proba'],
                holdout_result['classes'],
                overlap_mask
            )
            holdout_rows.extend(slice_rows)
            overlap_metrics = slice_rows

        holdout_macro_gain = float(holdout_row['macro_f1'] - locked_single_manifest['official_holdout_metrics']['macro_f1'])
        holdout_top3_ok = holdout_row['top_3_accuracy'] >= (
            locked_single_manifest['official_holdout_metrics']['top_3_accuracy'] - txt_fus.SINGLE_TOP3_DROP_LIMIT
        )
        holdout_ece = float(calibration_df.loc[calibration_df['section'].eq('overall'), 'ece'].iloc[0])
        holdout_ece_ok = (holdout_ece - locked_single_ece) <= txt_fus.SINGLE_ECE_WORSE_LIMIT
        promotion_status = 'promoted' if (
            holdout_macro_gain >= txt_fus.SINGLE_PROMOTE_HOLDOUT_DELTA
            and holdout_top3_ok
            and holdout_ece_ok
        ) else 'rejected_holdout'
        txt_fus.log_line(
            f'[single] holdout_2026 promotion_status={promotion_status} '
            f'macro_gain={holdout_macro_gain:+.4f} '
            f'top3_ok={str(bool(holdout_top3_ok)).lower()} '
            f'ece_delta={(holdout_ece - locked_single_ece):+.4f}'
        )
        completed_stages.append('holdout_2026')
        emit_checkpoint(
            'holdout_2026_complete',
            selected_family=selected_family,
            select_metrics=current_best_row,
            select_gate_pass=select_gate_pass,
            promotion_status=promotion_status
        )
    else:
        txt_fus.log_line(
            f'[single] select_2025 gate rejected; skipping holdout '
            f'family={selected_family}'
        )
        emit_checkpoint(
            'select_gate_rejected',
            selected_family=selected_family,
            select_metrics=current_best_row,
            select_gate_pass=select_gate_pass,
            promotion_status=promotion_status
        )

    return {
        'input_path': str(input_path),
        'text_sidecar_path': str(text_sidecar_path),
        'split_df': split_parts['split_df'],
        'screen_df': pd.DataFrame(screen_rows),
        'select_df': select_df_all,
        'holdout_df': pd.DataFrame(holdout_rows) if holdout_rows else txt_fus.empty_single_holdout_df(),
        'class_df': class_df,
        'confusion_df': confusion_df,
        'calibration_df': calibration_df,
        'selected_family': selected_family,
        'final_linear_model': args.final_linear_model,
        'screen_fusion_weight': late_screen['selected_text_weight'],
        'select_metrics': current_best_row,
        'select_gate_pass': select_gate_pass,
        'promotion_status': promotion_status,
        'overlap_metrics': overlap_metrics,
        'checkpoint_stage': 'completed',
        'completed_stages': list(completed_stages)
    }


# -----------------------------------------------------------------------------
# Multi-label wave
# -----------------------------------------------------------------------------
def run_multi_wave(args, structured_feature_info, locked_multi_select_row, locked_multi_manifest, checkpoint_fn=None):
    screen_rows = []
    select_rows = []
    holdout_rows = []
    label_df = txt_fus.empty_multi_label_df()
    overlap_metrics = []
    completed_stages = []

    raw_df, input_path = load_frame(MULTI_INPUT_STEM, input_path=args.multi_input_path)
    sidecar_df, text_sidecar_path = load_frame(SIDECAR_STEM, input_path=args.text_sidecar_path)
    case_df = prep_multi_label_cases(raw_df, structured_feature_info['feature_cols'])
    case_df = txt_fus.merge_text_sidecar(case_df, sidecar_df)
    split_parts = split_multi_label_cases_by_mode(case_df, split_mode=FEATURE_WAVE1_SPLIT_MODE)

    train_core_df = split_parts['train_core']
    screen_df = split_parts['screen_2024']
    dev_screen_df = split_parts['dev_2020_2024']
    select_df = split_parts['select_2025']
    dev_select_df = split_parts['dev_2020_2025']
    holdout_df = split_parts['holdout_2026']

    txt_fus.log_line(
        f'[multi] Split rows | train_core={len(train_core_df):,} screen_2024={len(screen_df):,} '
        f'select_2025={len(select_df):,} holdout_2026={len(holdout_df):,} '
        f'final_linear_model={args.final_linear_model}'
    )

    def emit_checkpoint(checkpoint_stage, selected_family=None, select_metrics=None, select_gate_pass=None, promotion_status='running'):
        if checkpoint_fn is None:
            return
        checkpoint_fn(
            checkpoint_stage,
            {
                'input_path': str(input_path),
                'text_sidecar_path': str(text_sidecar_path),
                'split_df': split_parts['split_df'],
                'screen_df': pd.DataFrame(screen_rows),
                'select_df': pd.DataFrame(select_rows),
                'holdout_df': pd.DataFrame(holdout_rows) if holdout_rows else txt_fus.empty_multi_holdout_df(),
                'label_df': label_df,
                'selected_family': selected_family,
                'final_linear_model': args.final_linear_model,
                'screen_fusion_weight': late_screen['selected_text_weight'] if 'late_screen' in locals() else None,
                'select_metrics': select_metrics,
                'select_gate_pass': select_gate_pass,
                'promotion_status': promotion_status,
                'overlap_metrics': overlap_metrics,
                'checkpoint_stage': checkpoint_stage,
                'completed_stages': list(completed_stages)
            }
        )

    structured_screen = txt_fus.fit_multi_structured_family(
        train_core_df,
        screen_df,
        structured_feature_info,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed
    )
    structured_screen_row = txt_fus.build_multi_row(
        'multi_label',
        txt_fus.STRUCTURED_FAMILY,
        input_path,
        text_sidecar_path,
        'screen_2024',
        'screen_2024',
        structured_screen['y_eval'],
        structured_screen['valid_pred'],
        structured_screen['valid_proba'],
        threshold=structured_screen['selected_threshold'],
        fit_seconds=structured_screen['fit_seconds'],
        selected_iteration=structured_screen['selected_iteration']
    )
    structured_screen_row['actual_task_type'] = structured_screen['actual_task_type']
    screen_rows.append(structured_screen_row)
    log_multi_family('screen_2024', txt_fus.STRUCTURED_FAMILY, structured_screen_row)

    text_only_screen = txt_fus.fit_multi_text_family(
        train_core_df,
        screen_df,
        structured_feature_info,
        txt_fus.TEXT_ONLY_FAMILY,
        final_model=False
    )
    text_only_screen_row = txt_fus.build_multi_row(
        'multi_label',
        txt_fus.TEXT_ONLY_FAMILY,
        input_path,
        text_sidecar_path,
        'screen_2024',
        'screen_2024',
        text_only_screen['y_eval'],
        text_only_screen['pred'],
        text_only_screen['eval_proba'],
        threshold=text_only_screen['threshold_choice']['threshold'],
        fit_seconds=text_only_screen['fit_seconds']
    )
    screen_rows.append(text_only_screen_row)
    log_multi_family('screen_2024', txt_fus.TEXT_ONLY_FAMILY, text_only_screen_row)

    if args.skip_text_plus:
        txt_fus.log_line('[multi] screen_2024 text_plus_structured_linear skipped by flag')
    else:
        text_plus_screen = txt_fus.fit_multi_text_family(
            train_core_df,
            screen_df,
            structured_feature_info,
            txt_fus.TEXT_PLUS_STRUCTURED_FAMILY,
            final_model=False
        )
        text_plus_screen_row = txt_fus.build_multi_row(
            'multi_label',
            txt_fus.TEXT_PLUS_STRUCTURED_FAMILY,
            input_path,
            text_sidecar_path,
            'screen_2024',
            'screen_2024',
            text_plus_screen['y_eval'],
            text_plus_screen['pred'],
            text_plus_screen['eval_proba'],
            threshold=text_plus_screen['threshold_choice']['threshold'],
            fit_seconds=text_plus_screen['fit_seconds']
        )
        screen_rows.append(text_plus_screen_row)
        log_multi_family('screen_2024', txt_fus.TEXT_PLUS_STRUCTURED_FAMILY, text_plus_screen_row)

    late_screen = txt_fus.select_multi_fusion_weight(
        structured_screen['y_eval'],
        text_only_screen['eval_proba'],
        structured_screen['valid_proba']
    )
    late_screen_row = txt_fus.build_multi_row(
        'multi_label',
        txt_fus.LATE_FUSION_FAMILY,
        input_path,
        text_sidecar_path,
        'screen_2024',
        'screen_2024',
        structured_screen['y_eval'],
        late_screen['pred'],
        late_screen['proba'],
        threshold=late_screen['selected_threshold'],
        fit_seconds=round(float(text_only_screen['fit_seconds']) + float(structured_screen['fit_seconds']), 2),
        selected_iteration=structured_screen['selected_iteration'],
        selected_text_weight=late_screen['selected_text_weight']
    )
    screen_rows.append(late_screen_row)
    log_multi_family('screen_2024', txt_fus.LATE_FUSION_FAMILY, late_screen_row)
    completed_stages.append('screen_2024')
    emit_checkpoint('screen_2024_complete')

    structured_select = txt_fus.fit_multi_structured_family(
        dev_screen_df,
        select_df,
        structured_feature_info,
        task_type=str(args.task_type).upper(),
        devices=args.devices,
        random_seed=args.random_seed
    )
    structured_select_row = txt_fus.build_multi_row(
        'multi_label',
        txt_fus.STRUCTURED_FAMILY,
        input_path,
        text_sidecar_path,
        'select_2025',
        'select_2025',
        structured_select['y_eval'],
        structured_select['valid_pred'],
        structured_select['valid_proba'],
        threshold=structured_select['selected_threshold'],
        fit_seconds=structured_select['fit_seconds'],
        selected_iteration=structured_select['selected_iteration']
    )
    structured_select_row['actual_task_type'] = structured_select['actual_task_type']
    select_rows.append(structured_select_row)
    log_multi_family('select_2025', txt_fus.STRUCTURED_FAMILY, structured_select_row)

    text_only_select = txt_fus.fit_multi_text_family(
        dev_screen_df,
        select_df,
        structured_feature_info,
        txt_fus.TEXT_ONLY_FAMILY,
        final_model=False
    )
    text_only_select_row = txt_fus.build_multi_row(
        'multi_label',
        txt_fus.TEXT_ONLY_FAMILY,
        input_path,
        text_sidecar_path,
        'select_2025',
        'select_2025',
        text_only_select['y_eval'],
        text_only_select['pred'],
        text_only_select['eval_proba'],
        threshold=text_only_select['threshold_choice']['threshold'],
        fit_seconds=text_only_select['fit_seconds']
    )
    select_rows.append(text_only_select_row)
    log_multi_family('select_2025', txt_fus.TEXT_ONLY_FAMILY, text_only_select_row)

    if args.skip_text_plus:
        txt_fus.log_line('[multi] select_2025 text_plus_structured_linear skipped by flag')
    else:
        text_plus_select = txt_fus.fit_multi_text_family(
            dev_screen_df,
            select_df,
            structured_feature_info,
            txt_fus.TEXT_PLUS_STRUCTURED_FAMILY,
            final_model=False
        )
        text_plus_select_row = txt_fus.build_multi_row(
            'multi_label',
            txt_fus.TEXT_PLUS_STRUCTURED_FAMILY,
            input_path,
            text_sidecar_path,
            'select_2025',
            'select_2025',
            text_plus_select['y_eval'],
            text_plus_select['pred'],
            text_plus_select['eval_proba'],
            threshold=text_plus_select['threshold_choice']['threshold'],
            fit_seconds=text_plus_select['fit_seconds']
        )
        select_rows.append(text_plus_select_row)
        log_multi_family('select_2025', txt_fus.TEXT_PLUS_STRUCTURED_FAMILY, text_plus_select_row)

    late_select = txt_fus.apply_multi_fusion_weight(
        structured_select['y_eval'],
        text_only_select['eval_proba'],
        structured_select['valid_proba'],
        text_weight=late_screen['selected_text_weight']
    )
    late_select_row = txt_fus.build_multi_row(
        'multi_label',
        txt_fus.LATE_FUSION_FAMILY,
        input_path,
        text_sidecar_path,
        'select_2025',
        'select_2025',
        structured_select['y_eval'],
        late_select['pred'],
        late_select['proba'],
        threshold=late_select['selected_threshold'],
        fit_seconds=round(float(text_only_select['fit_seconds']) + float(structured_select['fit_seconds']), 2),
        selected_iteration=structured_select['selected_iteration'],
        selected_text_weight=late_screen['selected_text_weight']
    )
    select_rows.append(late_select_row)
    log_multi_family('select_2025', txt_fus.LATE_FUSION_FAMILY, late_select_row)

    select_df_all = pd.DataFrame(select_rows)
    current_best_row = txt_fus.select_best_row(
        select_df_all,
        ['macro_f1', 'micro_f1', 'recall_at_3', 'precision_at_3']
    )
    selected_family = current_best_row['family_name']
    select_improvement = float(current_best_row['macro_f1'] - locked_multi_select_row['macro_f1'])
    select_gate_pass = select_improvement >= txt_fus.MULTI_PROMOTE_SELECT_DELTA
    promotion_status = 'rejected_select'
    txt_fus.log_line(
        f'[multi] select_2025 best={selected_family} '
        f'macro_f1={float(current_best_row["macro_f1"]):.4f} '
        f'delta_vs_locked={select_improvement:+.4f} '
        f'gate_pass={str(select_gate_pass).lower()}'
    )
    completed_stages.append('select_2025')
    emit_checkpoint(
        'select_2025_complete',
        selected_family=selected_family,
        select_metrics=current_best_row,
        select_gate_pass=select_gate_pass,
        promotion_status='running'
    )

    if select_gate_pass:
        txt_fus.log_line(f'[multi] holdout_2026 start family={selected_family}')
        emit_checkpoint(
            'holdout_2026_started',
            selected_family=selected_family,
            select_metrics=current_best_row,
            select_gate_pass=select_gate_pass,
            promotion_status='running'
        )
        if selected_family == txt_fus.STRUCTURED_FAMILY:
            txt_fus.log_line(
                f'[multi] holdout_2026 fitting structured carry-forward '
                f'iteration={int(structured_select["selected_iteration"])} '
                f'threshold={float(structured_select["selected_threshold"]):.3f}'
            )
            holdout_result = txt_fus.fit_multi_structured_holdout(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                structured_select,
                devices=args.devices,
                random_seed=args.random_seed
            )
            txt_fus.log_line(
                f'[multi] holdout_2026 structured fit complete '
                f'fit_seconds={float(holdout_result["fit_seconds"]):.2f}'
            )
            y_holdout = holdout_result['y_holdout']
            holdout_pred = holdout_result['holdout_pred']
            holdout_proba = holdout_result['holdout_proba']
            holdout_row = txt_fus.build_multi_row(
                'multi_label',
                txt_fus.STRUCTURED_FAMILY,
                input_path,
                text_sidecar_path,
                'final_holdout',
                'holdout_2026',
                y_holdout,
                holdout_pred,
                holdout_proba,
                threshold=holdout_result['selected_threshold'],
                fit_seconds=holdout_result['fit_seconds'],
                selected_iteration=holdout_result['selected_iteration']
            )
            mlb = holdout_result['mlb']
        elif selected_family == txt_fus.TEXT_ONLY_FAMILY:
            txt_fus.log_line(
                f'[multi] holdout_2026 fitting final text-only linear model '
                f'dev_rows={len(dev_select_df):,} '
                f'final_model={args.final_linear_model}'
            )
            text_holdout = txt_fus.fit_multi_text_family(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                txt_fus.TEXT_ONLY_FAMILY,
                final_model=True,
                final_model_kind=args.final_linear_model
            )
            txt_fus.log_line(
                f'[multi] holdout_2026 text-only fit complete '
                f'fit_seconds={float(text_holdout["fit_seconds"]):.2f}'
            )
            y_holdout = text_holdout['y_eval']
            holdout_proba = text_holdout['eval_proba']
            holdout_pred = apply_multilabel_threshold(
                holdout_proba,
                text_only_select['threshold_choice']['threshold'],
                min_positive_labels=txt_fus.MIN_POSITIVE_LABELS
            )
            holdout_row = txt_fus.build_multi_row(
                'multi_label',
                txt_fus.TEXT_ONLY_FAMILY,
                input_path,
                text_sidecar_path,
                'final_holdout',
                'holdout_2026',
                y_holdout,
                holdout_pred,
                holdout_proba,
                threshold=text_only_select['threshold_choice']['threshold'],
                fit_seconds=text_holdout['fit_seconds']
            )
            mlb = text_holdout['mlb']
        elif selected_family == txt_fus.TEXT_PLUS_STRUCTURED_FAMILY:
            txt_fus.log_line(
                f'[multi] holdout_2026 fitting final text+structured linear model '
                f'dev_rows={len(dev_select_df):,} '
                f'final_model={args.final_linear_model}'
            )
            text_holdout = txt_fus.fit_multi_text_family(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                txt_fus.TEXT_PLUS_STRUCTURED_FAMILY,
                final_model=True,
                final_model_kind=args.final_linear_model
            )
            txt_fus.log_line(
                f'[multi] holdout_2026 text+structured fit complete '
                f'fit_seconds={float(text_holdout["fit_seconds"]):.2f}'
            )
            y_holdout = text_holdout['y_eval']
            holdout_proba = text_holdout['eval_proba']
            holdout_pred = apply_multilabel_threshold(
                holdout_proba,
                text_plus_select['threshold_choice']['threshold'],
                min_positive_labels=txt_fus.MIN_POSITIVE_LABELS
            )
            holdout_row = txt_fus.build_multi_row(
                'multi_label',
                txt_fus.TEXT_PLUS_STRUCTURED_FAMILY,
                input_path,
                text_sidecar_path,
                'final_holdout',
                'holdout_2026',
                y_holdout,
                holdout_pred,
                holdout_proba,
                threshold=text_plus_select['threshold_choice']['threshold'],
                fit_seconds=text_holdout['fit_seconds']
            )
            mlb = text_holdout['mlb']
        else:
            txt_fus.log_line(
                f'[multi] holdout_2026 fitting late-fusion structured branch '
                f'iteration={int(structured_select["selected_iteration"])} '
                f'text_weight={float(late_screen["selected_text_weight"]):.2f}'
            )
            structured_holdout = txt_fus.fit_multi_structured_holdout(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                structured_select,
                devices=args.devices,
                random_seed=args.random_seed
            )
            txt_fus.log_line(
                f'[multi] holdout_2026 structured branch complete '
                f'fit_seconds={float(structured_holdout["fit_seconds"]):.2f}'
            )
            txt_fus.log_line(
                f'[multi] holdout_2026 fitting late-fusion text branch '
                f'dev_rows={len(dev_select_df):,} '
                f'final_model={args.final_linear_model}'
            )
            text_holdout = txt_fus.fit_multi_text_family(
                dev_select_df,
                holdout_df,
                structured_feature_info,
                txt_fus.TEXT_ONLY_FAMILY,
                final_model=True,
                final_model_kind=args.final_linear_model
            )
            txt_fus.log_line(
                f'[multi] holdout_2026 text branch complete '
                f'fit_seconds={float(text_holdout["fit_seconds"]):.2f}'
            )
            y_holdout = text_holdout['y_eval']
            holdout_proba = (
                float(late_screen['selected_text_weight']) * text_holdout['eval_proba']
                + (1.0 - float(late_screen['selected_text_weight'])) * structured_holdout['holdout_proba']
            )
            holdout_pred = apply_multilabel_threshold(
                holdout_proba,
                late_select['selected_threshold'],
                min_positive_labels=txt_fus.MIN_POSITIVE_LABELS
            )
            holdout_row = txt_fus.build_multi_row(
                'multi_label',
                txt_fus.LATE_FUSION_FAMILY,
                input_path,
                text_sidecar_path,
                'final_holdout',
                'holdout_2026',
                y_holdout,
                holdout_pred,
                holdout_proba,
                threshold=late_select['selected_threshold'],
                fit_seconds=round(float(text_holdout['fit_seconds']) + float(structured_holdout['fit_seconds']), 2),
                selected_iteration=structured_holdout['selected_iteration'],
                selected_text_weight=late_screen['selected_text_weight']
            )
            mlb = text_holdout['mlb']

        holdout_rows.append(holdout_row)
        log_multi_family('holdout_2026', selected_family, holdout_row)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_holdout,
            holdout_pred,
            average=None,
            zero_division=0
        )
        label_df = pd.DataFrame(
            {
                'component_group': mlb.classes_,
                'support': support,
                'precision': np.round(precision, 4),
                'recall': np.round(recall, 4),
                'f1': np.round(f1, 4)
            }
        ).sort_values(['support', 'f1'], ascending=[False, False]).reset_index(drop=True)

        if selected_family in txt_fus.TEXT_FAMILIES:
            overlap_mask = txt_fus.build_overlap_mask(
                dev_select_df['cdescr_model_text'],
                holdout_df['cdescr_model_text']
            )
            slice_rows = txt_fus.build_multi_overlap_rows(
                holdout_row,
                y_holdout,
                holdout_pred,
                holdout_proba,
                overlap_mask
            )
            holdout_rows.extend(slice_rows)
            overlap_metrics = slice_rows

        locked_holdout = locked_multi_manifest['official_holdout_metrics']
        macro_gain = float(holdout_row['macro_f1'] - locked_holdout['macro_f1'])
        micro_gain = float(holdout_row['micro_f1'] - locked_holdout['micro_f1'])
        promotion_status = 'promoted' if (
            (macro_gain >= txt_fus.MULTI_PROMOTE_HOLDOUT_MACRO_DELTA or micro_gain >= txt_fus.MULTI_PROMOTE_HOLDOUT_MICRO_DELTA)
            and holdout_row['recall_at_3'] >= locked_holdout['recall_at_3']
            and holdout_row['label_coverage'] >= txt_fus.MULTI_LABEL_COVERAGE_FLOOR
        ) else 'rejected_holdout'
        txt_fus.log_line(
            f'[multi] holdout_2026 promotion_status={promotion_status} '
            f'macro_gain={macro_gain:+.4f} '
            f'micro_gain={micro_gain:+.4f} '
            f'recall3_ok={str(holdout_row["recall_at_3"] >= locked_holdout["recall_at_3"]).lower()} '
            f'label_coverage={float(holdout_row["label_coverage"]):.4f}'
        )
        completed_stages.append('holdout_2026')
        emit_checkpoint(
            'holdout_2026_complete',
            selected_family=selected_family,
            select_metrics=current_best_row,
            select_gate_pass=select_gate_pass,
            promotion_status=promotion_status
        )
    else:
        txt_fus.log_line(
            f'[multi] select_2025 gate rejected; skipping holdout '
            f'family={selected_family}'
        )
        emit_checkpoint(
            'select_gate_rejected',
            selected_family=selected_family,
            select_metrics=current_best_row,
            select_gate_pass=select_gate_pass,
            promotion_status=promotion_status
        )

    return {
        'input_path': str(input_path),
        'text_sidecar_path': str(text_sidecar_path),
        'split_df': split_parts['split_df'],
        'screen_df': pd.DataFrame(screen_rows),
        'select_df': select_df_all,
        'holdout_df': pd.DataFrame(holdout_rows) if holdout_rows else txt_fus.empty_multi_holdout_df(),
        'label_df': label_df,
        'selected_family': selected_family,
        'final_linear_model': args.final_linear_model,
        'screen_fusion_weight': late_screen['selected_text_weight'],
        'select_metrics': current_best_row,
        'select_gate_pass': select_gate_pass,
        'promotion_status': promotion_status,
        'overlap_metrics': overlap_metrics,
        'checkpoint_stage': 'completed',
        'completed_stages': list(completed_stages)
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Run component text wave 2 with complaint narratives plus the carried-forward structured family'
    )
    parser.add_argument(
        '--task-type',
        choices=['CPU', 'GPU', 'cpu', 'gpu'],
        default='CPU',
        help='CatBoost processing target for the structured carry-forward family'
    )
    parser.add_argument(
        '--devices',
        default='0',
        help='GPU device string for CatBoost when task_type is GPU'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=settings.RANDOM_SEED
    )
    parser.add_argument('--single-input-path', default=None)
    parser.add_argument('--multi-input-path', default=None)
    parser.add_argument('--text-sidecar-path', default=None)
    parser.add_argument('--skip-single', action='store_true')
    parser.add_argument('--skip-multi', action='store_true')
    parser.add_argument(
        '--skip-text-plus',
        action='store_true',
        help='Skip the early-fusion text_plus_structured_linear family'
    )
    parser.add_argument(
        '--final-linear-model',
        choices=txt_fus.FINAL_LINEAR_MODEL_CHOICES,
        default=txt_fus.FINAL_LINEAR_MODEL_DEFAULT,
        help='Final refit estimator for promoted text families'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    if args.skip_single and args.skip_multi:
        raise ValueError('Nothing to do: both single and multi tasks were skipped')

    structured_feature_info = feature_manifest(txt_fus.STRUCTURED_FEATURE_SET)
    locked_single_select_row = txt_fus.read_locked_single_select_baseline()
    locked_multi_select_row = txt_fus.read_locked_multi_select_baseline()
    locked_single_manifest = txt_fus.load_json(txt_fus.LOCKED_SINGLE_MANIFEST)
    locked_multi_manifest = txt_fus.load_json(txt_fus.LOCKED_MULTI_MANIFEST)
    locked_single_selection = txt_fus.load_json(txt_fus.LOCKED_SINGLE_SELECTION)
    locked_single_ece = txt_fus.read_locked_single_ece()

    manifest = {
        'artifact_role': txt_fus.FEATUREWAVE_TASK,
        'feature_wave': 2,
        'split_mode': FEATURE_WAVE1_SPLIT_MODE,
        'public_benchmark_locked': True,
        'run_status': 'running',
        'structured_companion_feature_set': txt_fus.STRUCTURED_FEATURE_SET,
        'final_linear_model': args.final_linear_model,
        'text_config': txt_fus.TEXT_CONFIG,
        'last_checkpoint': None,
        'tasks': {}
    }
    write_json(manifest, OUTPUTS_DIR / txt_fus.GLOBAL_MANIFEST_NAME)

    def checkpoint_single(stage_name, result):
        # CSV wave 2 artifacts deleted per bloat reduction
        # txt_fus.write_single_outputs(result)
        manifest['tasks']['single_label'] = txt_fus.build_single_manifest_entry(
            result,
            locked_single_select_row,
            locked_single_manifest['official_holdout_metrics'],
            locked_single_ece
        )
        manifest['last_checkpoint'] = {
            'task': 'single_label',
            'stage': stage_name
        }
        write_json(manifest, OUTPUTS_DIR / txt_fus.GLOBAL_MANIFEST_NAME)

    def checkpoint_multi(stage_name, result):
        # CSV wave 2 artifacts deleted per bloat reduction
        # txt_fus.write_multi_outputs(result)
        manifest['tasks']['multi_label'] = txt_fus.build_multi_manifest_entry(
            result,
            locked_multi_select_row,
            locked_multi_manifest['official_holdout_metrics']
        )
        manifest['last_checkpoint'] = {
            'task': 'multi_label',
            'stage': stage_name
        }
        write_json(manifest, OUTPUTS_DIR / txt_fus.GLOBAL_MANIFEST_NAME)

    try:
        if not args.skip_single:
            txt_fus.log_line('[run] Single-label text wave 2')
            single_result = run_single_wave(
                args,
                structured_feature_info,
                locked_single_select_row,
                locked_single_manifest,
                locked_single_ece,
                locked_single_selection,
                checkpoint_fn=checkpoint_single
            )
            # CSV wave 2 artifacts deleted per bloat reduction
            # txt_fus.write_single_outputs(single_result)
            manifest['tasks']['single_label'] = txt_fus.build_single_manifest_entry(
                single_result,
                locked_single_select_row,
                locked_single_manifest['official_holdout_metrics'],
                locked_single_ece
            )

        if not args.skip_multi:
            txt_fus.log_line('[run] Multi-label text wave 2')
            multi_result = run_multi_wave(
                args,
                structured_feature_info,
                locked_multi_select_row,
                locked_multi_manifest,
                checkpoint_fn=checkpoint_multi
            )
            # CSV wave 2 artifacts deleted per bloat reduction
            # txt_fus.write_multi_outputs(multi_result)
            manifest['tasks']['multi_label'] = txt_fus.build_multi_manifest_entry(
                multi_result,
                locked_multi_select_row,
                locked_multi_manifest['official_holdout_metrics']
            )
    except Exception as exc:
        manifest['run_status'] = 'failed'
        manifest['error'] = str(exc)
        write_json(manifest, OUTPUTS_DIR / txt_fus.GLOBAL_MANIFEST_NAME)
        raise

    manifest['run_status'] = 'completed'
    write_json(manifest, OUTPUTS_DIR / txt_fus.GLOBAL_MANIFEST_NAME)
    print(f'[write] {OUTPUTS_DIR / txt_fus.GLOBAL_MANIFEST_NAME}')
    print('[done] Component text wave 2 finished')
    return 0


if __name__ == '__main__':
    sys.exit(main())
