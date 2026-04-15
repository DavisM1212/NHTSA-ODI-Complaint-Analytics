# Modeling Workflow

Start here if you want the modeling pipeline order without tracing imports

## Entrypoints

These files own the runnable workflow and output artifacts

1. `tune_component_catboost.py`
   - Purpose: choose the locked single-label CatBoost feature set and hyperparameters
   - Reads: single-label processed input
   - Writes: `component_single_label_selection_manifest.json` and tuning/search tables

2. `component_catboost.py`
   - Purpose: run the locked single-label benchmark suite
   - Reads: single-label processed input and the selection manifest from step 1
   - Writes: single-label benchmark metrics, calibration, confusion, class metrics, feature importance, benchmark manifest

3. `component_multilabel.py`
   - Purpose: run the locked multi-label benchmark suite
   - Reads: multi-label processed input
   - Writes: multi-label metrics, label metrics, split summary, manifest

4. `component_feature_wave1.py`
   - Purpose: compare structured feature families against the locked single-label and multi-label baselines
   - Reads: the locked single-label benchmark manifest, selection manifest, calibration file, and multi-label manifest
   - Writes: Wave 1 screen/select/holdout outputs plus a Wave 1 manifest
   - Note: this is a structured feature sweep branch, not a prerequisite for text Wave 2

5. `component_text_wave2.py`
   - Purpose: compare text, text-plus-structured, and late-fusion families against the locked baselines
   - Reads: single-label benchmark outputs, multi-label benchmark outputs, processed text sidecar, and processed single/multi inputs
   - Writes: Wave 2 screen/select/holdout outputs plus a Wave 2 manifest

6. `component_text_wave2b_calibration.py`
   - Purpose: calibrate the selected Wave 2 single-label model
   - Reads: Wave 2 manifest and the same processed single-label/text inputs used by Wave 2
   - Writes: calibrated holdout outputs, calibration search grid, and calibration manifest

## Shared Helper Files

These are not entrypoints and should usually be read only if you need implementation detail

- `component_common.py`
  - Broad shared modeling utilities, feature manifests, splits, and scoring helpers
- `component_multilabel_shared.py`
  - Narrow shared multilabel CatBoost helpers reused by the multilabel benchmark, Wave 1, and the Wave 2 helper layer
- `component_tuning_shared.py`
  - Narrow shared cross-seed tuning helpers reused by the CatBoost tuner and Wave 1
- `component_text_shared.py`
  - Reusable Wave 2 text and fusion helper layer used by the Wave 2 entrypoint and calibration step

## Reading Order

If you want to understand the pipeline quickly, read files in this order

1. `src/modeling/README.md`
2. The entrypoint you plan to run
3. The locked input or manifest constants near the top of that entrypoint
4. Only then open the matching `*_shared.py` helper if a specific implementation detail matters

## Practical Rule

To understand workflow order, stay in entrypoints

To understand reusable implementation details, open the matching shared helper
