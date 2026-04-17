import pandas as pd

# -----------------------------------------------------------------------------
# Split modes
# -----------------------------------------------------------------------------
BENCHMARK_SPLIT_MODE = 'benchmark_v1'
FEATURE_WAVE1_SPLIT_MODE = 'feature_wave1'


# -----------------------------------------------------------------------------
# Stable data-window anchors
# -----------------------------------------------------------------------------
TRAIN_END = pd.Timestamp('2024-12-31')
VALID_END = pd.Timestamp('2025-12-31')
TRAIN_CORE_END = pd.Timestamp('2023-12-31')
SCREEN_END = pd.Timestamp('2024-12-31')
SELECT_END = pd.Timestamp('2025-12-31')
REFERENCE_DATA_YEAR = 2026
REFERENCE_MODEL_YEAR_MAX = REFERENCE_DATA_YEAR + 1


# -----------------------------------------------------------------------------
# Split contracts
# -----------------------------------------------------------------------------
SPLIT_POLICIES = {
    BENCHMARK_SPLIT_MODE: {
        'train_end': TRAIN_END,
        'valid_end': VALID_END,
        'train_name': 'train',
        'valid_name': 'valid_2025',
        'holdout_name': 'holdout_2026',
        'selection_train_name': 'train',
        'selection_eval_name': 'valid_2025',
        'dev_name': 'dev_2020_2025',
        'holdout_policy': '2026 holdout untouched during official benchmark selection'
    },
    FEATURE_WAVE1_SPLIT_MODE: {
        'train_core_end': TRAIN_CORE_END,
        'screen_end': SCREEN_END,
        'select_end': SELECT_END,
        'train_name': 'train_core',
        'screen_name': 'screen_2024',
        'select_name': 'select_2025',
        'holdout_name': 'holdout_2026',
        'selection_train_name': 'train_core',
        'selection_eval_name': 'screen_2024',
        'select_train_name': 'dev_2020_2024',
        'dev_name': 'dev_2020_2025',
        'holdout_policy': '2026 holdout untouched during feature-family screening and promotion'
    }
}


def get_split_policy(split_mode=BENCHMARK_SPLIT_MODE):
    if split_mode not in SPLIT_POLICIES:
        choices = ', '.join(sorted(SPLIT_POLICIES))
        raise ValueError(f'Unknown split_mode {split_mode}. Choices: {choices}')
    return SPLIT_POLICIES[split_mode]
