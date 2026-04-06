import numpy as np

from src.modeling.component_text_wave2b_calibration import (
    apply_power_calibration,
    parse_float_list,
    select_calibration_alpha,
)


def test_power_calibration_preserves_row_sums_and_ranking():
    proba = np.array(
        [
            [0.60, 0.30, 0.10],
            [0.20, 0.70, 0.10],
        ]
    )

    calibrated = apply_power_calibration(proba, alpha=2.0)

    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.array_equal(np.argmax(calibrated, axis=1), np.argmax(proba, axis=1))
    assert calibrated[0, 0] > proba[0, 0]
    assert calibrated[1, 1] > proba[1, 1]


def test_power_calibration_identity_at_alpha_one():
    proba = np.array(
        [
            [0.60, 0.30, 0.10],
            [0.20, 0.70, 0.10],
        ]
    )

    calibrated = apply_power_calibration(proba, alpha=1.0)

    assert np.allclose(calibrated, proba)


def test_select_calibration_alpha_can_sharpen_underconfident_probabilities():
    y_true = np.array(['A', 'B', 'C', 'A'])
    classes = np.array(['A', 'B', 'C', 'D'])
    underconfident = np.array(
        [
            [0.45, 0.30, 0.15, 0.10],
            [0.25, 0.45, 0.20, 0.10],
            [0.20, 0.25, 0.45, 0.10],
            [0.45, 0.20, 0.25, 0.10],
        ]
    )

    selected, grid_df = select_calibration_alpha(
        y_true,
        underconfident,
        classes,
        [1.0, 2.0, 4.0],
    )

    assert selected['calibration_alpha'] > 1.0
    assert len(grid_df) == 3
    assert set(grid_df['calibration_alpha']) == {1.0, 2.0, 4.0}


def test_parse_float_list_rejects_nonpositive_values():
    assert parse_float_list('0.5,1,2') == [0.5, 1.0, 2.0]

    try:
        parse_float_list('1,0')
        assert False, 'Expected zero alpha to fail'
    except Exception as exc:
        assert 'positive' in str(exc)
