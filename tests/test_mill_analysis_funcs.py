"""
Unit tests for mill_analysis_funcs.py.

Run from the project directory with:
    pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from mill_analysis_funcs import (
    calculate_avg_magnitude,
    calculate_peak_magnitude,
    convert_data,
    predict_flank_wear,
    predict_time_to_threshold,
)


def make_mat_data(records):
    """Build a small MATLAB-structured input similar to scipy.io.loadmat output."""
    # Each field is a NumPy object array with one item per primary record.
    field_names = list(records[0].keys())
    dtype = [(name, object) for name in field_names]
    structured = np.empty((1,), dtype=dtype)

    for field in field_names:
        values = np.empty((len(records),), dtype=object)
        for i, record in enumerate(records):
            values[i] = record[field]
        structured[field][0] = values

    return {"mill": structured}


def make_record(case, vb, time=1.0, signal=None):
    """Create one synthetic record with fields used by convert_data."""
    if signal is None:
        signal = np.array([1.0, -2.0, 3.0])
    return {
        "case": case,
        "VB": vb,
        "time": time,
        "DOC": 0.75,
        "feed": 0.25,
        "material": 1,
        "vib_table": np.asarray(signal),
        "vib_spindle": np.asarray(signal),
        "AE_table": np.asarray(signal),
        "AE_spindle": np.asarray(signal),
        "smcAC": np.asarray(signal),
        "smcDC": np.asarray(signal),
    }


# ---------------------------------------------------------------------------
# Signal magnitude functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("signal", "expected"),
    [
        ([-2, 4, -6], 4.0),
        ([0, 0, 0], 0.0),
        ([3.5], 3.5),
        ([-1.5, 2.5], 2.0),
    ],
)
def test_calculate_avg_magnitude(signal, expected):
    assert calculate_avg_magnitude(signal) == pytest.approx(expected)


def test_calculate_avg_magnitude_empty_array():
    assert calculate_avg_magnitude([]) == 0.0


@pytest.mark.parametrize(
    ("signal", "expected"),
    [
        ([-8, 3, -2, 5], 8.0),
        ([0, 0, 0], 0.0),
        ([3.5], 3.5),
        ([-1.5, 2.5], 2.5),
    ],
)
def test_calculate_peak_magnitude(signal, expected):
    assert calculate_peak_magnitude(signal) == pytest.approx(expected)


def test_calculate_peak_magnitude_empty_array():
    assert calculate_peak_magnitude([]) == 0.0


# ---------------------------------------------------------------------------
# Data conversion and filtering
# ---------------------------------------------------------------------------

def test_convert_data_keeps_cases_with_multiple_valid_vb_rows():
    records = [
        make_record(case=1, vb=0.10, time=1),
        make_record(case=1, vb=0.20, time=2),
        make_record(case=2, vb=0.30, time=1),
        make_record(case=3, vb=np.nan, time=1),
        make_record(case=3, vb=0.40, time=2),
        make_record(case=3, vb=0.50, time=3),
    ]

    result = convert_data(make_mat_data(records))

    # Case 2 has only one non-NaN VB row and should be removed.
    # Case 3's NaN VB row should be removed, while its two valid rows remain.
    assert isinstance(result, pd.DataFrame)
    assert result["case"].tolist() == [1, 1, 3, 3]
    assert result["VB"].tolist() == pytest.approx([0.10, 0.20, 0.40, 0.50])
    assert result["time"].tolist() == [1, 2, 2, 3]


def test_convert_data_flattens_time_series_fields():
    signal = np.array([[1, -2], [3, -4]])
    records = [
        make_record(case=1, vb=0.1, time=1, signal=signal),
        make_record(case=1, vb=0.2, time=2, signal=signal),
    ]

    result = convert_data(make_mat_data(records))

    np.testing.assert_array_equal(result.iloc[0]["vib_spindle"], [1, -2, 3, -4])
    assert isinstance(result.iloc[0]["vib_spindle"], np.ndarray)


# ---------------------------------------------------------------------------
# Interactive prediction functions
# ---------------------------------------------------------------------------

class LinearWearModel:
    """Simple deterministic stand-in model for testing prediction functions."""

    def __init__(self, intercept=0.1, slope=0.01):
        self.intercept = intercept
        self.slope = slope
        self.last_input = None

    def predict(self, data):
        self.last_input = data.copy()
        # The model expects the same feature names as the application.
        return self.intercept + self.slope * data["time"].to_numpy()


def test_predict_flank_wear_returns_model_prediction(monkeypatch):
    # DOC, feed, material, time
    answers = iter(["0.75", "0.25", "1", "10"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    model = LinearWearModel(intercept=0.1, slope=0.01)

    result = predict_flank_wear(model)

    assert result == pytest.approx(0.2)
    assert model.last_input.iloc[0]["DOC"] == pytest.approx(0.75)
    assert model.last_input.iloc[0]["feed"] == pytest.approx(0.25)
    assert model.last_input.iloc[0]["material"] == 1
    assert model.last_input.iloc[0]["time"] == pytest.approx(10)


def test_predict_flank_wear_reprompts_for_invalid_inputs(monkeypatch):
    # Invalid DOC text, non-positive DOC, invalid feed, invalid material,
    # then valid values; negative time is rejected before accepting zero.
    answers = iter(["abc", "0", "0.75", "bad", "0.25", "3", "2", "-1", "0"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    model = LinearWearModel()

    result = predict_flank_wear(model)

    assert result == pytest.approx(0.1)
    assert model.last_input.iloc[0]["time"] == pytest.approx(0)


def test_predict_time_to_threshold_finds_known_time(monkeypatch):
    # VB(t) = 0.1 + 0.01*t, so VB = 0.3 at t = 20 minutes.
    answers = iter(["0.3", "0.75", "0.25", "1"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    model = LinearWearModel(intercept=0.1, slope=0.01)

    result = predict_time_to_threshold(model)

    assert result == pytest.approx(20.0, abs=1e-6)


def test_predict_time_to_threshold_returns_none_if_threshold_not_reached(monkeypatch, capsys):
    # This model predicts VB from 0.1 to 0.2 over t = 0..200;
    # a threshold of 0.5 is not reached in that interval.
    answers = iter(["0.5", "0.75", "0.25", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    model = LinearWearModel(intercept=0.1, slope=0.0005)

    result = predict_time_to_threshold(model)

    assert result is None
    assert "not reached" in capsys.readouterr().out.lower()


def test_predict_time_to_threshold_threshold_already_exceeded_at_start(
    monkeypatch, capsys
):
    # At t = 0, this model predicts VB = 0.4, above the target of 0.3.
    answers = iter(["0.3", "0.75", "0.25", "1"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    model = LinearWearModel(intercept=0.4, slope=0.001)

    result = predict_time_to_threshold(model)

    assert result is None
    assert "already above the threshold" in capsys.readouterr().out.lower()
