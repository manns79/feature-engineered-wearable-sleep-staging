import numpy as np
import pandas as pd
from src.config import TARGET_LABELS
from src.models.calibration import (
    OneVsRestPlattCalibrator,
    align_probabilities,
    predicted_labels,
)
from src.models.sequence_postprocessing import (
    apply_viterbi_by_participant,
    estimate_transition_model,
    viterbi_decode,
)


def test_align_probabilities_reorders_sklearn_classes_to_target_labels():
    probabilities = np.array([[0.2, 0.3, 0.5]])
    aligned = align_probabilities(probabilities, ["Non-REM", "REM", "Wake"])

    assert aligned.tolist() == [[0.5, 0.2, 0.3]]


def test_platt_calibrator_outputs_normalized_probabilities():
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.1, 0.8],
        ]
    )
    labels = ["Wake", "Wake", "Non-REM", "Non-REM", "REM", "REM"]

    calibrator = OneVsRestPlattCalibrator(classes=TARGET_LABELS).fit(
        probabilities, labels
    )
    calibrated = calibrator.predict_proba(probabilities)

    assert calibrated.shape == probabilities.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert predicted_labels(calibrated) == labels


def test_transition_model_and_viterbi_decode_by_participant():
    labels = pd.DataFrame(
        {
            "participant_id": ["S001", "S001", "S001", "S002", "S002", "S002"],
            "epoch_id": [0, 1, 2, 0, 1, 2],
            "label": ["Wake", "Non-REM", "REM", "Wake", "Non-REM", "REM"],
        }
    )
    transition_model = estimate_transition_model(labels, smoothing=0.1)
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )

    assert viterbi_decode(probabilities, transition_model) == [
        "Wake",
        "Non-REM",
        "REM",
    ]

    predictions = labels.rename(columns={"label": "true_label"}).copy()
    predictions["split"] = "validation"
    for index, label in enumerate(TARGET_LABELS):
        predictions[f"prob_{label.replace('-', '_')}"] = (
            list(probabilities[:, index]) * 2
        )
    decoded = apply_viterbi_by_participant(predictions, transition_model)

    assert decoded["pred_label"].tolist() == [
        "Wake",
        "Non-REM",
        "REM",
        "Wake",
        "Non-REM",
        "REM",
    ]
