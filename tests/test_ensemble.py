import numpy as np

from aigc_detector.experiments.ensemble import select_ensemble_policy


def test_select_ensemble_policy_obeys_per_dataset_fpr_cap():
    labels = np.array([0, 0, 1, 1])
    datasets = {
        "a": (labels, np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.2, 0.3, 0.7, 0.8])),
        "b": (labels, np.array([0.1, 0.4, 0.6, 0.9]), np.array([0.1, 0.2, 0.8, 0.9])),
    }
    selected, rows = select_ensemble_policy(datasets, [0.0, 0.5, 1.0], max_real_fpr=0.0)
    assert len(rows) == 3
    assert selected["worst_dataset_real_fpr"] == 0.0
    assert selected["macro_balanced_accuracy"] == 1.0


def test_selection_macro_averages_datasets_instead_of_images():
    small_labels = np.array([0, 1])
    large_labels = np.array([0] * 100 + [1] * 100)
    datasets = {
        "small": (small_labels, np.array([0.1, 0.9]), np.array([0.1, 0.9])),
        "large": (
            large_labels,
            np.concatenate([np.full(100, 0.1), np.full(100, 0.9)]),
            np.concatenate([np.full(100, 0.1), np.full(100, 0.9)]),
        ),
    }
    selected, _ = select_ensemble_policy(datasets, [0.5], max_real_fpr=0.05)
    assert selected["macro_balanced_accuracy"] == 1.0
