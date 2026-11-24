import numpy as np
import pytest
from napari.layers import Points

from napari_point_proofread._score_widget import (
    compute_class_aware_scores,
    compute_fbeta,
    create_points_layer_from_pairs,
    greedy_matching,
)


@pytest.fixture
def viewer(make_napari_viewer_proxy):
    return make_napari_viewer_proxy()


@pytest.fixture
def annotation_layer(viewer):
    layer = Points(
        np.array([[0, 0], [1, 1], [11, 11], [111, 111]]), name='test_points'
    )
    viewer.add_layer(layer)
    return layer


@pytest.fixture
def prediction_layer(viewer):
    probabilities_features = np.array(
        [0.3, 0.3, 0.7, 0.9]
    )  # arbitrary probabilities
    layer = Points(
        np.array([[0, 0], [1, 1], [12, 11], [11, 12]]), name='test_points'
    )
    layer.features['probability'] = probabilities_features
    viewer.add_layer(layer)
    return layer


def test_greedy_matching(annotation_layer, prediction_layer):
    """Test that the greedy matching algorithm works correctly."""
    distance_threshold = 2
    tp, fp, fn, distance_pairs, ambiguous = greedy_matching(
        annotation_layer.data,
        prediction_layer.data,
        distance_threshold,
        prediction_layer.features['probability'].to_numpy(),
        return_ambiguous=True,
    )
    assert tp == 3
    assert fp == 1
    assert fn == 1
    assert len(distance_pairs) == 3
    assert len(ambiguous) == 2

    np.testing.assert_array_equal(
        ambiguous,
        [
            0,
            1,
        ],
    )  # indexes of the ambiguous pairs in the predictions
    # assert distance_pairs == [[0, 0, 0.0], [1, 1, 0.0], [3, 2, 1.0]]  # indexes of the matched pairs
    np.testing.assert_array_equal(
        distance_pairs,
        [[0, 0, 0.0], [1, 1, 0.0], [3, 2, 1.0]],
        err_msg='Distance pairs do not match',
    )


def test_create_points_layer_from_pairs(annotation_layer, prediction_layer):
    """Test that the points layer is created correctly from the matched pairs."""
    distance_threshold = 2
    tp, fp, fn, distance_pairs, ambiguous = greedy_matching(
        annotation_layer.data,
        prediction_layer.data,
        distance_threshold,
        prediction_layer.features['probability'].to_numpy(),
        return_ambiguous=True,
    )
    points_layer = create_points_layer_from_pairs(
        annotation_layer.data, prediction_layer.data, distance_pairs, ambiguous
    )

    assert isinstance(points_layer, Points)
    assert points_layer.name == 'Greedy matching'
    np.testing.assert_array_equal(
        points_layer.data, [[0, 0], [1, 1], [11, 12], [12, 11], [111, 111]]
    )  # matched pairs

    # check features
    print(points_layer.features)
    assert 'type' in points_layer.features

    # check that the features are correct
    np.testing.assert_array_equal(
        points_layer.features['type'], ['TP', 'TP', 'TP', 'FP', 'FN']
    )

    print(points_layer.features['distance'].to_numpy())
    # check that the distances are correct
    np.testing.assert_array_equal(
        points_layer.features['distance'].to_numpy(),
        [0.0, 0.0, 1.0, np.nan, np.nan],
        err_msg='Distances do not match',
    )

    # check that the ambiguous feature is correct
    np.testing.assert_array_equal(
        points_layer.features['ambiguous'], [True, True, False, False, False]
    )


def test_compute_fbeta():
    """Test the F-beta score formula, including the F1 special case."""
    assert compute_fbeta(3, 1, 1) == pytest.approx(
        compute_fbeta(3, 1, 1, beta=1.0)
    )
    # beta = 0 reduces to precision (TP / (TP + FP))
    assert compute_fbeta(3, 1, 1, beta=0.0) == pytest.approx(0.75)
    # beta weights recall higher: with equal counts the score stays 0.75, but
    # moving one FN to FP should lower the beta=2 score.
    assert compute_fbeta(3, 2, 0) > compute_fbeta(3, 0, 2, beta=2.0)
    # Empty confusion matrix yields 0.
    assert compute_fbeta(0, 0, 0) == 0.0


def test_compute_class_aware_scores():
    """Test class-aware scoring penalizes misclassification.

    Two annotations (class 1 and 2) are detected at the same positions but
    the classes are swapped, so the class-agnostic F1 is 1 while the
    class-aware per-class F1 is 0 for both classes.
    """
    annotations = np.array([[0.0, 0.0], [10.0, 10.0]])
    detections = np.array([[0.0, 0.0], [10.0, 10.0]])
    annotation_classes = np.array([1, 2])
    detection_classes = np.array([2, 1])

    scores = compute_class_aware_scores(
        annotations=annotations,
        detections=detections,
        annotation_classes=annotation_classes,
        detection_classes=detection_classes,
        threshold=1.0,
    )

    # Detection quality is perfect, independent of class.
    assert scores['detection']['f'] == pytest.approx(1.0)
    assert scores['detection']['tp'] == 2

    # Class-aware: every point is a FP for its predicted class and a FN for
    # its true class.
    assert scores['per_class'][1]['tp'] == 0
    assert scores['per_class'][1]['fp'] == 1
    assert scores['per_class'][1]['fn'] == 1
    assert scores['per_class'][2]['tp'] == 0
    assert scores['per_class'][1]['f'] == 0.0
    assert scores['per_class'][2]['f'] == 0.0
    assert scores['micro']['f'] == 0.0
    assert scores['macro'] == 0.0


def test_compute_class_aware_scores_reports_mismatch():
    """class_mismatch counts location matches with a different class."""
    annotations = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    detections = np.array([[0.0, 0.0], [10.0, 10.0]])
    annotation_classes = np.array([1, 2, 1])
    detection_classes = np.array([1, 1])

    scores = compute_class_aware_scores(
        annotations=annotations,
        detections=detections,
        annotation_classes=annotation_classes,
        detection_classes=detection_classes,
        threshold=1.0,
    )

    # Two location-wise matches, one of which has the wrong class.
    assert scores['detection']['tp'] == 2
    assert scores['detection']['class_mismatch'] == 1


def test_create_points_layer_class_mismatch(
    annotation_layer, prediction_layer
):
    """Matched points with different classes are flagged orange."""
    distance_threshold = 2
    _, _, _, distance_pairs, _ = greedy_matching(
        annotation_layer.data,
        prediction_layer.data,
        distance_threshold,
        prediction_layer.features['probability'].to_numpy(),
        return_ambiguous=True,
    )

    annotation_classes = np.array([1, 1, 2, 2])
    detection_classes = np.array([1, 1, 2, 1])  # last match is a mismatch

    points_layer = create_points_layer_from_pairs(
        annotation_layer.data,
        prediction_layer.data,
        distance_pairs,
        annotation_classes=annotation_classes,
        detection_classes=detection_classes,
    )

    assert 'class_match' in points_layer.features
    assert 'annotation_class' in points_layer.features
    assert 'detection_class' in points_layer.features

    types = list(points_layer.features['type'])
    # Matched pairs in order: det0->ann0, det1->ann1, det3->ann2
    assert types[0] == 'TP'
    assert types[1] == 'TP'
    assert types[2] == 'class mismatch'

    np.testing.assert_array_equal(
        points_layer.features['class_match'],
        [True, True, False, False, False],
    )


def test_create_points_layer_without_classes(
    annotation_layer, prediction_layer
):
    """Without class info the layer still behaves class-agnostically."""
    _, _, _, distance_pairs, _ = greedy_matching(
        annotation_layer.data,
        prediction_layer.data,
        2,
        prediction_layer.features['probability'].to_numpy(),
        return_ambiguous=True,
    )

    points_layer = create_points_layer_from_pairs(
        annotation_layer.data, prediction_layer.data, distance_pairs
    )

    # Unknown classes are marked with -1 and never flagged as mismatches.
    np.testing.assert_array_equal(
        points_layer.features['annotation_class'][:3], [-1, -1, -1]
    )
    np.testing.assert_array_equal(
        points_layer.features['class_match'], [True, True, True, False, False]
    )
