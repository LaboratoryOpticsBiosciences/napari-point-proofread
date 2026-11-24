from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np
from magicgui import magic_factory
from magicgui.widgets import Container, Label
from napari import Viewer
from napari.layers import Points
from qtpy.QtWidgets import QMessageBox
from scipy.spatial.distance import cdist

Array: TypeAlias = np.ndarray
MatchPair: TypeAlias = tuple[int, int, float]


class MatchType(Enum):
    TRUE_POSITIVE = ('TP', 'green')
    FALSE_POSITIVE = ('FP', 'red')
    FALSE_NEGATIVE = ('FN', 'blue')
    AMBIGUOUS = ('ambiguous', 'yellow')
    CLASS_MISMATCH = ('class mismatch', 'orange')

    def __init__(self, abbreviation: str, color: str):
        self.abbreviation = abbreviation
        self.color = color


@dataclass(frozen=True)
class MatchCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ambiguous: int = 0

    def fbeta(self, beta: float = 1.0) -> float:
        return compute_fbeta(
            self.tp, self.fp, self.fn, beta=beta, am=self.ambiguous
        )

    @property
    def f1(self) -> float:
        return self.fbeta()


@dataclass
class MatchingResult:
    counts: MatchCounts
    pairs: list[MatchPair]
    ambiguous: list[int]


_AMBIGUOUS_TOOLTIP_TEXT = (
    'Ambiguous matches are detections that could be matched to more than '
    'one annotation.\n'
    'This can happen when multiple annotations are within the distance '
    'threshold of a detection.\n'
    'In this case, the detection is considered ambiguous and is not '
    'counted as a true positive.'
)

_TOOLTIP_TEXT = (
    'F1 score: harmonic mean of precision and recall\n'
    'TP: True Positives are detections that match an annotation within '
    'the distance threshold.\n'
    'FP: False Positives are detections that do not match any annotation.\n'
    'FN: False Negatives are annotations that do not match any detection.\n'
    'Class-aware F1: per-class counting metric computed by matching '
    'detections and annotations of the same class only.\n'
    'Class mismatch (orange): a detection that matches an annotation '
    'location-wise but has a different class. It counts as a TP for the '
    'class-agnostic F1 but as a FP/FN pair for the class-aware F1.\n'
    f'Ambiguous: {_AMBIGUOUS_TOOLTIP_TEXT}'
)


def compute_fbeta(
    tp: int,
    fp: int,
    fn: int,
    beta: float = 1.0,
    am: int = 0,
) -> float:
    """Compute the F-beta score, penalizing ambiguous matches.

    ``beta=0`` is allowed and reduces to precision.
    """
    if beta < 0:
        raise ValueError('beta must be non-negative.')

    if tp == fp == fn == am == 0:
        return 0.0

    beta2 = beta**2
    numerator = (1 + beta2) * tp
    denominator = numerator + beta2 * fn + fp + 0.5 * (1 + beta2) * am

    return numerator / denominator


def greedy_matching(
    annotations: Array,
    detections: Array,
    threshold: float,
    detection_probabilities: Array | None = None,
    return_ambiguous: bool = False,
) -> tuple:
    """Match detections to annotations using greedy distance matching.

    Matching is performed by selecting the closest available pair.
    When distances tie, detections with higher probabilities are preferred.

    Parameters
    ----------
    annotations : ndarray
        Annotation coordinates with shape (n, d).
    detections : ndarray
        Detection coordinates with shape (m, d).
    threshold : float
        Maximum matching distance.
    detection_probabilities : ndarray, optional
        Detection probabilities used to break distance ties.
    return_ambiguous : bool
        Whether to return ambiguous detection indices.

    Returns
    -------
    tuple
        TP, FP, FN, matched pairs, and optionally ambiguous detection indices.
    """
    annotations = np.asarray(annotations)
    detections = np.asarray(detections)

    n_annotations = len(annotations)
    n_detections = len(detections)

    if detection_probabilities is not None:
        detection_probabilities = np.asarray(detection_probabilities)

        if len(detection_probabilities) != n_detections:
            raise ValueError(
                'detection_probabilities must have the same length '
                'as detections.'
            )

    if n_annotations == 0 or n_detections == 0:
        result = (
            0,
            n_detections,
            n_annotations,
            [],
        )
        return (*result, []) if return_ambiguous else result

    distances = cdist(detections, annotations)

    # Sort candidate pairs by distance, then by descending probability.
    det_indices, ann_indices = np.indices(distances.shape)
    candidates = np.column_stack(
        (
            distances.ravel(),
            det_indices.ravel(),
            ann_indices.ravel(),
        )
    )

    if detection_probabilities is not None:
        probabilities = detection_probabilities[candidates[:, 1].astype(int)]
        order = np.lexsort((-probabilities, candidates[:, 0]))
    else:
        order = np.argsort(candidates[:, 0], kind='stable')

    matched_detections = set()
    matched_annotations = set()
    pairs = []

    for candidate in candidates[order]:
        distance, det_idx, ann_idx = candidate
        det_idx, ann_idx = int(det_idx), int(ann_idx)

        if distance > threshold:
            break

        if det_idx in matched_detections or ann_idx in matched_annotations:
            continue

        pairs.append((det_idx, ann_idx, float(distance)))
        matched_detections.add(det_idx)
        matched_annotations.add(ann_idx)

    ambiguous = []

    if return_ambiguous:
        ambiguous = [
            det_idx
            for det_idx, _, _ in pairs
            if np.count_nonzero(distances[det_idx] < threshold) > 1
        ]

    tp = len(pairs)
    fp = n_detections - tp
    fn = n_annotations - tp

    result = (tp, fp, fn, pairs)
    return (*result, ambiguous) if return_ambiguous else result


def compute_class_aware_scores(
    annotations: Array,
    detections: Array,
    annotation_classes: Array,
    detection_classes: Array,
    threshold: float,
    detection_probabilities: Array | None = None,
    beta: float = 1.0,
) -> dict:
    """Compute class-agnostic and per-class detection scores."""
    annotation_classes = np.asarray(annotation_classes)
    detection_classes = np.asarray(detection_classes)

    if len(annotation_classes) != len(annotations):
        raise ValueError(
            'annotation_classes must have the same length as annotations.'
        )

    if len(detection_classes) != len(detections):
        raise ValueError(
            'detection_classes must have the same length as detections.'
        )

    # Class-agnostic matching.
    tp, fp, fn, pairs = greedy_matching(
        annotations,
        detections,
        threshold,
        detection_probabilities,
    )

    class_mismatch = sum(
        annotation_classes[ann_idx] != detection_classes[det_idx]
        for det_idx, ann_idx, _ in pairs
    )

    detection_result = {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'class_mismatch': int(class_mismatch),
        'f': compute_fbeta(tp, fp, fn, beta),
    }

    # Same-class matching.
    classes = np.union1d(annotation_classes, detection_classes)
    per_class = {}

    for cls in classes:
        ann_mask = annotation_classes == cls
        det_mask = detection_classes == cls

        cls_tp, cls_fp, cls_fn, _ = greedy_matching(
            annotations[ann_mask],
            detections[det_mask],
            threshold,
            (
                detection_probabilities[det_mask]
                if detection_probabilities is not None
                else None
            ),
        )

        per_class[cls.item() if hasattr(cls, 'item') else cls] = {
            'tp': cls_tp,
            'fp': cls_fp,
            'fn': cls_fn,
            'f': compute_fbeta(cls_tp, cls_fp, cls_fn, beta),
        }

    total_tp = sum(result['tp'] for result in per_class.values())
    total_fp = sum(result['fp'] for result in per_class.values())
    total_fn = sum(result['fn'] for result in per_class.values())

    micro = compute_fbeta(total_tp, total_fp, total_fn, beta)

    macro = (
        float(np.mean([result['f'] for result in per_class.values()]))
        if per_class
        else 0.0
    )

    return {
        'detection': detection_result,
        'per_class': per_class,
        'micro': {
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'f': micro,
        },
        'macro': macro,
    }


def _layer_world_coordinates(layer: Points) -> Array:
    """Return points in world coordinates."""
    return np.asarray([layer.data_to_world(point) for point in layer.data])


def create_points_layer_from_pairs(
    annotations: Array,
    detections: Array,
    pairs: list[MatchPair],
    ambiguous: list[int] | None = None,
    annotation_classes: Array | None = None,
    detection_classes: Array | None = None,
) -> Points:
    """Create a napari Points layer visualizing matching results."""
    ambiguous = set(ambiguous or [])

    has_classes = (
        annotation_classes is not None and detection_classes is not None
    )

    if has_classes:
        annotation_classes = np.asarray(annotation_classes, dtype=int)
        detection_classes = np.asarray(detection_classes, dtype=int)
    else:
        annotation_classes = np.full(len(annotations), -1, dtype=int)
        detection_classes = np.full(len(detections), -1, dtype=int)

    matched_det = {det_idx for det_idx, _, _ in pairs}
    matched_ann = {ann_idx for _, ann_idx, _ in pairs}

    coords = []
    types = []
    distances = []
    face_colors = []
    border_colors = []
    class_matches = []
    ann_classes = []
    det_classes = []

    def append_point(
        coordinate,
        match_type: MatchType,
        distance: float = np.nan,
        ann_class: int = -1,
        det_class: int = -1,
        is_ambiguous: bool = False,
    ):
        coords.append(coordinate)
        types.append(match_type.abbreviation)
        distances.append(distance)
        face_colors.append(match_type.color)
        border_colors.append(
            MatchType.AMBIGUOUS.color if is_ambiguous else 'transparent'
        )
        class_matches.append(match_type == MatchType.TRUE_POSITIVE)
        ann_classes.append(ann_class)
        det_classes.append(det_class)

    for det_idx, ann_idx, distance in pairs:
        ann_class = int(annotation_classes[ann_idx])
        det_class = int(detection_classes[det_idx])

        mismatch = has_classes and ann_class != det_class
        match_type = (
            MatchType.CLASS_MISMATCH if mismatch else MatchType.TRUE_POSITIVE
        )

        append_point(
            detections[det_idx],
            match_type,
            distance,
            ann_class,
            det_class,
            det_idx in ambiguous,
        )

    for det_idx, coordinate in enumerate(detections):
        if det_idx not in matched_det:
            append_point(
                coordinate,
                MatchType.FALSE_POSITIVE,
                det_class=int(detection_classes[det_idx]),
                is_ambiguous=det_idx in ambiguous,
            )

    for ann_idx, coordinate in enumerate(annotations):
        if ann_idx not in matched_ann:
            append_point(
                coordinate,
                MatchType.FALSE_NEGATIVE,
                ann_class=int(annotation_classes[ann_idx]),
            )

    return Points(
        np.asarray(coords),
        name='Greedy matching',
        ndim=annotations.shape[1],
        properties={
            'type': types,
            'distance': distances,
            'ambiguous': [border != 'transparent' for border in border_colors],
            'class_match': class_matches,
            'annotation_class': ann_classes,
            'detection_class': det_classes,
        },
        face_color=face_colors,
        border_color=border_colors,
        size=5,
    )


def compute_score() -> Container:
    """Build the napari score widget."""

    score_label = Label(value='F1 score: -')
    score_label.label = ''
    score_label.tooltip = _TOOLTIP_TEXT

    @magic_factory(
        greedy_threshold={
            'widget_type': 'FloatSlider',
            'min': 0.0,
            'max': 100.0,
            'step': 1.0,
            'value': 10.0,
        },
        return_ambiguous={
            'widget_type': 'CheckBox',
            'label': 'Show ambiguous matches',
            'tooltip': _AMBIGUOUS_TOOLTIP_TEXT,
        },
        auto_call=True,
        call_button='Show greedy matching',
    )
    def _compute_score(
        viewer: Viewer,
        annotation_layer: Points,
        detection_layer: Points,
        greedy_threshold: float = 10.0,
        return_ambiguous: bool = False,
    ) -> Points | None:
        """Compute matching scores and display them in napari."""

        if annotation_layer is None or detection_layer is None:
            score_label.value = 'F1 score: -'
            return None

        annotations = _layer_world_coordinates(annotation_layer)
        detections = _layer_world_coordinates(detection_layer)

        if not len(annotations) or not len(detections):
            score_label.value = 'F1 score: -'
            QMessageBox.warning(
                None,
                'Empty layer',
                'One of the layers is empty. Please provide non-empty '
                'annotation and detection layers.',
            )
            return None

        probabilities = (
            detection_layer.features['probability'].to_numpy()
            if 'probability' in detection_layer.features
            else None
        )

        result = greedy_matching(
            annotations,
            detections,
            greedy_threshold,
            probabilities,
            return_ambiguous,
        )

        if return_ambiguous:
            tp, fp, fn, pairs, ambiguous = result
        else:
            tp, fp, fn, pairs = result
            ambiguous = []

        score = compute_fbeta(tp, fp, fn, beta=1.0, am=len(ambiguous))

        class_text = ''
        annotation_class_values = None
        detection_class_values = None

        if (
            'class' in annotation_layer.features
            and 'class' in detection_layer.features
        ):
            annotation_class_values = (
                annotation_layer.features['class'].to_numpy().astype(int)
            )
            detection_class_values = (
                detection_layer.features['class'].to_numpy().astype(int)
            )

            if len(annotation_class_values) == len(annotations) and len(
                detection_class_values
            ) == len(detections):
                class_scores = compute_class_aware_scores(
                    annotations,
                    detections,
                    annotation_class_values,
                    detection_class_values,
                    greedy_threshold,
                    probabilities,
                )

                mismatch = class_scores['detection']['class_mismatch']

                per_class = ', '.join(
                    f'class {cls}: {counts["f"]:.3f}'
                    for cls, counts in class_scores['per_class'].items()
                )

                class_text = (
                    f'<br>Class-aware F1: '
                    f'{class_scores["micro"]["f"]:.3f} '
                    f'(macro: {class_scores["macro"]:.3f}) '
                    f'[{per_class}]'
                    f'<br>Class mismatch: '
                    f'<span style="color: '
                    f'{MatchType.CLASS_MISMATCH.color};">'
                    f'{mismatch}</span>'
                )

        score_label.value = (
            f'F1 score: {score:.3f}<br>'
            f'TP: <span style="color: green;">{tp}</span>, '
            f'FP: <span style="color: red;">{fp}</span>, '
            f'FN: <span style="color: blue;">{fn}</span>, '
            f'Ambiguous: <span style="color: yellow;">'
            f'{len(ambiguous)}</span>'
            f'{class_text}'
        )

        name = (
            f'Greedy matching (annot: {annotation_layer.name}, '
            f'det: {detection_layer.name})'
        )

        existing = next(
            (layer for layer in viewer.layers if layer.name == name),
            None,
        )

        if existing is not None:
            viewer.layers.remove(existing)

        points = create_points_layer_from_pairs(
            annotations,
            detections,
            pairs,
            ambiguous,
            annotation_class_values,
            detection_class_values,
        )

        points.name = name
        viewer.add_layer(points)

        return points

    return Container(
        widgets=[_compute_score(), score_label],
        labels=False,
    )
