import numpy as np
import pytest
from napari.utils.key_bindings import coerce_keybinding

from napari_point_proofread._proofread_widget import (
    _CLASS_COLORS_HEX,
    _CLASS_COLORS_RGBA,
    ProofreadWidget,
)

NAV_KEYS = ('Left', 'Right', 'Up', 'Down')


@pytest.fixture
def viewer(make_napari_viewer_proxy):
    return make_napari_viewer_proxy()


@pytest.fixture
def make_widget(viewer):
    """Build a docked ProofreadWidget bound to a fresh points layer.

    Returns ``(widget, class_widget, layer)``. ``classes`` optionally
    pre-populates the ``class`` feature column.
    """

    def _make(data=((0, 0), (1, 1), (2, 2)), classes=None):
        features = {'class': classes} if classes is not None else None
        layer = viewer.add_points(
            np.asarray(data, dtype=float), features=features
        )
        widget = ProofreadWidget()
        viewer.window.add_dock_widget(widget)
        # `widget_points` is a module-level singleton: bind it explicitly so
        # tests don't depend on leftover state from earlier tests.
        widget.points_widget.reset_choices()
        widget.points_widget.point_layer.value = layer
        return widget, widget.point_class_widget, layer

    return _make


@pytest.fixture(autouse=True)
def _restore_nav_keys(viewer):
    saved = {k: viewer.keymap.get(coerce_keybinding(k)) for k in NAV_KEYS}
    yield
    for key, callback in saved.items():
        viewer.bind_key(key, callback, overwrite=True)


def classes(layer):
    return layer.features['class'].astype(int).tolist()


def test_enable_creates_class_column_and_colors(make_widget):
    _, pcw, layer = make_widget(classes=[0, 3, 5])
    pcw._enable_cb.value = True

    assert classes(layer) == [0, 3, 5]  # existing values are preserved
    for i, c in enumerate([0, 3, 5]):
        assert np.allclose(layer.face_color[i], _CLASS_COLORS_RGBA[c])
    assert layer.text.visible


def test_disable_hides_text_and_resets_color(make_widget):
    _, pcw, layer = make_widget()
    pcw._enable_cb.value = True
    pcw._enable_cb.value = False

    assert not layer.text.visible
    assert np.allclose(layer.face_color, 1.0)  # white


def test_switching_layer_deactivates_previous(make_widget, viewer):
    widget, pcw, layer_a = make_widget()
    layer_b = viewer.add_points(np.array([[5, 5], [6, 6]], dtype=float))
    widget.points_widget.reset_choices()

    pcw._enable_cb.value = True
    widget.points_widget.point_layer.value = layer_b

    assert not layer_a.text.visible
    assert layer_b.text.visible
    assert classes(layer_b) == [0, 0]


def test_assign_button_retags_only_selected(make_widget):
    _, pcw, layer = make_widget()
    pcw._enable_cb.value = True

    layer.selected_data = {1}
    pcw._class_spin.value = 1
    pcw._on_assign(True)

    assert classes(layer) == [0, 1, 0]
    assert np.allclose(layer.face_color[1], _CLASS_COLORS_RGBA[1])


def test_manual_spin_change_retags_selection_and_sets_default(make_widget):
    """Manual edits must flow through `_on_class_changed`."""
    _, pcw, layer = make_widget()
    pcw._enable_cb.value = True

    layer.selected_data = {2}
    pcw._class_spin.value = 4

    assert classes(layer) == [0, 0, 4]
    assert layer.current_properties['class'][0] == 4
    assert pcw._color_label.tooltip == _CLASS_COLORS_HEX[4]


def test_new_points_inherit_spin_class(make_widget):
    _, pcw, layer = make_widget()
    pcw._enable_cb.value = True

    layer.selected_data = set()  # otherwise the spin change retags a point
    pcw._class_spin.value = 2
    layer.add([5.0, 5.0])

    assert len(layer.data) == 4
    assert classes(layer) == [0, 0, 0, 2]


def test_removing_points_keeps_classes_aligned(make_widget):
    _, pcw, layer = make_widget(classes=[0, 3, 5])
    pcw._enable_cb.value = True

    layer.selected_data = {1}
    layer.remove_selected()

    assert classes(layer) == [0, 5]
    assert len(layer.face_color) == 2


def test_navigation_selects_first_point_and_mirrors_class(make_widget):
    _, pcw, layer = make_widget(classes=[2, 3, 5])
    pcw._enable_cb.value = True
    pcw._navigation_cb.value = True

    assert layer.selected_data == {0}
    assert pcw._class_spin.value == 2


def test_navigation_on_empty_layer_reverts_checkbox(make_widget):
    _, pcw, _ = make_widget(data=np.empty((0, 2)))
    pcw._enable_cb.value = True
    pcw._navigation_cb.value = True

    assert pcw._navigation_cb.value is False


def test_move_point_clamps_at_boundaries(make_widget):
    _, pcw, layer = make_widget()
    pcw._enable_cb.value = True
    pcw._navigation_cb.value = True

    pcw._move_point(-1)
    assert layer.selected_data == {0}

    pcw._move_point(1)
    pcw._move_point(1)
    pcw._move_point(1)  # past the end
    assert layer.selected_data == {2}


def test_move_point_updates_dims_and_camera(make_widget, viewer):
    _, pcw, layer = make_widget(data=[(0, 0, 0), (1, 2, 3), (4, 5, 6)])
    pcw._enable_cb.value = True
    pcw._navigation_cb.value = True

    pcw._move_point(1)
    assert viewer.dims.point[0] == 1
    np.testing.assert_allclose(viewer.scene.camera.center[1:], (2, 3))

    pcw._move_point(1)
    assert viewer.dims.point[0] == 4
    np.testing.assert_allclose(viewer.scene.camera.center[1:], (5, 6))


def test_navigating_never_modifies_classes(make_widget):
    """Mirroring a point's class into the spin box must not retag points.

    This is the regression the blocked-signal design exists to prevent.
    """
    _, pcw, layer = make_widget(classes=[0, 3, 5])
    pcw._enable_cb.value = True
    pcw._navigation_cb.value = True

    for step in (1, 1, -1, -1):
        pcw._move_point(step)

    assert classes(layer) == [0, 3, 5]
    assert pcw._class_spin.value == 0  # mirrors the final selected point


def test_shift_class_updates_features_directly(make_widget):
    _, pcw, layer = make_widget(classes=[0, 3, 5])
    pcw._enable_cb.value = True
    pcw._navigation_cb.value = True
    pcw._move_point(1)  # select the point with class 3

    pcw._shift_class(1)

    assert classes(layer) == [0, 4, 5]  # neighbours untouched
    assert pcw._class_spin.value == 4
    assert pcw._color_label.tooltip == _CLASS_COLORS_HEX[4]
    assert np.allclose(layer.face_color[1], _CLASS_COLORS_RGBA[4])
    assert layer.current_properties['class'][0] == 4


def test_shift_class_does_not_emit_spin_changed(make_widget):
    """Programmatic mirroring is silent; manual edits are not."""
    _, pcw, _ = make_widget(classes=[0, 3, 5])
    pcw._enable_cb.value = True
    pcw._navigation_cb.value = True

    emitted = []
    pcw._class_spin.changed.connect(emitted.append)

    pcw._shift_class(1)
    pcw._move_point(1)
    assert emitted == []

    pcw._class_spin.value = 7
    assert emitted == [7]


def test_shift_class_is_noop_without_selection_or_navigation(make_widget):
    _, pcw, layer = make_widget(classes=[0, 3, 5])
    pcw._enable_cb.value = True

    layer.selected_data = {1}
    pcw._shift_class(1)  # navigation mode is off
    assert classes(layer) == [0, 3, 5]

    pcw._navigation_cb.value = True
    layer.selected_data = set()
    pcw._shift_class(1)  # nothing selected
    assert classes(layer) == [0, 3, 5]


def test_canvas_focus_restored_after_keyboard_actions(
    make_widget, monkeypatch
):
    _, pcw, _ = make_widget()
    pcw._enable_cb.value = True
    pcw._navigation_cb.value = True

    calls = []
    monkeypatch.setattr(
        type(pcw), '_restore_canvas_focus', lambda self: calls.append(1)
    )

    pcw._move_point(1)
    pcw._shift_class(1)

    assert len(calls) == 2


def test_navigation_binds_and_restores_keys(make_widget, viewer):
    _, pcw, _ = make_widget()
    pcw._enable_cb.value = True
    before = {k: viewer.keymap.get(coerce_keybinding(k)) for k in NAV_KEYS}

    pcw._navigation_cb.value = True
    during = {k: viewer.keymap.get(coerce_keybinding(k)) for k in NAV_KEYS}
    assert all(
        during[k] is not None and during[k] is not before[k] for k in NAV_KEYS
    )

    pcw._navigation_cb.value = False
    after = {k: viewer.keymap.get(coerce_keybinding(k)) for k in NAV_KEYS}
    assert after == before


def test_disabling_annotation_turns_off_navigation(make_widget, viewer):
    _, pcw, _ = make_widget()
    before = {k: viewer.keymap.get(k) for k in NAV_KEYS}
    pcw._enable_cb.value = True
    pcw._navigation_cb.value = True

    pcw._enable_cb.value = False

    assert pcw._navigation_cb.value is False
    assert not pcw._navigation_cb.enabled
    assert {k: viewer.keymap.get(k) for k in NAV_KEYS} == before
