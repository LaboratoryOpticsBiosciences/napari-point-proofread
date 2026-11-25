import contextlib
from functools import partial
from itertools import pairwise
from string import ascii_uppercase

import napari
import numpy as np
from magicgui import magic_factory
from magicgui.widgets import (
    CheckBox,
    Container,
    Label,
    PushButton,
    SpinBox,
)
from napari import layers as L
from napari import types

from ._widget_utils import _make_titled_panel, setup_layout


@magic_factory(
    point_size={
        'label': 'Point size',
        'tooltip': 'Change size of all points at once',
        'widget_type': 'Slider',
        'value': 10,
        'max': 15,
    },
    auto_call=True,
    call_button=False,
)
def widget_points(
    point_layer: L.Points,
    point_size,
):
    """Set the displayed size for the selected points layer.

    Parameters
    ----------
    point_layer : napari.layers.Points or None
        Target points layer whose marker size will be changed.
    point_size : float
        New marker size to use for the layer.

    Returns
    -------
    None
        The points layer is updated in place.
    """
    if point_layer is not None:
        point_layer.size = point_size


@magic_factory(
    img_layer={'label': 'Image shape for grid'},
    bins={
        'label': 'bins (x, y)',
        'widget_type': 'TupleEdit',
        'value': (5, 5),
        'options': {'min': 2, 'max': 10, 'step': 1},
        'tooltip': 'Number of bins in x and y direction',
    },
    call_button='make grid',
)
def widget_grid(
    img_layer: 'napari.layers.Image',
    bins: tuple[int, int],
) -> types.LayerDataTuple:
    """Create a line grid overlay for the selected image layer.

    Parameters
    ----------
    img_layer : napari.layers.Image or None
        Image layer used to determine the grid extent.
    bins : tuple of int
        Number of bins to draw in the x and y directions.

    Returns
    -------
    LayerDataTuple or None
        Shapes layer data for the grid overlay, or ``None`` if no image exists.
    """
    if img_layer is None:
        return None

    xbins, ybins = bins
    shape = img_layer.data.shape
    is_rgb = shape[-1] == 3
    height, width = shape[-3:-1] if is_rgb else shape[-2:]

    bins_x = np.linspace(0, width, xbins + 1)
    bins_y = np.linspace(0, height, ybins + 1)

    # Lines are stored as [[y0, x0], [y1, x1]].
    lines = [np.array([[0, x], [height, x]]) for x in bins_x]  # vertical
    lines += [np.array([[y, 0], [y, width]]) for y in bins_y]  # horizontal

    # Dummy lines that carry the axis labels. `features` + `text` cannot give
    # each shape its own translation, so labels sit on these offset lines.
    yoffset = -height * 0.05
    xoffset = -width * 0.05
    lines += [
        np.array([[yoffset, x0], [yoffset, x1]]) for x0, x1 in pairwise(bins_x)
    ]
    lines += [
        np.array([[y0, xoffset], [y1, xoffset]]) for y0, y1 in pairwise(bins_y)
    ]

    n_grid_lines = xbins + ybins + 2
    labels = [
        *([''] * n_grid_lines),  # no label on the grid lines themselves
        *ascii_uppercase[:xbins],  # A, B, C, ...
        *map(str, range(ybins)),  # 0, 1, 2, ...
    ]
    kwargs = {
        'name': f'grid({img_layer.name})',
        'shape_type': 'line',
        'features': {'number': labels},
        'text': {'string': '{number}', 'size': 16, 'color': 'lime'},
    }
    return (lines, kwargs, 'shapes')


# ---------------------------------------------------------------------------
# Class annotation: constants
# ---------------------------------------------------------------------------

_CLASS_COLORS_HEX = (
    '#E6194B',  # 0 – red
    '#3CB44B',  # 1 – green
    '#4363D8',  # 2 – blue
    '#F58231',  # 3 – orange
    '#911EB4',  # 4 – purple
    '#42D4F4',  # 5 – cyan
    '#F032E6',  # 6 – magenta
    '#BFEF45',  # 7 – lime
    '#FABED4',  # 8 – pink
    '#469990',  # 9 – teal
)
_N_CLASSES = len(_CLASS_COLORS_HEX)


def _hex_to_rgba(hex_color: str) -> list[float]:
    """Convert a ``'#RRGGBB'`` string to an RGBA list in the range [0, 1]."""
    h = hex_color.lstrip('#')
    return [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)] + [1.0]


_CLASS_COLORS_RGBA = np.array([_hex_to_rgba(c) for c in _CLASS_COLORS_HEX])
_UNASSIGNED_RGBA = np.array([0.7, 0.7, 0.7, 0.7])  # grey, no valid class


# ---------------------------------------------------------------------------
# PointClassWidget
# ---------------------------------------------------------------------------


class PointClassWidget(Container):
    """Optional widget for per-point numerical class annotation.

    When enabled:
      * Every point added to the active layer is automatically tagged with
        the currently selected class.
      * Selected points can be re-tagged via 'Assign class to selected'.
      * Points are colored per-class and display their class number.
      * The 'class' column is included when the layer is saved as CSV.

    In class navigation mode:
      * Left/Right select the previous/next point.
      * Up/Down change the class of the selected point *directly in the
        layer features*; the spin box is only updated to mirror the result
        (with its signals blocked), so no callback recursion occurs.
      * Manual spin-box edits still flow through ``_on_class_changed``.
      * Keyboard focus is handed back to the canvas after each key action.
    """

    def __init__(self, point_layer_widget):
        """Create the class-annotation controls bound to a layer selector.

        Parameters
        ----------
        point_layer_widget : magicgui.widgets.Widget
            Widget that selects the active points layer.
        """
        self._point_layer_widget = point_layer_widget
        self._layer = None
        self._event_conn = None
        self._refreshing = False
        self._navigation_viewer = None
        self._previous_keybindings = {}

        self._enable_cb = CheckBox(value=False, text='Enable class annotation')
        self._enable_cb.tooltip = (
            'Enable or disable class annotation for the active points layer.'
        )

        self._class_spin = SpinBox(value=0, min=0, max=_N_CLASSES - 1)
        self._class_spin.enabled = False
        self._class_spin.tooltip = 'Select the class for new points.'

        self._color_label = Label(value='')
        self._color_label.native.setFixedSize(18, 18)
        self._update_color_label(0)

        self._class_row = Container(
            layout='horizontal',
            widgets=[
                Label(value='Current class'),
                self._class_spin,
                Label(value='Class color'),
                self._color_label,
            ],
            labels=False,
        )
        setup_layout(self._class_row)

        self._assign_btn = PushButton(text='Assign class to selected')
        self._assign_btn.enabled = False
        self._assign_btn.tooltip = (
            'Assign the currently selected class to the selected points.'
        )

        self._navigation_cb = CheckBox(
            value=False, text='class navigation mode'
        )
        self._navigation_cb.enabled = False
        self._navigation_cb.tooltip = (
            'Navigate between points using Left/Right arrows. '
            'Change the selected point class using Up/Down arrows.'
        )
        self._navigation_label = Label(
            value='Empty layer or no point selected'
        )

        # put the navigation cb and label in a horizontal row to save vertical space
        self._navigation_row = Container(
            layout='horizontal',
            widgets=[self._navigation_cb, self._navigation_label],
            labels=False,
        )

        self._enable_cb.changed.connect(self._on_toggle)
        self._class_spin.changed.connect(self._on_class_changed)
        self._assign_btn.changed.connect(self._on_assign)
        self._navigation_cb.changed.connect(self._on_navigation_toggle)
        point_layer_widget.changed.connect(self._on_layer_changed)

        super().__init__(
            layout='vertical',
            widgets=[
                self._enable_cb,
                self._class_row,
                self._assign_btn,
                self._navigation_row,
            ],
            labels=False,
        )

    # ---- slots -----------------------------------------------------------

    def _on_toggle(self, enabled):
        """Enable or disable the class annotation controls."""
        for widget in (
            self._class_spin,
            self._assign_btn,
            self._navigation_cb,
        ):
            widget.enabled = enabled

        layer = self._current_layer()
        if enabled:
            self._activate(layer)
        else:
            self._navigation_cb.value = False
            self._deactivate(layer)

        if layer is not None and len(layer.data) > 0:
            self._navigation_cb.enabled = enabled
        else:
            self._navigation_cb.value = False
            self._navigation_cb.enabled = False
            self._navigation_label.value = 'Empty layer or no point selected'

    def _on_layer_changed(self, layer):
        """Switch the widget to a new active points layer."""
        if self._enable_cb.value:
            self._deactivate(self._layer)
            self._activate(layer)

        self._layer = layer

        if self._navigation_cb.value:
            if layer is not None and len(layer.data) > 0:
                self._select_point(0)
                self._navigation_label.value = (
                    '←/→: select point, ↑/↓: change class'
                )
            else:
                self._navigation_cb.value = False
                self._navigation_label.value = (
                    'Empty layer or no point selected'
                )

    def _on_class_changed(self, cls):
        """Handle *manual* spin-box edits.

        New points inherit ``cls`` and any selected points are re-tagged.
        Programmatic updates never reach this slot (see
        ``_set_class_spin_value``).
        """
        self._update_color_label(cls)

        layer = self._current_layer()
        if layer is None or not self._enable_cb.value:
            return

        layer.current_properties = {'class': [cls]}
        if not self._assign_selected_points_class(layer, cls):
            self._refresh_display(layer)

    def _on_data_changed(self, event):
        """Synchronize class metadata after points are added or removed."""
        layer = self._current_layer()
        if layer is None or self._refreshing:
            return
        self._sync_class_column(layer)
        self._refresh_display(layer)

    def _on_assign(self, _):
        """Assign the spin-box class to the currently selected points."""
        layer = self._current_layer()
        if layer is not None:
            self._assign_selected_points_class(layer, self._class_spin.value)

    def _on_navigation_toggle(self, enabled):
        """Enable or disable keyboard-based class navigation."""
        if not enabled:
            self._unbind_navigation_keys()
            self._navigation_label.value = ''
            return

        layer = self._current_layer()
        if layer is None or len(layer.data) == 0:
            self._navigation_cb.value = False
            return

        self._bind_navigation_keys()
        selected = sorted(layer.selected_data)
        self._select_point(selected[0] if selected else 0)

    # ---- keyboard navigation ---------------------------------------------

    def _bind_navigation_keys(self):
        viewer = napari.current_viewer()
        if viewer is None:
            return

        self._navigation_viewer = viewer

        callbacks = {
            'Left': partial(self._move_point, -1),
            'Right': partial(self._move_point, 1),
            'Up': partial(self._shift_class, 1),
            'Down': partial(self._shift_class, -1),
        }

        for key, callback in callbacks.items():
            self._previous_keybindings[key] = viewer.bind_key(
                key, callback, overwrite=True
            )

    def _unbind_navigation_keys(self):
        viewer = self._navigation_viewer
        if viewer is None:
            return

        for key, previous in self._previous_keybindings.items():
            viewer.bind_key(key, previous, overwrite=True)

        self._previous_keybindings.clear()
        self._navigation_viewer = None

    def _move_point(self, step, _viewer=None):
        """Move the selection to an adjacent point (Left/Right)."""
        layer = self._current_layer()
        if (
            not self._navigation_cb.value
            or layer is None
            or not len(layer.data)
        ):
            return

        selected = sorted(layer.selected_data)
        current = selected[0] if selected else 0

        # Stay within the layer boundaries.
        new_index = int(np.clip(current + step, 0, len(layer.data) - 1))

        self._select_point(new_index)
        self._center_camera_on_point(layer, new_index)
        self._restore_canvas_focus()

    def _center_camera_on_point(self, layer, index):
        point = layer.data[index]
        viewer = self._navigation_viewer

        # Move non-displayed dimensions to the selected point.
        for dim in viewer.dims.not_displayed:
            viewer.dims.set_point(dim, point[dim])

        camera_center = tuple(point[dim] for dim in viewer.dims.displayed)

        # Center the camera on the selected point.
        viewer.camera.center = camera_center

    def _shift_class(self, delta, _viewer=None):
        """Change the class of the selected point(s) directly (Up/Down).

        The layer features are the source of truth: they are updated first,
        and the spin box merely mirrors the result with its signals blocked.
        """
        layer = self._current_layer()
        if not self._navigation_cb.value or layer is None:
            return

        selected = sorted(layer.selected_data)
        if not selected:
            return

        self._ensure_class_column(layer)
        current = int(layer.features['class'].iloc[selected[0]])
        new_cls = int(np.clip(current + delta, 0, _N_CLASSES - 1))

        self._assign_selected_points_class(layer, new_cls)
        layer.current_properties = {'class': [new_cls]}  # new points inherit
        self._set_class_spin_value(new_cls)
        self._restore_canvas_focus()

    def _select_point(self, index):
        """Select a single point and mirror its class in the spin box."""
        layer = self._current_layer()
        if layer is None or not 0 <= index < len(layer.data):
            return

        layer.selected_data = {index}
        self._ensure_class_column(layer)
        self._set_class_spin_value(int(layer.features['class'].iloc[index]))
        self._navigation_label.value = rf'←/→: select point, ↑/↓: change class. {index + 1} / {len(layer.data)}'

    # ---- helpers ---------------------------------------------------------

    def _update_color_label(self, cls):
        """Update the class-color swatch to match ``cls``."""
        color = _CLASS_COLORS_HEX[cls]
        self._color_label.native.setStyleSheet(
            f'background-color: {color};'
            'border: 1px solid palette(mid);'
            'min-width: 18px; max-width: 18px;'
            'min-height: 18px; max-height: 18px;'
        )
        self._color_label.tooltip = color

    def _set_class_spin_value(self, cls):
        """Update the spin box and swatch without emitting ``changed``."""
        with self._class_spin.changed.blocked():
            self._class_spin.value = cls
        self._update_color_label(cls)

    def _restore_canvas_focus(self):
        """Return keyboard focus to the napari canvas."""
        viewer = self._navigation_viewer
        if viewer is None:
            return
        # Relies on napari's private Qt wrapper; guard against API changes.
        with contextlib.suppress(AttributeError):
            viewer.window._qt_viewer.canvas.native.setFocus()

    @contextlib.contextmanager
    def _refreshing_guard(self):
        """Mark display updates as in progress to block re-entrant events."""
        self._refreshing = True
        try:
            yield
        finally:
            self._refreshing = False

    def _assign_selected_points_class(self, layer, cls):
        """Assign ``cls`` to all selected points; return whether any changed."""
        if layer is None:
            return False

        selected = sorted(layer.selected_data)
        if not selected:
            return False

        self._ensure_class_column(layer)

        # Reassigning layer.features emits the features event in napari.
        # In-place pandas edits may not invalidate text labels immediately.
        features = layer.features.copy()
        features.loc[selected, 'class'] = cls
        layer.features = features

        self._refresh_display(layer)
        layer.refresh()
        return True

    def _current_layer(self):
        """Return the points layer selected in the companion widget."""
        return self._point_layer_widget.value

    def _activate(self, layer):
        """Attach event handling and initialize class data for ``layer``."""
        if layer is None:
            return
        self._layer = layer
        self._ensure_class_column(layer)
        # New points inherit the currently selected class.
        layer.current_properties = {'class': [self._class_spin.value]}
        self._disconnect_events()
        self._event_conn = layer.events.data.connect(self._on_data_changed)
        self._refresh_display(layer)

    def _deactivate(self, layer):
        """Detach event handling and hide annotation visuals for ``layer``."""
        self._disconnect_events()
        if layer is None:
            return
        layer.text.visible = False
        if len(layer.data) > 0:
            layer.face_color = 'white'

    def _disconnect_events(self):
        """Disconnect the stored data-change callback if attached."""
        if self._event_conn is not None:
            with contextlib.suppress(Exception):
                self._event_conn.disconnect()
            self._event_conn = None

    def _ensure_class_column(self, layer):
        """Ensure the points layer has a ``class`` feature column."""
        if 'class' not in layer.features.columns:
            layer.features['class'] = np.zeros(len(layer.data), dtype=np.int64)

    def _sync_class_column(self, layer):
        """Resize the ``class`` column to match the number of points.

        New points get the spin-box class; removed points are truncated.
        """
        self._ensure_class_column(layer)
        n = len(layer.data)
        old = layer.features['class'].to_numpy()
        if len(old) == n:
            return
        if n > len(old):
            fill = np.full(
                n - len(old), self._class_spin.value, dtype=np.int64
            )
            layer.features['class'] = np.concatenate([old, fill])
        else:
            layer.features['class'] = old[:n]

    def _refresh_display(self, layer):
        """Recompute face colors and text labels from stored class values."""
        if self._refreshing or len(layer.data) == 0:
            return

        self._ensure_class_column(layer)
        classes = layer.features['class'].to_numpy().astype(int)

        valid = (classes >= 0) & (classes < _N_CLASSES)
        colors = np.where(
            valid[:, None],
            _CLASS_COLORS_RGBA[np.where(valid, classes, 0)],
            _UNASSIGNED_RGBA,
        )

        with self._refreshing_guard():
            layer.face_color = colors
            layer.text = {
                'string': '{class}',
                'size': 10,
                'color': 'white',
                'anchor': 'center',
                'visible': True,
            }


class ProofreadWidget(Container):
    """Main widget for point-proofreading tasks.

    Provides controls for point size, grid overlay creation, and optional
    class annotation on napari points layers.
    """

    def __init__(self):
        self.grid_widget = widget_grid()
        self.points_widget = widget_points()
        self.point_class_widget = PointClassWidget(
            self.points_widget.point_layer
        )

        for widget in (
            self.grid_widget,
            self.points_widget,
            self.point_class_widget,
        ):
            setup_layout(widget)

        super().__init__(
            layout='vertical',
            widgets=[
                _make_titled_panel('Grid Overlay Tools', [self.grid_widget]),
                _make_titled_panel(
                    'Point Layer Tools',
                    [self.points_widget, self.point_class_widget],
                ),
            ],
            labels=False,
        )
