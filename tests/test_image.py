import numpy as np
from napari.layers import Image

from napari_point_proofread._image_widget import (
    cvtRGB,
    find_RGB_channel_axis,
)


def test_widget_cvtRGB():
    data = np.random.rand(5, 3, 5, 2)
    img_layer = Image(data)

    new_data, metadata = cvtRGB(
        img_layer=img_layer,
        channel_axis=0,
    )

    assert new_data.shape == (3, 5, 2, 5)
    assert metadata['name'] == f'{img_layer.name}_RGB'

    # Test with channel_axis=1
    new_data, metadata = cvtRGB(
        img_layer=img_layer,
        channel_axis=1,
    )
    assert new_data.shape == (5, 5, 2, 3)

    # # Test with channel_axis=None, should find the channel axis automatically
    new_data, metadata = cvtRGB(
        img_layer=img_layer,
        channel_axis=None,
    )
    assert new_data.shape == (5, 5, 2, 3)


def test_find_RGB_channel_axis():
    # Test with an array that has a channel axis of size 3
    data = np.random.rand(5, 3, 5, 2)
    channel_axis = find_RGB_channel_axis(data)
    assert channel_axis == 1

    # Test with an array that has no axis of size 3
    data = np.random.rand(5, 4, 5, 2)
    with np.testing.assert_raises(ValueError):
        find_RGB_channel_axis(data)

    # Test with an array that has 2 axes of size 3
    data = np.random.rand(5, 3, 5, 3)
    channel_axis = find_RGB_channel_axis(data)
    assert channel_axis == 1  # It should return the first axis
