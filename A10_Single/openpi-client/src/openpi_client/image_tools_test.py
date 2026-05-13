import numpy as np

import openpi_client.image_tools as image_tools


def test_resize_with_pad_shapes():
    # Test case 1: Resize image with larger dimensions
    images = np.zeros((2, 10, 10, 3), dtype=np.uint8)  # Input images of shape (batch_size, height, width, channels)
    height = 20
    width = 20
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (2, height, width, 3)
    assert np.all(resized_images == 0)

    # Test case 2: Resize image with smaller dimensions
    images = np.zeros((3, 30, 30, 3), dtype=np.uint8)
    height = 15
    width = 15
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (3, height, width, 3)
    assert np.all(resized_images == 0)

    # Test case 3: Resize image with the same dimensions
    images = np.zeros((1, 50, 50, 3), dtype=np.uint8)
    height = 50
    width = 50
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert np.all(resized_images == 0)

    # Test case 3: Resize image with odd-numbered padding
    images = np.zeros((1, 256, 320, 3), dtype=np.uint8)
    height = 60
    width = 80
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert np.all(resized_images == 0)
