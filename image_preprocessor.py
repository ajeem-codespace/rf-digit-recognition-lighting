import cv2
import numpy as np

def preprocess_for_lighting(image_2d_uint8_0_255):
    if image_2d_uint8_0_255.ndim != 2:
        return None 

    # 1. Adaptive Thresholding
    block_size = 11
    C_value = 3
    binary_image = cv2.adaptiveThreshold(
        image_2d_uint8_0_255,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        C_value
    )

    # 2. Normalize to 0-1
    processed_image_normalized_2d = binary_image / 255.0

    # 3. Flatten for Random Forest
    flattened_image = processed_image_normalized_2d.flatten()
    
    return flattened_image

