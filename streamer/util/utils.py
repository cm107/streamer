import cv2
import numpy as np

def get_scaled_dims(width: int, height: int, scale_factor) -> (int, int):
    return int(width * scale_factor), int(height * scale_factor)

def scale_img(frame: np.ndarray, scale_factor) -> np.ndarray:
    height, width = frame.shape[:2]
    downsized_width, downsized_height = get_scaled_dims(width, height, scale_factor)

    if scale_factor == 1.0:    
        return frame
    elif scale_factor > 0.0 and scale_factor < 1.0:
        return cv2.resize(
            frame,
            (downsized_width, downsized_height),
            interpolation=cv2.INTER_AREA)
    elif scale_factor > 1.0:
        return cv2.resize(
            frame,
            (downsized_width, downsized_height),
            interpolation=cv2.INTER_LINEAR)
    else:
        raise Exception(f"Invalid scale_factor={scale_factor}. Expected scale_factor > 0.")

def get_grayscale(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)