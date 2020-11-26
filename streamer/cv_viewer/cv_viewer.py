import cv2
import numpy as np
from ..streamer.streamer import StreamerObject

def cv_simple_image_viewer(img: np.ndarray, preview_width: int, window_name: str="Simple Image Viewer") -> bool:
    quit_flag = False

    # Window Declaration
    img_h, img_w = img.shape[:2]
    scale_factor = preview_width / img_w
    window_w, window_h = int(scale_factor * img_w), int(scale_factor * img_h)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, (window_w, window_h))
    cv2.imshow(window_name, img)
    quit_flag = False
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            break
        elif key == ord('q'):
            quit_flag = True
            break
    cv2.destroyAllWindows()
    
    return quit_flag

def cv_simple_video_viewer(img: np.ndarray, preview_width: int, window_name: str="Simple Video Viewer") -> bool:
    quit_flag = False

    # Window Declaration
    img_h, img_w = img.shape[:2]
    scale_factor = preview_width / img_w
    window_w, window_h = int(scale_factor * img_w), int(scale_factor * img_h)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, (window_w, window_h))
    cv2.imshow(window_name, img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        quit_flag = True
    elif key == ord('p'):
        while True:
            key0 = cv2.waitKey(1) & 0xFF
            if key0 == ord('q'):
                quit_flag = True
                break
            elif key0 == ord('p'):
                break
    return quit_flag

def cv_dynamic_video_viewer(
    img: np.ndarray, preview_width: int, streamer: StreamerObject,
    current_step: int=10, step_factor: int=10, min_step: int=1, max_step: int=10000,
    window_name: str="Simple Image Viewer"
) -> bool:
    quit_flag = False
    next_step = current_step

    # Window Declaration
    img_h, img_w = img.shape[:2]
    scale_factor = preview_width / img_w
    window_w, window_h = int(scale_factor * img_w), int(scale_factor * img_h)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, (window_w, window_h))
    cv2.imshow(window_name, img)
    key = cv2.waitKey(1) & 0xFF
    print(f'key: {key}')
    if key == ord('q'):
        quit_flag = True
    elif key == ord('w'): # Increase Step
        next_step = int(current_step * step_factor)
        next_step = next_step if next_step <= max_step else max_step
    elif key == ord('s'): # Decrease Step
        next_step = int(current_step / step_factor)
        next_step = next_step if next_step >= min_step else min_step
    elif key == ord('a'): # Rewind
        streamer.rewind(current_step)
    elif key == ord('d'): #Fastforward
        streamer.fastforward(current_step)
    elif key == ord('p'):
        while True:
            key0 = cv2.waitKey(1) & 0xFF
            if key0 == ord('q'):
                quit_flag = True
                break
            elif key == ord('w'): # Increase Step
                next_step = int(current_step * step_factor)
                next_step = next_step if next_step <= max_step else max_step
            elif key == ord('s'): # Decrease Step
                next_step = int(current_step / step_factor)
                next_step = next_step if next_step >= min_step else min_step
            elif key == ord('a'): # Rewind
                streamer.rewind(current_step)
            elif key == ord('d'): #Fastforward
                streamer.fastforward(current_step)
            elif key0 == ord('p'):
                break
    return quit_flag, next_step

class SimpleVideoViewer:
    def __init__(self, preview_width: int, window_name: str="Simple Video Viewer"):
        self.preview_width = preview_width
        self.window_name = window_name
        self.resize_done = False

    def show(self, img: np.ndarray) -> bool:
        quit_flag = False

        # Window Declaration
        if not self.resize_done:
            img_h, img_w = img.shape[:2]
            scale_factor = self.preview_width / img_w
            window_w, window_h = int(scale_factor * img_w), int(scale_factor * img_h)
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, (window_w, window_h))
            self.resize_done = True
        cv2.imshow(self.window_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            quit_flag = True
        elif key == ord('p'):
            while True:
                key0 = cv2.waitKey(1) & 0xFF
                if key0 == ord('q'):
                    quit_flag = True
                    break
                elif key0 == ord('p'):
                    break
        return quit_flag

    def close(self):
        cv2.destroyAllWindows()