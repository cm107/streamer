import cv2
import numpy as np

class Recorder:
    def __init__(self, output_path: str, output_dims: list, fps: int):
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        self.output_dims = output_dims
        self.fps = fps
        self.record_out = cv2.VideoWriter(output_path, fourcc, fps, self.output_dims)

    def write(self, frame: np.ndarray):
        self.record_out.write(frame)

    def close(self):
        self.record_out.release()