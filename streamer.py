import cv2
import numpy as np
from abc import ABCMeta, abstractmethod

class StreamerObject(metaclass=ABCMeta):
    def __init__(self, src):
        self.worker = cv2.VideoCapture(src)
        self.full_width = None
        self.width = None
        self.height = None
    
    @abstractmethod
    def init_dims(self):
        pass

    @abstractmethod
    def get_frames(self) -> np.ndarray:
        pass

    def get_frame_count(self):
        return int(self.worker.get(7))

    def is_open(self):
        return self.worker.isOpened()

    def get_num_frames_read(self):
        return int(self.worker.get(1))

    def get_fps(self):
        return int(self.worker.get(5))

    def goto_frame(self, target_frame_num: int):
        success = self.worker.set(1, target_frame_num)
        return success

    def rewind(self, number_of_frames: int):
        total_frames = self.get_frame_count()
        current_frame_num = self.get_num_frames_read()
        target_frame = current_frame_num - number_of_frames
        target_frame = 0 if target_frame < 0 else target_frame
        success = self.goto_frame(target_frame)
        if not success:
            print(f'Failed to rewind to {target_frame}/{total_frames}')

    def fastforward(self, number_of_frames: int):
        total_frames = self.get_frame_count()
        current_frame_num = self.get_num_frames_read()
        target_frame = current_frame_num + number_of_frames
        target_frame = total_frames if target_frame > total_frames else target_frame
        success = self.goto_frame(target_frame)
        if not success:
            print(f'Failed to fastforward to {target_frame}/{total_frames}')

    def close(self):
        self.worker.release()
        cv2.destroyAllWindows()

class Streamer(StreamerObject):
    def __init__(self, src):
        super().__init__(src)
        self.init_dims()

    def init_dims(self):
        self.full_width = int(self.worker.get(3))
        self.width = self.full_width
        self.height = int(self.worker.get(4))

    def get_frames(self) -> np.ndarray:
        rec, frame = self.worker.read()
        if rec:
            return frame
        else:
            return None

class DualStreamer(StreamerObject):
    def __init__(self, src):
        super().__init__(src)

    def init_dims(self):
        self.full_width = self.worker.get(3)
        self.width = int(self.full_width / 2)
        self.height = self.worker.get(4)

    def get_frames(self) -> np.ndarray:
        rec, frame = self.worker.read()
        if rec:
            left_frame = frame[:,:self.width,:]
            right_frame = frame[:,self.width:,:]
            return left_frame, right_frame
        else:
            return None, None
