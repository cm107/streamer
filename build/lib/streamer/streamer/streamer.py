import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
from .util import scale_img
from logger import logger
from common_utils.check_utils import check_value
from common_utils.utils import get_class_string

class StreamerObject(metaclass=ABCMeta):
    def __init__(self, src, scale_factor: float=1.0):
        self.src = src
        self.worker = cv2.VideoCapture(src)
        self.full_width = None
        self.full_height = None
        self.width = None
        self.height = None
        self.scale_factor = scale_factor
        self.downsized_width = None
        self.downsized_height = None
    
    @abstractmethod
    def init_dims(self):
        pass

    @abstractmethod
    def get_frames(self):
        pass

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        pass

    @abstractmethod
    def _update_frame_buff(self, frame: np.ndarray) -> np.ndarray:
        pass

    def get_frame_count(self) -> int:
        return int(self.worker.get(7))

    def is_open(self) -> bool:
        return self.worker.isOpened()

    def get_num_frames_read(self) -> int:
        return int(self.worker.get(1))

    def is_playing(self) -> bool:
        return self.get_num_frames_read() < self.get_frame_count()

    def get_progress_ratio_string(self) -> str:
        return f"{self.get_num_frames_read()}/{self.get_frame_count()}"

    def get_fps(self) -> int:
        return int(self.worker.get(5))

    def goto_frame(self, target_frame_num: int) -> bool:
        success = self.worker.set(1, target_frame_num)
        return success

    def is_mono(self) -> bool:
        return isinstance(self, Streamer)

    def is_dual(self) -> bool:
        return isinstance(self, DualStreamer)

    def rewind(self, number_of_frames: int):
        total_frames = self.get_frame_count()
        current_frame_num = self.get_num_frames_read()
        target_frame = current_frame_num - number_of_frames
        target_frame = 0 if target_frame < 0 else target_frame
        success = self.goto_frame(target_frame)
        if not success:
            logger.warning(f'Failed to rewind to {target_frame}/{total_frames}')

    def fastforward(self, number_of_frames: int):
        total_frames = self.get_frame_count()
        current_frame_num = self.get_num_frames_read()
        target_frame = current_frame_num + number_of_frames
        target_frame = total_frames if target_frame > total_frames else target_frame
        success = self.goto_frame(target_frame)
        if not success:
            logger.warning(f'Failed to fastforward to {target_frame}/{total_frames}')

    def sample_frame_shape(self) -> list:
        frame = self.get_frame()
        self.rewind(1)
        return frame.shape

    def assert_open(self):
        if not self.is_open():
            logger.error(
                f"{get_class_string(self)} is not open.\n" + \
                "Please check your video source.\n" + \
                f"Attempted to use {self.src}")
            raise Exception

    def close(self):
        self.worker.release()
        cv2.destroyAllWindows()

class Streamer(StreamerObject):
    def __init__(self, src, scale_factor: float=1.0):
        super().__init__(src, scale_factor)
        self.assert_open()
        self.init_dims()
        self.current_frame = None

    def init_dims(self):
        self.full_width = int(self.worker.get(3))
        self.full_height = int(self.worker.get(4))
        self.width = self.full_width
        self.height = self.full_height
        self.downsized_width = int(self.width * self.scale_factor)
        self.downsized_height = int(self.height * self.scale_factor)

    def get_frames(self) -> np.ndarray:
        rec, frame = self.worker.read()
        if rec:
            self._update_frame_buff(frame)
            return scale_img(frame, self.scale_factor)
        else:
            return None

    def get_frame(self) -> np.ndarray:
        return self.get_frames()

    def _update_frame_buff(self, frame: np.ndarray):
        self.current_frame = frame if frame is not None else self.current_frame

class DualStreamer(StreamerObject):
    def __init__(self, src, scale_factor: float=1.0, direction: int=0):
        """
        direction=0 : left-right direction
        direction=1 : top-bottom direction
        """

        super().__init__(src, scale_factor)
        self.assert_open()
        check_value(item=direction, valid_value_list=[0, 1])
        self.direction = direction
        self.init_dims()
        self.current_left_frame = None
        self.current_right_frame = None

    def init_dims(self):
        self.full_width = int(self.worker.get(3))
        self.full_height = int(self.worker.get(4))
        if self.direction == 0: # left-right direction
            self.width = int(self.full_width / 2)
            self.height = self.full_height
        elif self.direction == 1: # top-down direction
            self.width = self.full_width
            self.height = int(self.full_height / 2)
        else:
            logger.error(f"Invalid direction: {self.direction}.")
            logger.error("Expecting direction in [0, 1]")
            raise Exception
        self.downsized_width = int(self.width * self.scale_factor)
        self.downsized_height = int(self.height * self.scale_factor)

    def get_frames(self) -> (np.ndarray, np.ndarray):
        rec, frame = self.worker.read()
        if rec:
            if self.direction == 0:
                left_frame = frame[:,:self.width,:]
                right_frame = frame[:,self.width:,:]
            elif self.direction == 1:
                left_frame = frame[:self.height,:,:]
                right_frame = frame[self.height:,:,:]
            self._update_frame_buff(left_frame, right_frame)
            return scale_img(left_frame, self.scale_factor), scale_img(right_frame, self.scale_factor)
        else:
            return None, None

    def get_frame(self, side: str=None) -> np.ndarray:
        frame0, frame1 = self.get_frames()
        if self.direction == 0: # left-right direction
            use_side = None
            if side is not None:
                check_value(item=side, valid_value_list=['left', 'right'])
                use_side = side
            else:
                use_side = 'left'
            if use_side == 'left':
                return frame0
            elif use_side == 'right':
                return frame1
            else:
                raise Exception
        elif self.direction == 1: # top-down direction
            use_side = None
            if side is not None:
                check_value(item=side, valid_value_list=['top', 'down'])
                use_side = side
            else:
                use_side = 'top'
            if use_side == 'top':
                return frame0
            elif use_side == 'right':
                return frame1
            else:
                raise Exception
        else:
            raise Exception

    def _update_frame_buff(self, left_frame: np.ndarray, right_frame: np.ndarray):
        self.current_left_frame = left_frame if left_frame is not None else self.current_left_frame
        self.current_right_frame = right_frame if right_frame is not None else self.current_right_frame
