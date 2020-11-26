import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, cast

from abc import ABCMeta, abstractmethod
from ..util import scale_img
from logger import logger
from common_utils.check_utils import check_value
from common_utils.utils import get_class_string
from common_utils.file_utils import file_exists
from common_utils.image_utils import collage_from_img_buffer
from common_utils.cv_drawing_utils import draw_text_rows_in_corner

from ..recorder import Recorder

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
        if type(self.src) is str:
            return int(self.worker.get(7))
        else:
            return -1

    def is_open(self) -> bool:
        return self.worker.isOpened()

    def get_num_frames_read(self) -> int:
        if type(self.src) is str:
            return int(self.worker.get(1))
        else:
            return -1

    def is_playing(self) -> bool:
        if type(self.src) is str:
            return self.get_num_frames_read() < self.get_frame_count()
        else:
            return self.is_open()

    def get_progress_ratio_string(self) -> str:
        return f"{self.get_num_frames_read()}/{self.get_frame_count()}"

    def get_fps(self) -> int:
        return int(self.worker.get(5))

    def goto_frame(self, target_frame_num: int) -> bool:
        if type(self.src) is str:
            success = self.worker.set(1, target_frame_num)
            return success
        else:
            return False

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
        if type(self.src) is str:
            frame = self.get_frame()
            self.rewind(1)
            return frame.shape
        elif type(self.src) is int:
            frame = self.get_frame()
            return frame.shape
        else:
            raise Exception

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

class StreamerList:
    def __init__(self, src_list: List[str], scale_factor_list: List[float]=None):
        for src in src_list:
            if isinstance(src, int):
                pass
            elif isinstance(src, str):
                assert file_exists(src)
        if scale_factor_list is not None:
            assert len(scale_factor_list) == len(src_list)
        else:
            scale_factor_list = [1.0]*len(src_list)
        self.src_list = src_list
        self.streamer_list = [Streamer(src=src, scale_factor=scale_factor) for src, scale_factor in zip(src_list, scale_factor_list)]
    
    def get_frame_count(self) -> int:
        frame_counts = [streamer.get_frame_count() for streamer in self.streamer_list]
        return min(frame_counts)

    def get_num_frames_read(self) -> int:
        num_frames_read_list = [streamer.get_num_frames_read() for streamer in self.streamer_list]
        assert all([num_frames_read_list[0] == val for val in num_frames_read_list[1:]])
        return num_frames_read_list[0]

    def is_open(self) -> bool:
        return all([streamer.is_open() for streamer in self.streamer_list])

    def is_playing(self) -> bool:
        if all([type(src) is str for src in self.src_list]):
            return self.get_num_frames_read() < self.get_frame_count()
        else:
            return self.is_open()

    def get_frame(self) -> List[np.ndarray]:
        return [streamer.get_frame() for streamer in self.streamer_list]

    def get_fps(self) -> int:
        return min([streamer.get_fps() for streamer in self.streamer_list])

    def close(self):
        for streamer in self.streamer_list:
            streamer.close()

    def concatenate_streams(
        self, save_path: str='concat.avi', show_pbar: bool=True, collage_shape: Tuple[int]=None,
        labels: List[str]=None, label_corner: str='topleft', row_height_prop: float=0.05,
        label_color: tuple=(255,0,255), fps: int=None
    ):
        if labels is not None:
            assert isinstance(labels, list) and len(labels) == len(self.src_list)
            labels0 = []
            for label in labels:
                if isinstance(label, str):
                    labels0.append([label])
                elif isinstance(label, list):
                    for label_part in label:
                        assert isinstance(label_part, str)
                    labels0.append(label)
        else:
            labels0 = None

        recorder = cast(Recorder, None)
        pbar = tqdm(total=self.get_frame_count(), unit='frame(s)') if show_pbar else None
        while self.is_playing():
            frame_buffer = self.get_frame()
            if labels0 is not None:
                for i in range(len(frame_buffer)):
                    frame_buffer[i] = draw_text_rows_in_corner(
                        img=frame_buffer[i],
                        row_text_list=labels0[i],
                        row_height=row_height_prop*frame_buffer[i].shape[0],
                        corner=label_corner,
                        color=label_color
                    )
            frame = collage_from_img_buffer(
                img_buffer=frame_buffer,
                collage_shape=collage_shape if collage_shape is not None else (1, len(self.src_list))
            )
            if recorder is None:
                frame_h, frame_w = frame.shape[:2]
                recorder = Recorder(
                    output_path=save_path, output_dims=(frame_w, frame_h),
                    fps=fps if fps is not None else self.get_fps()
                )
            recorder.write(frame)
            if pbar is not None:
                pbar.update()
        recorder.close()
        if pbar is not None:
            pbar.close()

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
