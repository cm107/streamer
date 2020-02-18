from typing import List
import cv2
import numpy as np
from logger import logger
from common_utils.check_utils import check_type, check_file_exists, check_value
from common_utils.image_utils import concat_images

from ...streamer import Streamer, DualStreamer, StreamerObject
from ...recorder import Recorder

class VideoConcatenator:
    def __init__(
        self, src0: str, src1: str, output_path: str,
        stream_mode: str='mono', default_fps: int=25,
        src0_scale_factor: float=1.0, src1_scale_factor: float=1.0, output_scale_factor: float=1.0,
        orientation: int=1
    ):
        """
        orientation
        0: horizontal
        1: vertical
        """
        
        # Create Streamers
        self.streamer0 = self.create_streamer(src=src0, mode=stream_mode, scale_factor=src0_scale_factor, verbose=True)
        self.streamer1 = self.create_streamer(src=src1, mode=stream_mode, scale_factor=src1_scale_factor, verbose=True)

        fps0, fps1 = self.streamer0.get_fps(), self.streamer1.get_fps()
        self.fps = fps0 if fps0 == fps1 else default_fps

        # Recorder
        self.recorder = None

        # Other Parameters
        self.output_path = output_path
        self.navigation_step = 10
        self.src0_scale_factor, self.src1_scale_factor = src0_scale_factor, src1_scale_factor
        self.output_scale_factor = output_scale_factor
        self.orientation = orientation

    def create_streamer(self, src: str, mode: str='mono', scale_factor: float=1.0, verbose: bool=False) -> StreamerObject:
        if verbose: logger.info(f"Creating Streamer for src={src}")
        check_value(item=mode, valid_value_list=['mono', 'dual'])
        check_file_exists(src)
        if mode == 'mono':
            streamer = Streamer(src=src, scale_factor=scale_factor)
        elif mode == 'dual':
            streamer = DualStreamer(src=src, scale_factor=scale_factor, direction=0)
        else:
            raise Exception
        return streamer

    def write_and_display_result(self, result: np.ndarray) -> bool:
        break_flag = False
        result_h, result_w = result.shape[:2]

        if self.recorder is None:
            self.recorder = Recorder(output_path=self.output_path, output_dims=(result_w, result_h), fps=self.fps)
        
        self.recorder.write(result)
        window_h, window_w = int(result_h * self.output_scale_factor), int(result_w * self.output_scale_factor)
        window_name = 'test'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, (window_w, window_h))
        cv2.imshow(window_name, result)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break_flag = True
        elif key == ord('a'):
            self.streamer0.rewind(self.navigation_step)
            self.streamer1.rewind(self.navigation_step)
        elif key == ord('d'):
            self.streamer0.fastforward(self.navigation_step)
            self.streamer1.fastforward(self.navigation_step)
        elif key == ord('w'):
            self.navigation_step = self.navigation_step * 10 if self.navigation_step < 100 else 100
        elif key == ord('s'):
            self.navigation_step = self.navigation_step / 10 if self.navigation_step > 1 else 100

        return break_flag

    def run(self):
        # Main Loop
        logger.info("Main Loop Start")
        while self.streamer0.is_playing() and self.streamer1.is_playing():
            logger.info(f"Frame {self.streamer0.get_progress_ratio_string()} | {self.streamer1.get_progress_ratio_string()}")
            if not self.streamer0.is_open() or not self.streamer1.is_open():
                logger.warning(f"Streamer is not open. Terminating...")
                break
            frame0, frame1 = self.streamer0.get_frame(), self.streamer1.get_frame()
            if frame0 is None or frame1 is None:
                logger.warning(f"Frame is None. Terminating...")
                break

            if self.orientation == 0 and frame0.shape[0] != frame1.shape[0]:
                logger.error(f"frame0.shape[0] == {frame0.shape[0]} != {frame1.shape[0]} == frame1.shape[0]")
                raise Exception
            elif self.orientation == 1 and frame0.shape[1] != frame1.shape[1]:
                logger.error(f"frame0.shape[1] == {frame0.shape[1]} != {frame1.shape[1]} == frame1.shape[1]")
                raise Exception

            result = concat_images(imga=frame0, imgb=frame1, orientation=self.orientation)

            break_flag = self.write_and_display_result(result=result)
            if break_flag:
                logger.warning(f"Break flag triggered. Terminating...")
                break

        self.recorder.close()
        self.streamer0.close()
        self.streamer1.close()