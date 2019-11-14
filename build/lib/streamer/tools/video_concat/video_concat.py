import cv2
import numpy as np
from logger import logger
from common_utils.check_utils import check_file_exists, check_value
from common_utils.image_utils import concat_images

from ...streamer import Streamer, DualStreamer, StreamerObject
from ...recorder import Recorder

class VideoConcatenator:
    def __init__(self, src0: str, src1: str, output_path: str, stream_mode: str='mono', default_fps: int=25, scale_factor: float=1.0):
        # Create Streamers
        self.streamer0 = self.create_streamer(src=src0, mode=stream_mode, verbose=True)
        self.streamer1 = self.create_streamer(src=src1, mode=stream_mode, verbose=True)

        fps0, fps1 = self.streamer0.get_fps(), self.streamer1.get_fps()
        self.fps = fps0 if fps0 == fps1 else default_fps

        # Recorder
        self.recorder = None

        # Other Parameters
        self.output_path = output_path
        self.navigation_step = 10
        self.scale_factor = scale_factor

    def create_streamer(self, src: str, mode: str='mono', verbose: bool=False) -> StreamerObject:
        if verbose: logger.info(f"Creating Streamer for src={src}")
        check_value(item=mode, valid_value_list=['mono', 'dual'])
        check_file_exists(src)
        if mode == 'mono':
            streamer = Streamer(src=src, scale_factor=1.0)
        elif mode == 'dual':
            streamer = DualStreamer(src=src, scale_factor=1.0, direction=0)
        else:
            raise Exception
        return streamer

    def write_and_display_result(self, result: np.ndarray) -> bool:
        break_flag = False
        result_h, result_w = result.shape[:2]

        if self.recorder is None:
            self.recorder = Recorder(output_path=self.output_path, output_dims=(result_w, result_h), fps=self.fps)
        
        self.recorder.write(result)
        window_h, window_w = int(result_h * self.scale_factor), int(result_w * self.scale_factor)
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

            if frame0.shape != frame1.shape:
                logger.error(f"frame0.shape == {frame0.shape} != {frame1.shape} == frame1.shape")
                raise Exception

            result = concat_images(imga=frame0, imgb=frame1, orientation=1)

            break_flag = self.write_and_display_result(result=result)
            if break_flag:
                logger.warning(f"Break flag triggered. Terminating...")
                break

        self.recorder.close()
        self.streamer0.close()
        self.streamer1.close()