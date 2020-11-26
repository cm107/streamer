import cv2
from streamer.streamer import Streamer
from streamer.recorder import Recorder
from streamer.cv_viewer import SimpleVideoViewer

# Open Workers
streamer = Streamer(src='/home/clayton/Downloads/result_0.avi')
recorder = None
viewer = SimpleVideoViewer(preview_width=1000, window_name='Test')

output_path = '/home/clayton/Downloads/grayscale_result_0.avi'

while streamer.is_playing():
    # Read Frame
    print(f'Frame {streamer.get_progress_ratio_string()}')
    frame = streamer.get_frame()
    if frame is None:
        print('Failed to read frame.')
        break

    # Image Processing
    result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    # Write & Show Result
    if recorder is None:
        result_h, result_w = result.shape[:2]
        recorder = Recorder(output_path=output_path, output_dims=(result_w, result_h), fps=streamer.get_fps())
    recorder.write(result)
    viewer.show(result)

# Close Workers
streamer.close()
if recorder is not None:
    recorder.close()
viewer.close()