from streamer.tools.video_concat import VideoConcatenator

worker = VideoConcatenator(
    src0='/home/clayton/Downloads/result_0.avi',
    src1='/home/clayton/Downloads/result_1.avi',
    output_path='/home/clayton/Downloads/combined.avi',
    stream_mode='mono',
    default_fps=25,
    src0_scale_factor=1.0,
    src1_scale_factor=1.0,
    output_scale_factor=0.5,
    orientation=1 # vertical
)
worker.run()