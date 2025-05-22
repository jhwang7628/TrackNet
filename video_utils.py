import cv2
import numpy as np
from tqdm import tqdm

import logging
import os
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)

def capture_video(video_path):
    """Loading the video using VideoCapture in opencv.

    Args:
        video_path: Path to the video

    Returns:
        The VideoCapture object.
    """
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video path cannot be captured: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video cannot be opened: {video_path}")
    return cap

def get_video_fps(cap):
    """Getting the video FPS.

    Args:
        cap: VideoCapture object from opencv.

    Returns:
        FPS of the video.
    """
    return cap.get(cv2.CAP_PROP_FPS)

def get_video_number_frames(cap):
    """Getting the number of frames in the video.

    Args:
        cap: VideoCapture object from opencv.

    Returns:
        Number of frames in the video.
    """
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def get_video_duration(cap):
    """Getting the duration of the video in seconds.

    Args:
        cap: VIdeo Capture object from opencv.

    Returns:
        Video duration in seconds.
    """
    return get_video_number_frames(cap) / get_video_fps(cap)

def extract_frames(cap, start_frame_index=0, num_frames=None, stride=1, color_format="BGR"):
    """Extract all the frames given a sampling parameter set.

    Args:
        cap                 : VideoCapture object from opencv.
        start_frame_index   : Starting frame index.
        num_frames          : Number of frames to be extracted.
        stride              : Stride of the frame sampling.
        color_format        : String of "BGR" or "RGB".

    Returns:
        Tuple of (frames as numpy uint8 ndarray in the video, frame indices). The returned
        `frame_indices` is an absolute index. For the details about the frames, see the
        documentation for `extract_frames_from_frame_indices`. The frames returned are guaranteed
        to not have `None` objects.
    """
    assert isinstance(stride, int) and stride >= 1

    # creating list of frames to be extracted
    total_num_frames = get_video_number_frames(cap)

    if num_frames is not None:
        end_frame_index = int(min(start_frame_index + num_frames * stride, total_num_frames))
    else:
        end_frame_index = total_num_frames

    frame_indices = range(start_frame_index, end_frame_index, stride)

    frames = extract_frames_from_frame_indices(cap, frame_indices, color_format=color_format)

    # trim all the None frames and indices from the returned list
    earliest_failed_index = None
    for f_idx, frame in enumerate(frames):
        if frame is None:
            earliest_failed_index = f_idx
            break

    if earliest_failed_index is not None:
        assert earliest_failed_index < len(frames)
        frames = frames[:earliest_failed_index]
        frame_indices = frame_indices[:earliest_failed_index]

    return frames, frame_indices

def extract_frames_from_frame_indices(cap, frame_indices, color_format="BGR"):
    """Extract all the frames given a frame index list.

    Args:
        cap             : VideoCapture object from opencv.
        frame_indices   : List of frame indices.
        color_format    : String of "BGR" or "RGB".

    Returns:
        List of frames as numpy ndarray in the video. The default is to return the BGR format
        consistent with OpenCV. If  This list is guaranteed to be the same length
        as `frame_indices`. If frame index is out-of-bound, the corresponding returned frame will be
        None.
    """
    if color_format == "RGB":
        logger.debug(f"Color conversion from BGR (OpenCV default) to RGB")
    logger.info(f"Extracting frames from frame indices of size {len(frame_indices)}")
    frames = []
    for frame_index in tqdm(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret and color_format == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    logger.debug(f"{len(frames)} frames of type {type(frames[0]) if len(frames) > 0 else None} extracted.")

    return frames

def save_frames_into_mp4(frames: Union[List[np.ndarray], np.ndarray],
                         out_file: Union[str, os.PathLike],
                         fps: int = 30,
                         debug_text: Union[None, str] = None):
    """Save frames into a file.

    Args:
        frames  : A (list of) numpy.ndarray of uint8 from 0-255; each has dimension h, w, c in BGR.
        out_file: Output file.
    """
    if len(frames) == 0:
        logger.warning(f"Input frames have zero length. Skipping; no file is saved.")
        return

    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    fps = fps

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1

    color = (255, 255, 255)
    thickness = 2
    position = (10, 30)

    out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

    for f_idx, frame in enumerate(frames):
        text = f"{debug_text}-{f_idx}"
        if debug_text is not None:
            overlay_frame = frame.copy()
            cv2.putText(overlay_frame, text, position, font, font_scale, color, thickness,
                        cv2.LINE_AA)
            out.write(overlay_frame)
        else:
            out.write(frame)
    out.release()
    logger.info(f"Video saved as {out_file}")
