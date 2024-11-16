import tensorflow as tf
from typing import List
import numpy as np
import cv2
import os

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def load_video(path: str) -> List[float]:
    # print(path)
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


def load_alignments(path: str) -> List[str]:
    # print(path)
    with open(path, "r") as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != "sil":
            tokens = [*tokens, " ", line[2]]
    return char_to_num(
        tf.reshape(tf.strings.unicode_split(tokens, input_encoding="UTF-8"), (-1))
    )[1:]


def load_data(path: str):
    path = bytes.decode(path.numpy())
    file_name = path.split("/")[-1].split(".")[0]
    # File name splitting for windows
    file_name = path.split("\\")[-1].split(".")[0]
    video_path = os.path.join("..", "demo-data", "videos", "s1", f"{file_name}.mpg")
    alignment_path = os.path.join("..", "demo-data", "s1", f"{file_name}.align")
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments


def preprocess_video(
    path: str, target_frames: int = 75, height: int = 46, width: int = 140
):

    # Open video file
    cap = cv2.VideoCapture(path)
    frames = []

    while len(frames) < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale
        frame = tf.image.rgb_to_grayscale(frame).numpy()
        # Crop to the region of interest
        frame = frame[190:236, 80:220]
        frames.append(frame)

    cap.release()

    # Handle videos shorter than target_frames by repeating the last frame
    if len(frames) < target_frames:
        frames += [frames[-1]] * (target_frames - len(frames))

    # Convert to numpy array
    frames = np.array(frames, dtype=np.float32)

    # Normalize
    mean = np.mean(frames)
    std = np.std(frames)
    frames = (frames - mean) / std

    # # Ensure correct shape: (75, 46, 140, 1)
    # frames = np.expand_dims(frames, axis=-1)  # Add channel dimension

    # # Add batch dimension: (1, 75, 46, 140, 1)
    # frames = np.expand_dims(frames, axis=0)

    return frames


import numpy as np
from typing import List, Union
import cv2


def predict_from_video(
    video_path: str,
    model,
    slice_size: int = 75,
    overlap: int = 25,
    target_height: int = 46,
    target_width: int = 140,
) -> List[str]:
    """
    Complete pipeline for processing and predicting from a video.
    Handles variable length videos using sliding windows with overlap.

    Args:
        video_path: Path to the video file
        model: Trained LipNet model
        slice_size: Number of frames per slice (default: 75)
        overlap: Number of overlapping frames between slices (default: 25)
        target_height: Height to resize frames to (default: 46)
        target_width: Width to resize frames to (default: 140)

    Returns:
        List of predictions for the video
    """

    def load_video(video_path: str) -> np.ndarray:
        """Internal function to load and preprocess video."""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize
            resized = cv2.resize(gray, (target_width, target_height))

            # Normalize pixels to [0,1]
            normalized = resized.astype(np.float32) / 255.0

            # Add channel dimension
            normalized = normalized.reshape(target_height, target_width, 1)

            frames.append(normalized)

        cap.release()
        return np.array(frames)

    try:
        # Load and preprocess video
        print(f"Loading video: {video_path}")
        video_frames = load_video(video_path)
        total_frames = len(video_frames)

        if total_frames == 0:
            raise ValueError("No frames were loaded from the video")

        print(f"Loaded {total_frames} frames")

        # Process video in overlapping slices
        predictions = []
        stride = slice_size - overlap

        for start_idx in range(0, total_frames, stride):
            # Extract slice
            end_idx = start_idx + slice_size
            current_slice = video_frames[start_idx : min(end_idx, total_frames)]

            # Pad if necessary
            if len(current_slice) < slice_size:
                padding_needed = slice_size - len(current_slice)
                padding_frames = np.repeat(current_slice[0:1], padding_needed, axis=0)
                current_slice = np.concatenate([current_slice, padding_frames])

            # Prepare batch for model
            batch = current_slice.reshape(1, slice_size, target_height, target_width, 1)

            # Make prediction
            print(f"Predicting frames {start_idx} to {min(end_idx, total_frames)}")
            pred = model.predict(batch, verbose=0)

            # Decode prediction (modify this based on your model's output format)
            # Example: if your model outputs character probabilities
            decoded_pred = tf.keras.backend.ctc_decode(pred, [75], greedy=True)[0][
                0
            ].numpy()
            # You'll need to implement this
            predictions.append(decoded_pred)

        # Combine predictions (modify based on your needs)
        final_prediction = " ".join(predictions)

        print("Prediction complete!")
        return final_prediction

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None
