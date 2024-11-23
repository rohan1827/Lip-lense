import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    MaxPool3D,
    Activation,
    Reshape,
    SpatialDropout3D,
    BatchNormalization,
    TimeDistributed,
    Flatten,
)
import tensorflow as tf
from typing import List
import numpy as np
import cv2
import os


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


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


model = tf.keras.models.load_model(
    r"D:\repositories\Lip-Sense\notebooks\lip_reading_model.h5"
)
print("model_loaded")

file_path = "D:\repositories\Lip-Sense\demo-data\videos\s1\bbaf2n.mpg"
video = load_video(file_path)
yhat = model.predict(tf.expand_dims(video, axis=0))
decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()


converted_prediction = (
    tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")
)
print(converted_prediction)
