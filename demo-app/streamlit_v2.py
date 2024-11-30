import streamlit as st
import os
import imageio
from moviepy.editor import VideoFileClip

import tensorflow as tf
from utils import (
    load_data,
    num_to_char,
    preprocess_video,
    load_video,
    predict_from_video,
)
from modelutils import load_model

# Set the layout to the streamlit app as wide
st.set_page_config(layout="wide")


# Display the app title and subtitle
st.markdown(
    "<h1 style='text-align: center; color: vividblue;'>Lip Sense 🎥</h1>",
    unsafe_allow_html=True,
)

# Define columns
col1, col2 = st.columns(2)

# Column 1 content
with col1:
    st.info("Testing on the test set", icon="🔬")
    options = os.listdir(os.path.join("..", "demo-data", "videos", "s1"))
    selected_video = st.selectbox("Choose video", options)

    st.info("Rendering video", icon="🎬")
    file_path = os.path.join("..", "demo-data", "videos", "s1", selected_video)

    # Convert video to mp4 format and load the .mpg video
    clip = VideoFileClip(file_path)
    output_path = os.path.join("..", "demo-data", "test_video.mp4")
    clip.write_videofile(output_path, codec="libx264")

    # Render video inside the app
    with open(output_path, "rb") as video_file:
        video_bytes = video_file.read()
        st.video(video_bytes)

    # Model predictions
    model = load_model()
    video, annotations = load_data(tf.convert_to_tensor(file_path))
    yhat = model.predict(tf.expand_dims(video, axis=0))
    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

    # Convert prediction to text
    st.info("Model says:")
    converted_prediction = (
        tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")
    )
    st.text(converted_prediction)

# Column 2 content
with col2:
    st.info("Testing on custom videos", icon="📹")
    st.text("Dummy, change this later")

    custom_options = os.listdir(os.path.join("..", "custom_videos"))
    selected_video = st.selectbox("Choose video", custom_options)

    st.info("Rendering video", icon="🎬")
    custom_file_path = os.path.join("..", "custom_videos", selected_video)

    # Convert video to mp4 format and load the .mpg video
    if not file_path.lower().endswith(".mp4"):
        clip = VideoFileClip(custom_file_path)
        output_path = os.path.join("..", "demo-data", "test_video.mp4")
        clip.write_videofile(output_path, codec="libx264")

    # Render video inside the app
    with open(output_path, "rb") as video_file:
        video_bytes = video_file.read()
        st.video(video_bytes)

    # Model predictions
    model = load_model()

    # video = load_video(tf.convert_to_tensor(custom_file_path))
    # video = load_video(custom_file_path)
    # yhat = model.predict(tf.expand_dims(video, axis=0))
    # decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

    # # Convert prediction to text
    # st.info("Model says:")
    # converted_prediction = (
    #     tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")
    # )
    # st.text(converted_prediction)

result = predict_from_video(
    video_path=custom_file_path,
    model=model,
    slice_size=75,  # your model's expected frame count
    overlap=25,  # adjust based on your needs
)

print("Final prediction:", result)
