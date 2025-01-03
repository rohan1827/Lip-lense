# Import all of the dependencies
import streamlit as st
import os
import imageio
from moviepy.editor import VideoFileClip

import tensorflow as tf
from utils import load_data, num_to_char, load_video
from modelutils import load_model

# Set the layout to the streamlit app as wide
st.set_page_config(layout="wide")

# Setup the sidebar
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Lip Net App :) ")
    st.info("Inspired by the LipNet Model!.")

# st.title("LipNet App")
st.markdown(
    "<h1 style='text-align: center; color: vividblue;'>Lip Sense 🎥</h1>",
    unsafe_allow_html=True,
)
# Generating a list of options or videos
options = os.listdir(
    os.path.join("..", "custom_videos")
)  # change the directory to bring the S1 out of the videos and just have it there.

selected_video = st.selectbox("Choose video", options)
file_path = os.path.join("..", "custom_videos", selected_video)


# Generate two columns
col1, col2 = st.columns(2)

if options:
    # Rendering the video
    with col1:
        pass
        # st.info("Rendering video", icon="🎬")
        # file_path = os.path.join("..", "demo-data", "videos", "s1", selected_video)

        # # Convert video to mp4 format ; Load the .mpg video
        # clip = VideoFileClip(file_path)
        # # Write the video as .mp4
        # output_path = os.path.join("..", "demo-data", "test video.mp4")
        # clip.write_videofile(output_path, codec="libx264")

        # # Rendering inside of the app
        # video = open(os.path.join("..", "demo-data", "test video.mp4"), "rb")
        # video_bytes = video.read()
        # st.video(video_bytes)

    with col2:
        st.info(
            "This is all the machine learning model sees",
            icon="🤖",
        )
        print()
        # video, annotations = load_data(tf.convert_to_tensor(file_path))
        video = load_video(file_path)
        imageio.mimsave("animation.gif", video, fps=10)
        st.image("animation.gif", width=900)

        st.info("This is the output of the machine learning model as tokens")
        model = load_model()

        # Predictions from the loaded model
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info("Decode the raw tokens into words")
        converted_prediction = (
            tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")
        )
        st.text(converted_prediction)
