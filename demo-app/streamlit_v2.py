import streamlit as st
import os
# import imageio
# from moviepy.editor import VideoFileClip

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

st.info("Testing on the test set", icon="🔬")
options = os.listdir(os.path.join("..", "demo-data", "videos", "s1"))
selected_video = st.selectbox("Choose video", options)

st.info("Rendering video", icon="🎬")
file_path = os.path.join("..", "demo-data", "videos", "s1", selected_video)

# Model predictions
model = load_model()
video, annotations = load_data(tf.convert_to_tensor(file_path))
yhat = model.predict(tf.expand_dims(video, axis=0))
decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

# # Convert prediction to text
st.info("Model says:")
converted_prediction = (
tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")
)
st.text(converted_prediction)