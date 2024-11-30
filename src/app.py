import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from utils import (
    load_data,
    num_to_char,
    preprocess_video,
    load_video,
    predict_from_video,
)
from modelutils import load_model

app = Flask(__name__)

# Load the model once when the application starts
global_model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file was uploaded
        if 'video' not in request.files:
            print("No video file uploaded")
            return jsonify({'error': 'No video file uploaded'}), 400
        
        video_file = request.files['video']
        
        # Save the uploaded video temporarily
        video_path = os.path.join('uploads', video_file.filename)
        os.makedirs('uploads', exist_ok=True)
        video_file.save(video_path)
        
        try:
            # Load video and preprocess
            print(f"Processing video: {video_path}")
            processed_video = load_video(video_path)
            
            # Print shape of processed video for debugging
            print(f"Processed video shape: {processed_video.shape}")
            
            # Expand dimensions for model prediction
            input_video = tf.expand_dims(processed_video, axis=0)
            
            # Run inference using the global model
            print("Running model prediction")
            yhat = global_model.predict(input_video)
            
            # Print prediction shape for debugging
            print(f"Prediction shape: {yhat.shape}")
            
            # Decode the prediction
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            
            # Convert prediction to text
            converted_prediction = (
                tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")
            )

            # Debug print of the prediction
            print(f"Converted Prediction: {converted_prediction}")
            
            # Return prediction
            return jsonify({
                'prediction': converted_prediction,
            })
        
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return jsonify({'error': str(e)}), 500
        
        finally:
            # Clean up the uploaded file
            if os.path.exists(video_path):
                os.remove(video_path)
    
    except Exception as e:
        print(f"Internal server error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Ensure uploads directory exists
    os.makedirs('uploads', exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)