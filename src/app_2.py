import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import tensorflow as tf
from werkzeug.security import generate_password_hash
from flask_sqlalchemy import SQLAlchemy 
from utils import (
    load_data,
    num_to_char,
    preprocess_video,
    load_video,
    predict_from_video,
)
from modelutils import load_model
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from.env file

app = Flask(__name__)
# MongoDB Atlas (Cloud Hosting)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQL_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
print('connected:', db)
# Load the model once when the application starts

global_model = load_model()
print('model loaded')

# Define your User model (example)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

    def __init__(self, name, email, password):
        self.name = name
        self.email = email
        self.password = password

# Route: Registration page
from flask import jsonify

@app.route('/', methods=['GET', 'POST'])
def register_user():
    if request.method == 'GET':
        return render_template('registration.html')
    
    elif request.method == 'POST':
        # Log user registration details
        app.logger.debug("Name: %s", request.form['name'])
        app.logger.debug("Email: %s", request.form['email'])
        app.logger.debug("Password: %s", request.form['password'])
        app.logger.info("User registration attempt")
        
        try:
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            
            # Check if the user is already registered
            users = db.session.query(User).filter_by(email=email).first()
            if users is None:
                # Hash password
                hashed_password = generate_password_hash(request.form['password'])
                
                # Insert user details into the database
                db.session.add(User(name=name, email=email, password=hashed_password))
                db.session.commit()
                app.logger.info("User registered successfully")
                
                # Return success response
                return jsonify({'success': True, 'message': 'Registration successful'})
            else:
                # User already exists
                app.logger.warning("User already exists: %s", email)
                return jsonify({'success': False, 'message': 'User already exists'}), 400
        except Exception as e:
            app.logger.error("Error during registration: %s", str(e))
            return jsonify({'success': False, 'message': str(e)}), 500

    return render_template('registration.html')


# Route: Index page after registration
# @app.route('/predict')
# def index():
#     return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Render the prediction page 
        return render_template('predict.html')
    
    elif request.method == 'POST':
        video_path = None  # Define the variable to ensure it exists in all cases
        try:
            # Check if a file was uploaded
            if 'video' not in request.files:
                app.logger.warning("No video file uploaded")
                return jsonify({'error': 'No video file uploaded'}), 400
            
            video_file = request.files['video']
            
            # Save the uploaded video temporarily
            video_path = os.path.join('uploads', video_file.filename)
            os.makedirs('uploads', exist_ok=True)
            video_file.save(video_path)
            
            # Load video and preprocess
            app.logger.debug("Processing video: %s", video_path)
            processed_video = load_video(video_path)
            
            # Log shape of processed video for debugging
            app.logger.debug("Processed video shape: %s", processed_video.shape)
            
            # Expand dimensions for model prediction
            input_video = tf.expand_dims(processed_video, axis=0)
            
            # Run inference using the global model
            app.logger.debug("Running model prediction")
            yhat = global_model.predict(input_video)
            
            # Log prediction shape for debugging
            app.logger.debug("Prediction shape: %s", yhat.shape)
            
            # Decode the prediction
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            
            # Convert prediction to text
            converted_prediction = (
                tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")
            )
            
            # Log converted prediction
            app.logger.debug("Converted Prediction: %s", converted_prediction)
            
            # Return prediction
            return jsonify({'prediction': converted_prediction})
        
        except Exception as e:
            app.logger.error("Error during prediction: %s", str(e))
            return jsonify({'error': 'I Cant Understand that..sorry😕'}), 500
        
        finally:
            # Clean up the uploaded file
            if video_path and os.path.exists(video_path):
                os.remove(video_path)


if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Ensure uploads directory exists
    os.makedirs('uploads', exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)