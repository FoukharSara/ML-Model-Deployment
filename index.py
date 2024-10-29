from flask import Flask, redirect, url_for, flash, request, send_file, render_template
import tensorflow as tf
from PIL import Image
import io
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = '123456'
model = load_model('model/emotion_detector_model.h5')

new_size = (64, 64)

@app.route('/detect', methods=["GET","POST"])
def detect():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    img = Image.open(file).convert('L')  # Convert to grayscale
    img_resized = img.resize(new_size)  # Resize to (64, 64)
    img_array = np.array(img_resized)

    # Add channels dimension (for grayscale it should be 1)
    img_array = np.expand_dims(img_array, axis=-1) 

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # Now shape is (1, 64, 64, 1)
    img_array = img_array / 255.0 

    prediction = model.predict(img_array)
    class_names = ['Cat', 'Dog']
    predicted_class = class_names[np.argmax(prediction)]

    # Return the predicted class in the response
    return f'The image is predicted to be: {predicted_class}'

@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
