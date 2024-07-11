import numpy as np
import os
from keras.preprocessing import image
import tensorflow.compat.v1 as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

tf.disable_eager_execution()
sess = tf.Session()
tf.disable_v2_behavior()
graph = tf.get_default_graph()

app = Flask(__name__)
set_session(sess)

# Load the model
model = load_model('adp.h5')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('alzheimers.html')

@app.route('/predict1', methods=['GET'])
def predict1():
    # Prediction page
    return render_template('alzpre.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']
        
        # Save the file to /uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static', 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Preprocess the image to the size expected by the model
        img = image.load_img(file_path, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        # Make prediction
        with graph.as_default():
            set_session(sess)
            prediction = model.predict(x)[0]
            print(prediction)
        
        # Determine prediction text based on your model's output
        prediction_class = np.argmax(prediction)
        if prediction_class == 0:
            text = "Mild Demented"
        elif prediction_class == 1:
            text = "Moderate Demented"
        elif prediction_class == 2:
            text = "Non Demented"
        else:
            text = "Very Mild Demented"
        
        return render_template('alzpre.html', result=text, image_path='static/uploads/' + secure_filename(f.filename))

if __name__ == "__main__":
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    app.run(debug=True)
