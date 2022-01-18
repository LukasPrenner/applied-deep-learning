import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.backend import set_session
from skimage.transform import resize
from sklearn.preprocessing import Normalizer
from numpy import expand_dims, asarray
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from PIL import Image
from os import listdir
import tensorflow as tf
import numpy as np
import pickle

# extracts a single face from an image provided via file_path


def extract_face(file_path, required_size=(160, 160)):
    image = Image.open(file_path)
    image = image.convert('RGB')
    pixels = asarray(image)
    detector = MTCNN()
    detector_results = detector.detect_faces(
        pixels)  # detect faces in the image
    # extract bounding box from face
    x1, y1, width, height = detector_results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]  # extract the face
    image = Image.fromarray(face)
    # resize pixels to required model size (160 x 160 px)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# gets face embedding for a single face


def get_embedding(facenet_model, face_pixels):
    face_pixels = face_pixels.astype('float32')  # scale pixel values
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std  # standardize pixel values
    sample = expand_dims(face_pixels, axis=0)  # transform face into one sample
    y_hat = facenet_model.predict(sample)  # get embedding via facenet model
    return y_hat[0]


print("Loading model...")
global sess
global is_running
is_running = False
sess = tf.compat.v1.Session()
set_session(sess)
# global model

global graph
graph = tf.compat.v1.get_default_graph()
print("Model loaded successfully.")

app = Flask(__name__, static_url_path="", static_folder="templates")


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')


@app.route('/prediction/<filename>')
def prediction(filename):
    my_image = plt.imread(os.path.join('uploads', filename))
    my_image_re = my_image
    my_image_re = resize(my_image, (160, 160, 3))

    with graph.as_default():
        # import models
        path = '01_Models/kurz-schallenberg-model.sav'
        classifier = pickle.load(open(path, 'rb'))
        path = '01_Models/facenet_keras.h5'
        model = load_model(path)

        # imports encoders
        path = '02_Encoders/in_encoder.pkl'
        in_encoder = pickle.load(open(path, 'rb'))

        path = '02_Encoders/out_encoder.pkl'
        out_encoder = pickle.load(open(path, 'rb'))

        # gets face and embedding of a single image
        face = extract_face(os.path.join('uploads', filename))
        face_embedding = get_embedding(model, face)

        # preprocesses input (normalizing)
        face_embedding_normalized = in_encoder.transform(
            expand_dims(face_embedding, axis=0))

        # makes prediction
        y_hat_class = classifier.predict(face_embedding_normalized)
        y_hat_prob = classifier.predict_proba(face_embedding_normalized)

        # gets prediction name
        class_index = y_hat_class[0]
        class_probability = y_hat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(y_hat_class)
        print(y_hat_prob)
        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

        predictions = {
            "class1": predict_names[0],
            "prob1": class_probability,
        }
    if class_probability < 85:
        return render_template('predict_unsure.html')

    return render_template('predict.html', predictions=predictions)


app.run(host='0.0.0.0', port=80)
