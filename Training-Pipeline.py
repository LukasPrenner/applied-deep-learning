# imports packages
import sklearn
import pickle
from os import listdir
from os.path import isdir, isfile, join
from keras.models import load_model
from PIL import Image
import numpy as np
from numpy import asarray
from numpy import load
from numpy import expand_dims
from numpy import savez_compressed
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

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

# loads images and extract faces for all images in a directory


def load_faces(file_directory):
    faces = list()
    # iterate through all files in the directory
    for filename in listdir(file_directory):
        if not filename.startswith('.'):
            path = file_directory + filename
            face = extract_face(path)
            faces.append(face)
    return faces

# loads dataset grouped by class (=subdirectory) and returns as array


def load_dataset(file_directory):
    X, y = list(), list()
    # iterate through subfolders of each class
    for subdir in listdir(file_directory):
        if not subdir.startswith('.'):
            path = file_directory + subdir + '/'
            faces = load_faces(path)
            labels = [subdir for _ in range(len(faces))]  # create labels
            print('>loaded %d examples for class: %s' % (len(faces), subdir))
            X.extend(faces)
            y.extend(labels)
    return asarray(X), asarray(y)

# gets face embedding for a single face


def get_embedding(facenet_model, face_pixels):
    face_pixels = face_pixels.astype('float32')  # scale pixel values
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std  # standardize pixel values
    sample = expand_dims(face_pixels, axis=0)  # transform face into one sample
    y_hat = facenet_model.predict(sample)  # get embedding via facenet model
    return y_hat[0]

# iterates through raw faces and returns embeddings


def get_face_embeddings(data_X, facenet_model):
    new_data_X = list()
    for face_pixels in data_X:
        embedding = get_embedding(facenet_model, face_pixels)
        new_data_X.append(embedding)
    return asarray(new_data_X)

# gets raw face data as input and returns embeddings of faces


def convert_faces_to_embeddings(train_X, train_y, test_X, test_y):
    facenet_model = load_model('01_Models/facenet_keras.h5')  # load model
    new_train_X = get_face_embeddings(train_X, facenet_model)
    new_test_X = get_face_embeddings(test_X, facenet_model)
    return new_train_X, train_y, new_test_X, test_y

# preprocesses input and target data


def preprocess_data(train_X, train_y, test_X, test_y):
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    train_X = in_encoder.transform(train_X)
    test_X = in_encoder.transform(test_X)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(train_y)
    train_y = out_encoder.transform(train_y)
    test_y = out_encoder.transform(test_y)

    # save normalizer and scaler
    filename = 'in_encoder.pkl'
    pickle.dump(in_encoder, open(filename, 'wb'))
    filename = 'out_encoder.pkl'
    pickle.dump(out_encoder, open(filename, 'wb'))

    return train_X, train_y, test_X, test_y, out_encoder

# fits train data to defined model


def train_model(train_X, train_y, model=SVC(kernel='linear', probability=True)):
    return model.fit(train_X, train_y)


# loads raw train and test dataset
print("Loading Training Data...")
train_X, train_y = load_dataset('03_Data/train/')
print("Loading Test Data...")
test_X, test_y = load_dataset('03_Data/test/')

# gets embeddings of train and test data
print("Retrieving Face Embeddings...")
train_X, train_y, test_X, test_y = convert_faces_to_embeddings(
    train_X, train_y, test_X, test_y)

# preprocesses data
print("Preprocessing Data...")
train_X, train_y, test_X, test_y, out_encoder = preprocess_data(
    train_X, train_y, test_X, test_y)

# trains classifier
print("Training Model...")
model = train_model(train_X, train_y)

# saves trained classifier
filename = 'kurz-schallenberg-model.sav'
pickle.dump(model, open(filename, 'wb'))
print("Model successfully trained and saved.")
