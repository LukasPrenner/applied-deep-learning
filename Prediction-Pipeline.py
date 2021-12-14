# imports packages
import pickle
from sklearn.preprocessing import Normalizer
from keras.models import load_model
from numpy import expand_dims
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from PIL import Image
from os import listdir

# imports models
filename = '01_Models/kurz-schallenberg-model.sav'
classifier = pickle.load(open(filename, 'rb'))

filename = '01_Models/facenet_keras.h5'
facenet_model = load_model(filename)

# imports encoders
filename = '02_Encoders/in_encoder.pkl'
in_encoder = pickle.load(open(filename, 'rb'))

filename = '02_Encoders/out_encoder.pkl'
out_encoder = pickle.load(open(filename, 'rb'))

# extracts a single face from an image provided via file_path
def extract_face(file_path, required_size=(160, 160)):
	image = Image.open(file_path)
	image = image.convert('RGB')
	pixels = asarray(image)
	detector = MTCNN()
	detector_results = detector.detect_faces(pixels) # detect faces in the image
	x1, y1, width, height = detector_results[0]['box'] # extract bounding box from face
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	face = pixels[y1:y2, x1:x2] # extract the face
	image = Image.fromarray(face)
	image = image.resize(required_size) # resize pixels to required model size (160 x 160 px)
	face_array = asarray(image)
	return face_array

# gets face embedding for a single face
def get_embedding(facenet_model, face_pixels):
	face_pixels = face_pixels.astype('float32') # scale pixel values
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std # standardize pixel values
	sample = expand_dims(face_pixels, axis=0) # transform face into one sample
	y_hat = facenet_model.predict(sample) # get embedding via facenet model
	return y_hat[0]

# gets face and embedding of a single image
path = '04_Prediction-Data'
face = extract_face(path + "/" + listdir(path)[0])
face_embedding = get_embedding(facenet_model, face)

# preprocesses input (normalizing)
face_embedding_normalized = in_encoder.transform(expand_dims(face_embedding, axis=0))

# makes prediction
y_hat_class = classifier.predict(face_embedding_normalized)
y_hat_prob = classifier.predict_proba(face_embedding_normalized)

# gets prediction name
class_index = y_hat_class[0]
class_probability = y_hat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(y_hat_class)
print(y_hat_prob)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
