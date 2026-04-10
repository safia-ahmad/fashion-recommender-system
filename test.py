import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2
import os

# ---------------- LOAD FEATURES ----------------
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# ---------------- LOAD MODEL ----------------
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# ---------------- INPUT IMAGE ----------------
img_path = 'sample/shirt.png'

if not os.path.exists(img_path):
    raise FileNotFoundError(f"{img_path} not found")

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)

# ---------------- FEATURE EXTRACTION ----------------
result = model.predict(preprocessed_img, verbose=0).flatten()
normalized_result = result / norm(result)

# ---------------- SIMILARITY ----------------
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])

print("Indices:", indices)
print("Distances:", distances)

# ---------------- DISPLAY RESULTS ----------------
for i, file in enumerate(indices[0][1:6]):   # skip first (same image)
    temp_img = cv2.imread(filenames[file])

    if temp_img is None:
        print(f"Error loading image: {filenames[file]}")
        continue

    temp_img = cv2.resize(temp_img, (512, 512))


    cv2.imshow(f'output_{i}', temp_img)

cv2.waitKey(0)
cv2.destroyAllWindows()