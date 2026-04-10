import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# -------------------- LOAD MODEL --------------------
model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


# -------------------- FEATURE FUNCTION --------------------
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / norm(result)

    return normalized_result


# -------------------- MAIN LOGIC --------------------
if os.path.exists('embeddings.pkl') and os.path.exists('filenames.pkl'):
    print("✅ Loading saved features...")

    feature_list = pickle.load(open('embeddings.pkl', 'rb'))
    filenames = pickle.load(open('filenames.pkl', 'rb'))

else:
    print("⏳ Extracting features... (only first time)")

    filenames = []
    for file in os.listdir('images'):
        filenames.append(os.path.join('images', file))

    feature_list = []

    for file in tqdm(filenames):
        feature_list.append(extract_features(file, model))

    pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
    pickle.dump(filenames, open('filenames.pkl', 'wb'))

    print("✅ Features saved successfully!")