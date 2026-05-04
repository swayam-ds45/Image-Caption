import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# Paths
BASE_DIR = r"c:\Users\swaya\OneDrive\Desktop\image caption"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "Images")
OUTPUT_FEATURES = os.path.join(DATASET_DIR, "features.pkl")

def extract_features(directory):
    # Load Xception model
    model = Xception(weights='imagenet')
    # Remove the classification layer to get the 2048-dim feature vector
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    print(model.summary())
    
    features = {}
    
    image_files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Total images found: {len(image_files)}")
    
    for img_name in tqdm(image_files, desc="Extracting Features"):
        img_path = os.path.join(directory, img_name)
        # Xception requires 299x299 image size
        image = load_img(img_path, target_size=(299, 299))
        image = img_to_array(image)
        # Reshape for the model (1, 299, 299, 3)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # Preprocess input (normalize pixels)
        image = preprocess_input(image)
        # Get features
        feature = model.predict(image, verbose=0)
        # Store feature
        features[img_name] = feature
        
    return features

if __name__ == "__main__":
    if os.path.exists(OUTPUT_FEATURES):
        print(f"Features already extracted and saved at {OUTPUT_FEATURES}")
    else:
        features = extract_features(IMAGES_DIR)
        with open(OUTPUT_FEATURES, 'wb') as f:
            pickle.dump(features, f)
        print(f"Successfully extracted features for {len(features)} images and saved to {OUTPUT_FEATURES}")
