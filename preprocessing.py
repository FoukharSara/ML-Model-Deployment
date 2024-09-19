import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Labels for cats and dogs
data_dir= 'data'

categories = {
    'Cat': 0,
    'Dog': 1
}

def load_data(data_dir, categories):
    data = []
    labels = []
    for category, label in categories.items():
        path = os.path.join(data_dir, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
            if image is not None:
                image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)  # Resize to 64x64
                data.append(image)
                labels.append(label)
    data = np.array(data).reshape(-1, 64, 64, 1)  # Reshape for the model
    data = data / 255.0  # Normalize pixel values
    labels = np.array(labels)  # Convert labels to numpy array
    return data, labels

data,labels = load_data(data_dir,categories)
X_train,X_test,Y_train,Y_test=train_test_split(data,labels,test_size=0.2,random_state=42)

datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)



np.save("data/X_train.npy",X_train)
np.save("data/X_test.npy",X_test) 
np.save("data/Y_train.npy",Y_train) 
np.save("data/Y_test.npy",Y_test)
