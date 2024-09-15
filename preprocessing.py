import tensorflow as tf
import numpy as np 
import os
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)

train_path = 'data/train'
test_path='data/test'

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size= (200,200),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary",
    shuffle=True,
    seed=42   
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size= (200,200),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary",
    shuffle=True,
    seed=42   
)

# class_names = list(test_generator.class_indices.keys())
# print(class_names)