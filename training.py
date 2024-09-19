import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

X_train=np.load("data/X_train.npy")
X_test=np.load("data/X_test.npy") 
Y_train=np.load("data/Y_train.npy") 
Y_test=np.load("data/Y_test.npy")

model = models.Sequential([
    layers.Conv2D(128,(3,3), activation='relu',input_shape=(64,64,1)),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(32,(3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(16,(3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(X_train,Y_train,epochs=10, validation_data=(X_test, Y_test))

test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f'Test accuracy: {test_accuracy}')
model.save('model/emotion_detector_model.h5')

#Test accuracy: 0.8364729285240173