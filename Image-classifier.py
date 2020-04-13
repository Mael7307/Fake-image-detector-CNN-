

#CNN
import os
os.chdir("/Users/admin/Python/Project 2019/Convolutional_Neural_Network")
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model

# initialising CNN

classifier = Sequential()

# Convolution
classifier.add(Convolution2D(32, (3,3),input_shape = (64,64,3), activation = 'relu'))
# Pooling
classifier.add(MaxPooling2D(2,2))

# 2nd Convolution

classifier.add(Convolution2D(32,(3,3), activation = 'relu'))

# 2nd pooling

classifier.add(MaxPooling2D(2,2))

# 3rd Convolution
classifier.add(Convolution2D(64, (3,3), activation = 'relu'))

# 3rd pooling

classifier.add(MaxPooling2D(2,2))

#Flatten for FC

classifier.add(Flatten())

# Full connections

classifier.add(Dense( output_dim = 128, activation = 'relu'))
classifier.add(Dense( output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# training set

training_set = train_datagen.flow_from_directory(
        'Dataset/Training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
# Test_set
test_set= test_datagen.flow_from_directory(
        'Dataset/Test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=2040,
        epochs=20,
        validation_data=test_set,
        validation_steps=7)

classifier.save('Classifier')
del classifier
classifier = load_model('Classifier')