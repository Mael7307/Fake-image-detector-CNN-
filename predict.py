import os
os.chdir("/Users/admin/Python/Project 2019/Convolutional_Neural_Network")
import numpy as np
from keras.preprocessing import image
from keras.models import load_model



#Load Model
model = load_model('classifier')

#Compile
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

test_image = image.load_img("unknown img/test.jpg", target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

classes = model.predict(test_image)

if classes[0][0] == 1:
    prediction = 'Real'
else:
    prediction = 'Fake'
    
print (prediction) 

