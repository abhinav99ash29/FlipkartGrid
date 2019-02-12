from keras.models import model_from_json
import pandas as pd
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import cv2
import numpy as np

preprocess = imagenet_utils.preprocess_input

inputShape = (197, 197)

print('Loading the model')

json_file = open('Detector.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn = model_from_json(loaded_model_json)
cnn.load_weights("Detector.h5")

print('Loaded model from disk')

filename = 'images/'

test = pd.read_csv('test.csv')

for i in range(len(test)):
    image = test.iloc[i,0]
    image = cv2.imread(filename+str(test.iloc[i,0]))
    image = cv2.resize(image, inputShape)
    image = image[...,::-1]
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0) # image.shape = (1, 224, 224, 3)
    image = preprocess(image)
    out = cnn.predict(image)
    print(out)
    test.iloc[i,1] = out[0][0]*(640/197)
    test.iloc[i,2] = out[0][1]*(480/197)
    test.iloc[i,3] = out[0][2]*(640/197)
    test.iloc[i,4] = out[0][3]*(480/197)
    
    
test.to_csv('test_pred.csv',index=False)  
print('Done!')  
    

