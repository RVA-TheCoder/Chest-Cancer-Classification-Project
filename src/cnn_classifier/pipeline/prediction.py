import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:

    def __init__(self,filename):

        self.filename=filename


    def predict(self):

        ## load model
        # model = load_model(os.path.join("artifacts","training", "trained_model.keras"))
        #model = load_model(os.path.join("model", "trained_model.keras"))
        model = load_model("E:/STUDY/TENSORFLOW/Projects/1_CNN_Project/trained_model/training/trained_model.keras")

        # Load the image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224)) 
        test_image = image.img_to_array(test_image) # convert the img into numpy array 
        test_image = np.expand_dims(test_image, axis = 0) # dim : [1,224,224]

        # Making Prediction
        prediction=model.predict(test_image)
        print("Model prediction : ",prediction)
        result = np.argmax(prediction, axis=1)
        print(" result of argmax : ",result)

        if result[0] == 1:
            prediction = 'Normal'
            return [{ "image" : prediction}]
        else:
            prediction = 'Adenocarcinoma Cancer'
            return [{ "image" : prediction}]