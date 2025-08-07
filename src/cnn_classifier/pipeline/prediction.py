import numpy as np
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image




class PredictionPipeline:
    
    """
    A pipeline class to handle the prediction of Adenocarcinoma or Normal class
    using a pre-trained CNN model.
    """

    def __init__(self,filename):
        
        """
        Initialize the prediction pipeline.

        Parameters : 
            (a) filename (str): The path to the input image file.
        """

        self.filename=filename


    def predict(self):
        
        """
        Loads the pre-trained model, processes the input image, and performs prediction.

        Returns:
            List[dict]: A list containing a dictionary with the predicted label.
                        Example: [{ "image": "Normal" }] or [{ "image": "Adenocarcinoma Cancer" }]
        """

        # load model
        
        # uncomment below line if running on the local system
        # Use below line with an absolute path if we're running the code outside the project root (e.g., in an IDE or different working directory)
        #model = load_model("E:/STUDY/TENSORFLOW/Projects/1_CNN_Project/trained_model/training/trained_model.keras")
        
        # Use below line if the script is run from the project root (recommended and works in both local and production environments)
        model = load_model(os.path.join("trained_model","training","trained_model.keras"))

        # Load the image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224)) 
        test_image = image.img_to_array(test_image) # convert the img into numpy array 
        test_image = np.expand_dims(test_image, axis = 0) # dim : [1,224,224]
        
        #test_image = test_image / 255.0  # Ensure this if model was trained with normalization

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
        
        
        
        