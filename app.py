from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin

from cnn_classifier.utils.common import decodeImage
from cnn_classifier.pipeline.prediction import PredictionPipeline

# Set environment variables for UTF-8 support
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')



app = Flask(__name__)
CORS(app)


class ClientApp:
    
    """
    Client-side application wrapper that initializes the image filename
    and loads the prediction pipeline.
    """

    def __init__(self):
        
        """
        Initializes the filename for the uploaded image and instantiates the PredictionPipeline class.
        
        """

        # whatever image is uploaded from frontend will be saved as inputimage.jpg
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)
        


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    
    """
    Renders the home page (index.html) for the web application.
    
    Returns:
        str: Rendered HTML content.
    """
    
    return render_template('index.html')



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    
    """
    Endpoint to handle image prediction requests from the frontend.
    
    Steps:
        1. Accepts base64 encoded image from JSON body.
        2. Decodes and saves it as an image file.
        3. Runs the CNN prediction pipeline.
        4. Returns the prediction result as JSON.

    Returns:
        Response: JSON response containing prediction result.
    """
    
    image = request.json['image']
    
    decodeImage(image, clApp.filename)
    # calling the method on the class object PredictionPipeline(self.filename)
    result = clApp.classifier.predict()

    response = jsonify(result)

    # To access the original dictionary (key-value pairs)
    data = response.get_json()  # Converts the jsonify object back to a dictionary
    print(data[0]["image"])

    return response




if __name__ == "__main__":
    
    # Instantiate the client application and run Flask server
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080)     # Accessible publicly (e.g., on AWS)




