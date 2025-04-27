from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin

from cnn_classifier.utils.common import decodeImage
from cnn_classifier.pipeline.prediction import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:

    def __init__(self):

        # whatever image is uploaded from frontend will be saved as inputimage.jpg
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)





@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')



# @app.route("/train", methods=['GET','POST'])
# @cross_origin()
# def trainRoute():
#     os.system("python main.py")
#     # os.system("dvc repro")
#     return "Training done successfully!"




@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    
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
    
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080)     #for AWS




