from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.applications import ResNet50
from pyspark.ml import PipelineModel
from keras import backend
import pandas as pd
from PIL import Image
import numpy as np
import flask
import io
import os
from pyspark.sql.session import SparkSession
from sklearn.externals import joblib
import requests
import json


# PUBLIC_IP = "0.0.0.0"
# NODE_PORT = "50000"

PUBLIC_IP = "20.184.57.73"
NODE_PORT = "80"

app = flask.Flask(__name__)
credit_model = None
application_url = "https://{}:{}/score".format(PUBLIC_IP, NODE_PORT)

def load_credit_model():
    global credit_model

    credit_model_path = os.path.join(os.getcwd(), 'models', 'credit', 'german_credit_risk.joblib')
    credit_model = joblib.load(credit_model_path)


@app.route("/v1/deployments/credit/online", methods=["POST"])
def credit_online():
    response = {}
    # labels = ['Risk', 'No Risk']

    if flask.request.method == "POST":
        payload = flask.request.get_json()

        if payload is not None:
            #df = pd.DataFrame.from_records(payload['values'], columns=payload['fields'])
            ip_values = payload['values']
            data = "{\"data\":[{}]}".format(ip_values)
            scoring_url = application_url + '/score'
            headers = {'Content-Type':'application/json'}
            resp = requests.post(scoring_url, data, headers=headers)
            predictions = json.loads(resp.text)
            response = {'fields': ['prediction'], 
                        'values': predictions}

    return flask.jsonify(response)


@app.route("/v1/deployments", methods=["GET"])
def get_deployments():
    response = {}

    if flask.request.method == "GET":
        response = {
            "count": 3,
            "resources": [
                
                {
                    "metadata": {
                        "guid": "credit",
                        "created_at": "2019-01-01T10:11:12Z",
                        "modified_at": "2019-01-02T12:00:22Z"
                    },
                    "entity": {
                        "name": "German credit risk compliant deployment",
                        "description": "Scikit-learn credit risk model deployment",
                        "scoring_url": "{}/score".format(application_url),
                        "asset": {
                              "name": "credit",
                              "guid": "credit"
                        },
                        "asset_properties": {
                               "problem_type": "binary",
                               "input_data_type": "structured",
                        }
                    }
                }

            ]
        }

    return flask.jsonify(response)


if __name__ == "__main__":
    #load_resnet50_model()
    #load_action_model()
    # load_credit_model()
    port = os.getenv('PORT', '5000')
    app.run(host='0.0.0.0', port=int(port))

