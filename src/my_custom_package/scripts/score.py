# pylint: skip-file
import json
import os
import joblib
import numpy as np
from azureml.core.model import Model
from scripts import preprocessing_code




def init():
    global model
    # Get the path where the deployed model can be found.
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'))
    # load models
    model = joblib.load(model_path + '/model.joblib')


def run(data):
    try:
        data = json.loads(data)
        data = data['data']
        result = model.predict(np.array(data))
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
