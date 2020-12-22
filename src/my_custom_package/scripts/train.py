import argparse
import json
import urllib
import config
from pipeline_steps import train_steps
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib


import azureml.core
from azureml.core import Run
from azureml.core.dataset import Dataset
from azureml.core.datastore import Datastore
from azure.storage.blob import BlockBlobService
from io import StringIO
from azureml.core.model import Model



print("Executing train.py")
print("As a data scientist, this is where I write my training code.")
print("Azure Machine Learning SDK version: {}".format(azureml.core.VERSION))

#-------------------------------------------------------------------
#
# Processing input arguments
#
#-------------------------------------------------------------------

parser = argparse.ArgumentParser("train")

parser.add_argument("--model_name", type=str, help="model name", dest="model_name", required=True)
parser.add_argument("--build_number", type=str, help="build number", dest="build_number", required=True)
parser.add_argument("--output", type=str, help="output directory for saved model", dest="output", required=True)

args = parser.parse_args()

print("Argument 1: %s" % args.model_name)
print("Argument 2: %s" % args.build_number)
print("Argument 2: %s" % args.output)

#-------------------------------------------------------------------
#
# Define internal variables
#
#-------------------------------------------------------------------

datasets_folder = './datasets'

ds_name = 'surge_price_dataset'
ds_description = 'Surge price dataset consists of the data related to surge price class'



run = Run.get_context()
ws = run.experiment.workspace
ds = Datastore.get_default(ws)


#-------------------------------------------------------------------
#
# Process connected surge  dataset
#
#-------------------------------------------------------------------

print('Processing Surge price dataset...')

# Download the current version of the dataset and save a snapshot in the datastore
# using the build number as the subfolder name


block_blob_service = BlockBlobService(account_name='surgepricews0844156890',
                                      account_key='b8lRuyo8cO5ajGGt5D5n8cOifn8aDE4Uo3eDPOgAUbvI4JSSVsSG6vmLK/4Hxjchmp7VOd9nHrPYi1Ll6zI9QQ==')
# get data from blob storage in the form of bytes
blob_byte_data = block_blob_service.get_blob_to_bytes('azureml-blobstore-54e849ab-35db-4d06-aff7-1839f96d3aed','train.csv')
# convert to bytes data into pandas df to fit scaler transform
s = str(blob_byte_data.content, 'utf-8')
bytedata = StringIO(s)
final = pd.read_csv(bytedata)

final_ds = Dataset.Tabular.from_delimited_files(path=[(ds, '/train.csv')])

# For each run, register a new version of the dataset and tag it with the build number.
# This provides full traceability using a specific Azure DevOps build number.

final_ds.register(workspace=ws, name="Surge Dataset", description="Surge Dataset",
                  tags={"build_number": args.build_number}, create_new_version=True)
print('surge dataset successfully registered.')



print("Processing  components data completed.")


#-------------------------------------------------------------------
#
# Create training, validation, and testing data
#
#-------------------------------------------------------------------


X = final.drop(config.TARGET, axis=1)
y = final[config.TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




#-------------------------------------------------------------------
#
# Build and train the model
#
#-------------------------------------------------------------------

# intiate the pipeline to train
train_steps.fit(X_train, y_train)

print("Training model completed.")

print("Saving model files...")
os.makedirs(args.output, exist_ok=True)

# Convert the ML model to Pickle
# Save the model locally...
filename = os.path.join(args.output, "model.joblib")
joblib.dump(value=train_steps, filename=filename)

print("Saving model files completed.")


#-------------------------------------------------------------------
#
# Evaluate the model
#
#-------------------------------------------------------------------

print("Creating Model")

y_predict = train_steps.predict(X_test)
acc = metrics.accuracy_score(y_test, y_predict)
precision = metrics.precision_score(y_test, y_predict, average='weighted')
recall = metrics.recall_score(y_test, y_predict, average='weighted')  #
f1 = metrics.f1_score(y_test, y_predict, average='weighted')

print("Model build completed")

# print(metrics.classification_report(y_test, y_predict))
print('Accuracy is {}'.format(acc))
print('Precision is {}'.format(precision))
print('Recall is {}'.format(recall))
print('F1-Score is {}'.format(f1))

print("Running model completed")

run.log('Accuracy', acc)
run.log('Precision', precision)
run.log('Recall', recall)
run.log('F1-score', f1)

train_info = {}
train_info['train_run_id'] = run.id
train_filepath = os.path.join(args.output, 'train_info.json')
with open(train_filepath, "w") as f:
    json.dump(train_info, f)
print('train_info.json saved!')
