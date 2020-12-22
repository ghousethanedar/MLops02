import argparse
import azureml.core
from azureml.core import Workspace, Experiment, Run, Datastore
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData, PipelineRun, StepRun
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.authentication import AzureCliAuthentication
import os

print("In piplines_master.py")
print("Pipeline SDK-specific imports completed")
# Check core SDK version number
print("Azure ML SDK version:", azureml.core.VERSION)

parser = argparse.ArgumentParser("pipelines_master")
parser.add_argument("--aml_compute_target", type=str, help="compute target name", dest="aml_compute_target",
                    required=True)
parser.add_argument("--model_name", type=str, help="model name", dest="model_name", required=True)
parser.add_argument("--aks_name", type=str, help="aks name", dest="aks_name", required=True)
parser.add_argument("--aks_region", type=str, help="aks region", dest="aks_region", required=True)
parser.add_argument("--build_number", type=str, help="build number", dest="build_number", required=True)
parser.add_argument("--path", type=str, help="path", dest="path", required=True)
args = parser.parse_args()

print("Argument 1: %s" % args.aml_compute_target)
print("Argument 2: %s" % args.model_name)
print("Argument 3: %s" % args.aks_name)
print("Argument 4: %s" % args.aks_region)
print("Argument 5: %s" % args.build_number)
print("Argument 6: %s" % args.path)

print('creating AzureCliAuthentication...')
cli_auth = AzureCliAuthentication()
print('done creating AzureCliAuthentication!')

print('get workspace...')
ws = Workspace.from_config(path=args.path, auth=cli_auth)
print('done getting workspace!')

print("looking for existing compute target.")
aml_compute = AmlCompute(ws, args.aml_compute_target)
print("found existing compute target.")

# Create a new runconfig object
run_amlcompute = RunConfiguration()

# Use the cpu_cluster you created above. 
run_amlcompute.target = args.aml_compute_target

# Enable Docker
run_amlcompute.environment.docker.enabled = True

# Set Docker base image to the default CPU-based image
run_amlcompute.environment.docker.base_image = DEFAULT_CPU_IMAGE

# Use conda_dependencies.yml to create a conda environment in the Docker image for execution
run_amlcompute.environment.python.user_managed_dependencies = False

# Specify CondaDependencies obj, add necessary packages
run_amlcompute.environment.python.conda_dependencies = CondaDependencies.create(pip_packages=[
    'numpy',
    'pandas',
    'scikit-learn',
    'azure-storage-blob==2.1.0',
    'azureml-sdk',
    'azureml-dataprep[pandas]',

])

scripts_folder = 'scripts'
def_blob_store = ws.get_default_datastore()

train_output = PipelineData('train_output', datastore=def_blob_store)
print("train_output PipelineData object created")

trainStep = PythonScriptStep(
    name="train",
    script_name="train.py",
    arguments=["--model_name", args.model_name,

               "--build_number", args.build_number,
               "--output", train_output],
    outputs=[train_output],
    compute_target=aml_compute,
    runconfig=run_amlcompute,
    source_directory=scripts_folder,
    allow_reuse=False
)
print("trainStep created")

evaluate_output = PipelineData('evaluate_output', datastore=def_blob_store)

evaluateStep = PythonScriptStep(
    name="evaluate",
    script_name="evaluate.py",
    arguments=["--model_name", args.model_name,
               "--build_number", args.build_number,
               "--input", train_output,
               "--output", evaluate_output],
    inputs=[train_output],
    outputs=[evaluate_output],
    compute_target=aml_compute,
    runconfig=run_amlcompute,
    source_directory=scripts_folder,
    allow_reuse=False
)
print("evaluateStep created")

evaluateStep.run_after(trainStep)
steps = [evaluateStep]

pipeline = Pipeline(workspace=ws, steps=steps)
print("Pipeline is built")

pipeline.validate()
print("Simple validation complete")

run = Run.get_context()
experiment_name = run.experiment.name

pipeline_run = Experiment(ws, experiment_name).submit(pipeline)
print("Pipeline is submitted for execution")

pipeline_run.wait_for_completion(show_output=True, timeout_seconds=43200)

print("Get StepRun for evaluate step...")
pipeline_run_id = pipeline_run.id
step_run_id = pipeline_run.find_step_run('evaluate')[0].id
node_id = pipeline_run.get_graph().node_name_dict['evaluate'][0].node_id
print('Pipeline Run ID: {} Step Run ID: {}, Step Run Node ID: {}'.format(pipeline_run_id, step_run_id, node_id))
step_run = StepRun(run.experiment, step_run_id, pipeline_run_id, node_id)
print(step_run)

print("Downloading evaluation results...")
# access the evaluate_output
# data = pipeline_run.find_step_run('evaluate')[0].get_output_data('evaluate_output')
data = step_run.get_output_data('evaluate_output')
# download the predictions to local path
data.download('.', show_progress=True)

import json

# load the eval info json
with open(os.path.join('./', data.path_on_datastore, 'eval_info.json')) as f:
    eval_info = json.load(f)
print("Printing evaluation results...")
print(eval_info)

deploy_model = eval_info["deploy_model"]
aks_name = args.aks_name
aks_region = args.aks_region

compute_list = ws.compute_targets
aks_target = None
if aks_name in compute_list:
    aks_target = compute_list[aks_name]

if deploy_model and (aks_target == None):
    print("Model passed the evaluation criteria")
    print("No AKS found. Creating new Aks: {} for production deployment.".format(aks_name))
    prov_config = AksCompute.provisioning_configuration(location=aks_region)
    # Create the cluster
    aks_target = ComputeTarget.create(workspace=ws, name=aks_name, provisioning_configuration=prov_config)
    aks_target.wait_for_completion(show_output=True)
    print(aks_target.provisioning_state)
    print(aks_target.provisioning_errors)

print("Saving evaluation results for release pipeline...")
output_dir = os.path.join(args.path, 'outputs')
os.makedirs(output_dir, exist_ok=True)
filepath = os.path.join(output_dir, 'eval_info.json')

with open(filepath, "w") as f:
    json.dump(eval_info, f)
    print('eval_info.json saved!')
