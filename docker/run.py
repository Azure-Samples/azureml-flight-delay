#%%
import os
os.chdir('./train')

#%%
# Configure workspace
from azureml.core import Workspace

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')

#%%
# Setup compute
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your GPU cluster
cluster_name = "gpucluster"

# Verify that cluster does not exist already
try:
    gpu_cluster = ComputeTarget(workspace = ws, name = cluster_name)
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6',
                                                           min_nodes=0,
                                                           max_nodes=4)
    gpu_cluster = ComputeTarget.create(ws, cluster_name, compute_config)

gpu_cluster.wait_for_completion(show_output=True)

#%%
# Prepare data
ds = ws.get_default_datastore()
ds.upload('./data')

#%%
# Configure experiment
from azureml.core import Experiment

experiment_name = 'hard-hat'
exp = Experiment(workspace=ws, name=experiment_name)

#%%
# Submit the experiment
from azureml.core import Run
from azureml.train.dnn import TensorFlow
from azureml.core.runconfig import AzureContainerRegistry, DockerEnvironment, EnvironmentDefinition, PythonEnvironment

registry = AzureContainerRegistry()
registry.address = 'contosomanufac5523805767.azurecr.io'
registry.username = 'contosomanufac5523805767'
registry.password = 'RC+cx6OiEhgK8MY1rSGkkaj8eYnGncNC'

docker_config = DockerEnvironment()
docker_config.enabled = True
docker_config.base_image = 'contosoml/base-gpu:0.2.1'
docker_config.base_image_registry = registry
docker_config.gpu_support = True

python_config = PythonEnvironment()
python_config.user_managed_dependencies = True

env_def = EnvironmentDefinition()
env_def.docker = docker_config
env_def.python = python_config

script_params = {
    '--model_dir': './outputs',
    '--pipeline_config_path': './faster_rcnn_resnet101_hardhats.config'
}

tf_est = TensorFlow(source_directory = './src',
                    script_params=script_params,
                    compute_target=gpu_cluster,
                    entry_script='train.py',
                    inputs=[ds.as_download(path_on_compute='/data')],
                    environment_definition=env_def)
run = exp.submit(tf_est)
run

#%%
run.wait_for_completion(show_output=True)

#%%
os.chdir('..')