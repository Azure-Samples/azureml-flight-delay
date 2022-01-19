from azureml.core.runconfig import EnvironmentDefinition
from azureml.core import ContainerRegistry
from azureml.core.environment import PythonSection, DockerSection
registry = ContainerRegistry()
registry.address = '<address>'
registry.username = '<username>'
registry.password = '<password>'

docker_config = DockerSection()
docker_config.enabled = True
docker_config.base_image = 'aml/dl-image:v1'
docker_config.base_image_registry = registry

python_config = PythonSection()
python_config.user_managed_dependencies = True
