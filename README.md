# Flight Delay ML Demo

This repo contains a set of Azure ML demos to introduce Azure ML and the ML Pro "Hero Scenarios" across  the different experiences. It is based on a demo built for Gartner to show off Azure ML and has been used widely since by the field when presenting to customers. 

The demo starts with setting up a workspace usnig an ARM template. A script walks you through traning a model to predict if an airline flight is likely to be delayed based up on past history. The model training is done using AutoML, Designer, and Notebooks with Python scripts. Along the way you will work with data, test models using an endpoint, and evaluate Responsible AI metrics. 

The demo walk through document is in azureml-flight-delay/tree/main/docs/AzureML Overview Demo.docx

## Getting started

### Prerequisites

The following prerequisites are required:
1. Azure subscription (you need to be able to create new resources)

For some modules, additional prerequisites are required:
1. Azure CLI
1. Visual Studio Code

### Setup

Follow the documentation in AzureML Overview Demo.docx under the `docs` directory to setup the required infrastructure for this demo.

In case you already have an Azure Machine Learning workspace, you can setup the necessary datasets and upload the notebooks into your existing workspace.

## Modules

This repository contains the following modules. Each is designed as a standalone demo; however, they can also be presented sequentially.

1. Setup
2. Automated ML UX
3. Designer
4. Notebook with AutoML and Responsible AI
5. Notebook with MLOps, MLflow, and Managed Endpoints



## Code Structure

| File/folder                                                                 | Description                                                                                                             |
| --------------------------------------------------------------------------  | ----------------------------------------------------------------------------------------------------------------------- |
| `armTemplates`                                                              | Main directory for ARM Templates used for the deployment of the infrastructure of this demo.                            |
| `azureStorageFiles`                                                         | Main directory for datasets used across this demo modules.                                                              |
| `docker`                                                                    | Docker image requiered for the Deep Learning notebook module.                                                           |
| `docs`                                                                      | Main directory for documentation, setup guides and scripts can be found here.                                           |
| `notebooks/flight-delay-automl`                                             | Flight Delay notebook with Azure AutoML, Explainability, Fairlearn, Homomorphic Encryption and others.                  |
| `notebooks/flight-delay-mlops`                                              | Flight Delay notebooks with Dataset Datadrift, MLFlow and MLFlow projects.                                              |
