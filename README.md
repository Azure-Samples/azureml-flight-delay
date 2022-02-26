# Flight Delay ML Demo

This repo contains a set of Azure ML demos to introduce Azure ML and the ML Pro "Hero Scenarios" across  the different experiences. It is based on a demo built for Gartner to show off Azure ML and has been used widely since by the field when presenting to customers. 

The demo starts with setting up a workspace usnig an ARM template. A script walks you through traning a model to predict if an airline flight is likely to be delayed based up on past history. The model training is done using AutoML, Designer, and Notebooks with Python scripts. Along the way you will work with data, test models using an endpoint, and evaluate Responsible AI metrics. 

The demo walk through document is in azureml-flight-delay/tree/main/docs/AzureML Overview Demo.docx

## Getting started

### Prerequisites

The following prerequisites are required:
1. Azure subscription 
2. Owner role on the subscription or a resource group in the subscription to create the Workspace and required Azure resources

### Setup

AzureML Overview Demo.docx under the `docs` directory walks you through setup and the various demos of Azure ML scenarios.

This button will launch the "Tempalte deployment with a custom template" experience in the Azure Portal to create the workspace and required resources.

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure-Samples%2Fazureml-flight-delay%2Fmain%2FarmTemplates%2Faml.json)


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
