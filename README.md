# Flight Delay ML Demo

**Important:** Work in progress. Demo data rehosted in Blob container (account name: azuremlexamples, container name: flight-delay, paths: data/, images/).

This repo contains the resources for an extended Azure ML demo based on the flight delay scenario.

## Getting started

### Prerequisites

The following prerequisites are required:
1. Azure subscription

For some modules, additional prerequisites are required:
1. Azure CLI
1. Visual Studio Code

### Setup

Follow the documentation under the `docs` directory to setup the required infrastructure for this demo.

In case you already have an Azure Machine Learning workspace, you can setup the necessary datasets and upload the notebooks into your existing workspace.

## Modules

This repository contains the following modules. Each is designed as a standalone demo; however, they can also be presented sequentially.

1. Automated ML UX
1. Designer
1. Notebooks
  * Automated ML & Responsible ML & ML Pipelines
  * Classical ML with VS Code
  * Deep Learning & Labeling
1. MLOps
1. MLOps with MLflow
1. Hybrid ML with Azure Arc
1. Security & Enterprise Readiness
1. Responsible ML
  * Differential Privacy with SmartNoise
  * Fairness, Explainability, and Error Analysis
  * Homomorphic Encryption

## Code Structure

| File/folder                                                                 | Description                                                                                                             |
| --------------------------------------------------------------------------  | ----------------------------------------------------------------------------------------------------------------------- |
| `armTemplates`                                                              | Main directory for ARM Templates used for the deployment of the infrastructure of this demo.                            |
| `azureStorageFiles`                                                         | Main directory for datasets used across this demo modules.                                                              |
| `docker`                                                                    | Docker image requiered for the Deep Learning notebook module.                                                           |
| `docs`                                                                      | Main directory for documentation, setup guides and scripts can be found here.                                           |
| `notebooks/flight-delay-arc`                                                | Flight Delay notebook with Azure Arc - Hybrid ML Training.                                                              |
| `notebooks/flight-delay-automl`                                             | Flight Delay notebook with Azure AutoML, Explainability, Fairlearn, Homomorphic Encryption and others.                  |
| `notebooks/flight-delay-automl-private`                                     | Flight Delay notebook for execution in a private AML workspace.                                                              |
| `notebooks/flight-delay-classicalml-local`                                  | Flight Delay notebook with local training and AKS deployment.                                                           |
| `notebooks/flight-delay-dl`                                                 | Flight Delay notebook with Keras Image Augmentation, TensorFlow, TensorBoard and AKS deployment.                        |
| `notebooks/flight-delay-mlops`                                              | Flight Delay notebooks with Dataset Datadrift, MLFlow and MLFlow projects.                                              |
| `notebooks/responsible-ml`                                                  | Responsible ML notebooks including SmartNoise, Fairlearn, Explainability and Homomorphic Encryption.                    |
