# Canopy UI Helm Chart

A Helm chart for deploying the Canopy UI frontend application.

## Description

This chart deploys a web-based frontend interface for Canopy, an AI-powered text summarization application. The UI runs on port 8501 and can be configured to connect to various language model endpoints.

## Installation

```bash
helm install canopy-ui ./chart
```

## Configuration

The following table lists the configurable parameters and their default values:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `MODEL_NAME` | Name of the model to use | `"llama32"` |
| `LLM_ENDPOINT` | Language model endpoint URL | `""` |
| `BACKEND_ENDPOINT` | Endpoint URL for the backend service | `""` |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI | `""` |
| `MLFLOW_PROMPT_NAME` | MLflow prompt registry name to fetch the system prompt | `"summarization"` |
| `MLFLOW_PROMPT_VERSION` | MLflow prompt version to fetch (alias or number) | `"latest"` |
| `image.name` | Name of the container image | `"canopy-ui"` |
| `image.tag` | Tag of the container image | `"simple-0.3"` |


## Components

- **Deployment**: Runs the Canopy UI container
- **Service**: Exposes the application on port 8501
- **Route**: Provides external access to the application

## Chart Information

- **Version**: 0.0.9
- **Image**: `quay.io/rhoai-genaiops/canopy-ui:simple-0.3`
  
