# Nigeria_Crime

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py


# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/Milind-Shende/Nigeria_Crime.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n MLflow python=3.8 -y
```

```bash
conda activate MLflow
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```
## PyGWalker

[Documentation](https://docs.kanaries.net/pygwalker)
[How to use](https://docs.kanaries.net/pygwalker/use-pygwalker-with-streamlit)


## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/milind.shende24/Nigeria_Crime.mlflow \
MLFLOW_TRACKING_USERNAME=milind.shende24 \
MLFLOW_TRACKING_PASSWORD=12f6ae09a364684a207d59291ae873c4f015096e \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/milind.shende24/Nigeria_Crime.mlflow

export MLFLOW_TRACKING_USERNAME=milind.shende24 

export MLFLOW_TRACKING_PASSWORD=12f6ae09a364684a207d59291ae873c4f015096e