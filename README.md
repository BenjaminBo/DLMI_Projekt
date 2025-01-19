# Setup
This repository was developed with Python 3.10.16
## Install Cuda 11.8 
Follow the steps from: https://developer.nvidia.com/cuda-11-8-0-download-archive
## Inatall dependencies
### Conda
```
conda install --yes --file requirements.txt
```
### Pip
```
pip install -r requirements.txt
```
# Dataset 
## Capsule Dataset
This repository was trained on a subset of the **SEEE-AI** dataset
- Subset: https://hessenbox.tu-darmstadt.de/getlink/fi8G8VfXgXPa8zEW8asqDL7D/project_capsule_dataset
- Original: https://www.kaggle.com/datasets/capsuleyolo/kyucapsule
# Usage
## Data Prep
Prepares the given dataset to a readable format for the models.

```
python main.py prep_data -r path_to_datase_root 
```
### Default values
```
-r: './project_capsule_dataset'
```

## Finetune model
Loads a model - through a given model name - and it's pretrained parameters. It will then be trained on the given datasetname (enough if the dataset is placed in this repository) or the given dataset path.

```
python main.py finetune -m model_name -d dataset_name/_path
```

### Default values
```
-m: "mobilenetv3_small_050"
-d: "capsule_dataset"
```