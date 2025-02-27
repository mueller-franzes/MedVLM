# MedVLM

Medical Vision Language Model

<br/>

## Step 1: Setup 
* Clone this repository 
* Make sure that your graphics card driver supports the current CUDA version. You can check this by running `nvidia-smi`. 
If not, change `pytorch-cuda=XX.X` in the [environment.yaml](environment.yaml) file or update your driver.
* Run: `conda env create -f environment.yaml`
* Run `conda activate MedVLM`


## Step 2: Run Training
* Run Script: [scripts/main_train.py](scripts/main_train.py)

## Step 3: Predict & Evaluate Performance
* Run Script: [scripts/main_predict.py](scripts/main_predict.py)


## Acknowledgment
The project is funded by the German Federal Ministry of Education and Research (BMBF) under the DataXperiment program (grant 01KD2430). 