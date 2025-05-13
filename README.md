# AML-Coursework-2025

Download the dataset from [here](https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets) and extract it.


```
data/
├── ORIGA/
│   ├── Images_Square/
│   └── Masks_Square/
├── G1020/
│   ├── Images_Square/
│   └── Masks_Square/
└── REFUGE/
    ├── Images_Square/
    └── Masks_Square/
```


### Running the U-Net Model

To train the U-Net model for optic disc and cup segmentation, run the following command:

```bash
python U-Net.py --ORIGA_path path/to/ORIGA --G1020_path path/to/G1020 --REFUGE_path path/to/REFUGE
```

#### Command Line Arguments:
- `--lr`: Learning rate (default: 0.0001)
- `--opt`: Optimizer type ("adam" or "sgd", default: "adam")
- `--total_epoch`: Total number of epochs for training (default: 10)
- `--ORIGA_path`: Path to ORIGA dataset directory
- `--G1020_path`: Path to G1020 dataset directory
- `--REFUGE_path`: Path to REFUGE dataset directory (used for validation)

#### Model Output:
- The model will print training and validation metrics after each epoch
- Best model weights will be saved as 'best_seg.pth' when validation performance improves
- Training metrics include:
  - Loss values
  - Dice score for optic disc (OD) and optic cup (OC) segmentation
  - Vertical cup-to-disc ratio (vCDR) error



### Preparing data for training U-Mamba and FednnU-Net

Edit `prepare_dataset.py` and set SRC_ROOT to the base directory of the dataset. Set NNUNET_RAW to U-Mamba/data/nnUNet_raw folder. Then run the following command to preprocess the dataset and convert it to the nnUNet format:

```bash
python prepare_dataset.py
```

### Running U-Mamba

#### Create and setup conda environment for U-Mamba:
```bash
# Create a new conda environment with Python 3.10
conda create -n u-mamba python=3.10 -y
conda activate u-mamba

# Install required dependencies
pip install torch
pip install causal-conv1d
pip install mamba-ssm

cd U-Mamba/umamba
pip install -e .
```

#### Preprocess the dataset:
```bash
nnUNetv2_plan_and_preprocess -d 202 --verify_dataset_integrity
```
#### Train the model:
```bash
nnUNetv2_train 202 2d all -tr nnUNetTrainerUMambaBot
```

#### Inference:
```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 202 -c 2d -f all -tr nnUNetTrainerUMambaBot --disable_tta
```


### Running FednnU-Net

#### Create and setup conda environment for FednnU-Net:
```bash
# Create a new conda environment with Python 3.10
conda create -n fednnunet python=3.10 -y
conda activate fednnunet

# Install required dependencies
cd FednnU-Net
pip install -e .
```

#### Set environment variables:
```bash
export nnUNet_raw="path/to/U-Mamba/data/nnUNet_raw"
export nnUNet_preprocessed="path/to/U-Mamba/data/nnUNet_preprocessed"
export nnUNet_results="path/to/U-Mamba/data/nnUNet_results"
```


#### Preprocess the dataset (skip this step if you have already done it for U-Mamba):
```bash
nnUNetv2_plan_and_preprocess -d 202 --verify_dataset_integrity
```

#### Train the model:
```bash
CUDA_VISIBLE_DEVICES=0 python nnunetv2/run/run_training.py 202 2d 5 --unseen_site 0 --client_num 4
```


#### Inference:
```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 202 -c 2d -f 5
```
