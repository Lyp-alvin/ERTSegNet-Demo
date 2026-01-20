# ERTSegNet: Robust Recognition of Anomalous Distribution from ERT

This repository contains the official implementation and minimal demonstration for the paper:
**"Robust Recognition of Anomalous Distribution from Electrical Resistivity Tomography"** (IEEE Geoscience and Remote Sensing Letters).

##  Repository Contents
* `model.py`: The implementation of the ERTSegNet (VSSM) network.
* `demo.py`: The script to run inference and reproduce the quantitative metrics.
* **Validation Data**: The dataset folders (`Labels`, `small`, `middle`, `large`) are currently located in the root directory and need to be organized (see instructions below).

> **Note:** The model weights are hosted in the **[Releases](../../releases)** section due to file size limits.

##  Environment Setup (Important)
The model relies on **VMamba** and requires a robust CUDA environment. Please follow the steps below to set up the environment using Conda to ensure successful compilation of CUDA kernels.

**1. Create and Activate Conda Environment**
```bash
conda create -n mamba_env python=3.11 -y
conda activate mamba_env
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo "CUDA_HOME is set to: $CUDA_HOME"
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install timm scipy einops transformers ninja packaging matplotlib scikit-learn tqdm
git clone https://github.com/state-spaces/mamba
cd mamba
pip install . --no-build-isolation
```

##  Setup & Usage
To run the demo, please follow these steps to prepare the directory structure.

Step 1: Download Model Weights
Go to the Releases page of this repository and download the Checkpoints.zip file.


Step 2: Organize Data Directory
The validation data folders (Labels, small, middle, large) are currently in the root directory. Please move them into a Mini Dataset folder to match the script's configuration.

1.Create a folder named Mini Dataset in the root.

2.Inside Mini Dataset, create a subfolder named Inputs.

3.Move the folders as follows:

Move Labels/ --> inside Mini Dataset/

Move small/ --> inside Mini Dataset/Inputs/

Move middle/ --> inside Mini Dataset/Inputs/

Move large/ --> inside Mini Dataset/Inputs/

Step 3: Verify Structure
Before running the code, ensure your final directory structure looks exactly like this:

```text
ERTSegNet-Demo/
├── Checkpoints/
│   └── ERTSegNet.pth      <-- Downloaded from Releases
├── Mini Dataset/               <-- Created manually
│   ├── Labels/                 <-- Moved from root
│   └── Inputs/                 <-- Created manually
│       ├── small/              <-- Moved from root
│       ├── middle/             <-- Moved from root
│       └── large/              <-- Moved from root
├── model.py
├── demo.py
└── README.md
```
Step 4: Run Inference
Once the environment is active and the structure is correct, run the demo script:

```Bash

python demo.py
```
