# MolDisc 1.0


MolDisc is an autonomus molecualr discovery tool based on SMILES (Simplified Molecular Input Line Entry System). 

## MolDisc Pipeline

MolDisc is developed using following existing tools,
- SMILESX (https://github.com/Lambard-ML-Team/SMILES-X)
- GPT2 (https://huggingface.co/openai-community/gpt2)
- RDKit (https://www.rdkit.org/)

The overall pipline of MolDisc is shown the following figure.

![Pipeline](moldisc_pipeline.jpg)

Input data to the MolDisc has to be provided as a label.csv and unlabeled.csv files containing SMILES with labels and only SMILES respectively. At the first state of pipeline SMILESX is trained with labeled data. In the second stage of the pipeline both labeled and unlabeled SMILES are used to finetune the pretrained GPT model. From the trained GPT model new SMILES are generated which are passed through RDKit to sanitize to retain valid molecules. The trained SMILESX model is used to predict the properties of SMILES to select an appropriate set to aggregate to the unlabled dataset. The pipeline continues SMILES generations until the satisfaction of termination conditions. For more details of the pipeline please refer to the paper (paper.pdf).     


## Installation

MolDisc requires a [conda environment](https://www.anaconda.com/). 
We recommend miniconda for non-commercial usage. Installation guide for miniconda is available here https://www.anaconda.com/docs/getting-started/miniconda/install.

Moldisc requires both TensorFlow and Pytorch. Due to possible version conflicts of TensorFlow and Pytorch MolDisc requires two separate conda environments for SMILES-X and GPT. <!--In order to create and install the conda environments  run the requirements_main.txt and requirements_gpt.txt as follows -->

#### Setting up environment for SMILES-X

Execute the following command to create the environment *moldisc_main* for SMILES-X installation with TensorFlow.

```
conda env create -f environment.yml
conda activate moldisc_main
```
Run the following commands to install the required SMILES-X and related software.

```
python -m venv moldisc_main
source moldisc_main/bin/activate  # Windows: moldisc_main\Scripts\activate
pip install -r requirements.txt
```

#### Setting up environment for GPT2

Next, Execute the following command to create the environment for GPT2 *subGPT* for GPT-2 installation with Pytorch.

```
conda env create -f environmentGPT.yml
conda activate subGPT
```

After activating the *subGPT* environment and run the following commands to install necessary software for the GPT2 model.

```
python -m venv subGPT
source subGPT/bin/activate  # Windows: subGPT-test\Scripts\activate
pip install -r requirementsGPT.txt
```

## Data Folder

Create a folder (preferable with project name) inside the data folder to store data for traning and inference. Install labeled and unlabled smiles in CSV file format with names  labeled.csv and unlabled.csv, respectively.




## Tutorial

A step-by-step guide for molecular generation is available in this [Jupyter tutorial](./example.ipynb).

## Reference

Use the following reference to cite MolDisc

```
-- ,-- ,-- , 
```
