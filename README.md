# MolDisc 1.0 will be available soon . . .




## Installation

<!--Moldisc requires both tensorflow and torch environments. Due to possible version conflicts of tensorflow and torch MolDisc requires two separate conda environments for SMILES-X and GPT. In order to create and install the conda environments  run the requirements_main.txt and requirements_gpt.txt as follows -->

#### Setting up environment for SMILES-X

Execure the following command to create the environment for SMILES-X

```
conda create --name main_smilesx python=3.10
```
The following command activates the -- environment and install the required software for SMILES-X.

```
conda activate main_smilesx
pip install -r requirements_main.txt
```

#### Setting up environment for GPT2

Next, Execure the following command to create the environment for GPT2

```
conda create env subGPT python=3.10
```

The following command activates the -- environment and install the required software for GPT2.

```
conda activate subGPT
pip install -r requirements_gpt.txt
```


## 


## Tutorial

A step-by-step guide for molecular generation is available in this [Jupyter tutorial](./example.ipynb).
