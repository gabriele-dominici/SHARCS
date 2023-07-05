# SHARCS :shark: : SHARed Concept Space for Explainable Multimodal Learning 

Official code for the paper [**SHARCS: Shared Concept Space for
Explainable Multimodal Learning**](https://arxiv.org/abs/2307.00316?context=cs.AI)

# Installation

**Requirement:** Python 3.9+


**Installing SHARCS**
```
cd src
python -m venv SHARCS
source  SHARCS/bin/activate
pip install -r requirements.txt
```

# Dataset

**XOR-AND-XOR**: created inside the `src/data_utils.py` file

**MNIST+SUPERPIXELS**: downloaded original dataset and merged inside the `src/data_utils.py` file

**HalfMNIST**: downloaded original dataset and merged inside the `src/data_utils.py` file

**CLEVR**: follow the instruction in the [official repository](https://github.com/facebookresearch/clevr-dataset-gen), constrining the generation to have only one object. In the question process you need to use the `src/clevr_data/questions.json` file as template. However, we provide a smaller dataset (100 questions) to try the models.   


# Running experiments

**Requirement:** Wandb already configured ([Section 1 and 2](https://docs.wandb.ai/quickstart))

```
wandb sweep --project SAHRCS ./config/[DATASET_NAME]_sweep.yaml
```
where `[DATASET_NAME]` could be `clevr`, `xor`, `halfmnist`, `mnist+superpixels`.

The previous command gives you the `[AGENT_ID]` and the full command to run the wandb agent. It is similar to the following one:
```
wandb agent [WANDB_ID]/SHARCS/[AGENT_ID]
```

Additionally, we set up `src/clevr.ipynb`, a Jupyter Notebook where it is possible to run step by step an experiments on CLEVR (it is set up with the smaller dataset included in the repository, therefore the results are not comparable with the one trained on the full dataset)
