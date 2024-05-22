# NONOS: Neural Oscillation and Non-Oscillation Separator for Securing Temporal Resolution

This repository is the official implementation of [NONOS: Neural Oscillation and Non-Oscillation Separator for Securing Temporal Resolution]. 

![image](https://github.com/jkwrbcc/NONOS/assets/170528215/7f0c98e6-d32b-4e74-be92-f068da58d625)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Generate simulation data

To generate the simulation data in the paper, run this command:

```train
python generate_sim.py --exp_num 4_1_1
```

The experimental number (exp_num) corresponds to Section number in our paper.

## Training

To train the model(s) in the paper, run this command:

```train
python NONOS_train.py --input-data <path_to_data> --gpus 4 --epochs 200 --mode <simple or SpecParam> --specpara-result <path_to_result>
```

- We provide two versions of models: NONOS-simple and NONOS-SpecParam.
- For NONOS-SpecParam, a pre-calculated spectral parameterization result is required for smooth computation.


## Evaluation

To evaluate my model on ImageNet, run:

```eval
python NONOS_eval.py --model-file mymodel.checkpoint --input-data <path_to_data>
```

## Results

Our model achieves the following performance on :

![image](https://github.com/jkwrbcc/NONOS/assets/170528215/32012f6c-864a-476e-b2ae-d4bbc0d3a995)

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 
