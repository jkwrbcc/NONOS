>📋  A template README.md for code accompanying a Machine Learning paper

# NONOS: Neural Oscillation and Non-Oscillation Separator for Securing Temporal Resolution

This repository is the official implementation of [NONOS: Neural Oscillation and Non-Oscillation Separator for Securing Temporal Resolution]. 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --gpus 4 --epochs 200
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --input-data <path_to_data>
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Results

Our model achieves the following performance on :

![image](https://github.com/jkwrbcc/NONOS/assets/170528215/32012f6c-864a-476e-b2ae-d4bbc0d3a995)

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Acknowledgements
The implementation of CoST relies on resources from the following repositories, we thank the original authors for open-sourcing their work.
- https://neurodsp-tools.github.io/neurodsp/
- https://github.com/fooof-tools/fooof

>📋  Pick a licence and describe how to contribute to your code repository. 
