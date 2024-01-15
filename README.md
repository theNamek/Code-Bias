# Uncovering and Quantifying Social Biases in Code Generation

## Introduction
Data, trained classifiers, and code for the NeurIPS 2023 paper: Uncovering and Quantifying Social Biases in Code Generation

## Data
With 5 types of modifiers and 8 types of demographic dimensions, we construct our code prompt
dataset with 392 samples in total. We use this dataset to prompt Codex, InCoder, and CodeGen. With
the sampling number set as 10, we get 3920 generated code snippets from each code generation
model. We then ask humans to annotate the generated code.

## Trained Classifiers
In order to directly quantify the social bias in generated code, we propose to train code bias classifiers.
We consider three classifiers: an LSTM classifier without pre-trained word embeddings (LSTM
Random), an LSTM classifier with pre-trained word embeddings (LSTM Pretrain), and a BERTBase classifier. 

## Code
We conduct social bias analysis on three pre-trained code generation models with different quantities
of parameters: Codex (100B+), InCoder (1.3B), InCoder (6.7B), CodeGen (350M), CodeGen (2.7B),
and CodeGen (6.1B).

## Citation
```bibtex
@misc{liu2023uncovering,
      title={Uncovering and Quantifying Social Biases in Code Generation}, 
      author={Yan Liu and Xiaokang Chen and Yan Gao and Zhe Su and Fengji Zhang and Daoguang Zan and Jian-Guang Lou and Pin-Yu Chen and Tsung-Yi Ho},
      year={2023},
      eprint={2305.15377},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

