# Partial-fine-tuning-7B-model-Evaluation-with-Opencompass

Partial fine-tuning 7B model Evaluation with Opencompass

With the help of **opencompass**: <https://github.com/open-compass/opencompass> and **LLaMA-Factory**: <https://github.com/hiyouga/LLaMA-Factory>.

## Abstract

Comparison models' performance on MMLU (Massive Multitask Language Understanding) & CMMLU (Chinese Massive Multi-Level Language Understanding) two datasets all 124 different fields of data test.

Models include:

Llama-2-7b-chat-hf: <https://modelscope.cn/models/shakechen/Llama-2-7b-chat-hf>

Chinese-Alpaca-2-7B: <https://github.com/ymcui/Chinese-LLaMA-Alpaca-2>

Local partial-fine-tuning models

## Quick Start

### Clone the Repositories

```bash
git clone https://github.com/IAMFULAO/Partial-fine-tuning-7B-model-Evaluation-with-Opencompass.git
```

### Environment Setup

As this project uses two system to complete the partial-fine-tuning and evaluation, creating your own virtual environment with conda is strongly recommended.

```bash
cd Partial_fine_tuning_task

# Setup environment for opencompass
conda create --name opencompass python=3.10 -y
conda activate opencompass
cd opencompass 
pip install -e .

cd ..

# Setup environment for LLaMA-Factory
conda create --name llama-factory python=3.10 -y
conda activate llama-factory
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

## Brief Summary

### Improvement Effect between Models

<p align="center"><b>Llama-2-7B-chat VS LoRA-fine-tuning-model</b></p>

<div align="center">

| Parameters | Amplitude |
| :---: | :---: |
| **Average Accuracy (before fine-tuning)** | 31.77% |
| **Average Accuracy (after fine-tuning)** | 32.33% |
| **Average Improvement** | +0.56% |
| **Max Improvement** | +18.32% |
| **Min Improvement** | -17.48% |
| **Average Improvemnt on CMMLU** | +5.03% |
| **Average Improvemnt on MMLU** | -4.70% |

</div>

![origin-vs-lora](https://github.com/IAMFULAO/Partial-fine-tuning-7B-model-Evaluation-with-Opencompass/blob/main/comparison/origin-lora/summary_comparison.png?raw=true)


<p align="center"><b>Llama-2-7B-chat VS Chinese-Alpaca-7B</b></p>

<div align="center">

| Parameters | Amplitude |
| :---: | :---: |
| **Average Accuracy (before fine-tuning)** | 31.77% |
| **Average Accuracy (after fine-tuning)** | 34.79% |
| **Average Improvement** | +3.02% |
| **Max Improvement** | +26.88% |
| **Min Improvement** | -43.88% |
| **Average Improvement on CMMLU** | +9.86% |
| **Average Improvement on MMLU** | -5.02% |

</div>

![origin-vs-alpaca](https://github.com/IAMFULAO/Partial-fine-tuning-7B-model-Evaluation-with-Opencompass/blob/main/comparison/origin-alpaca/summary_comparison.png?raw=true)


<p align="center"><b>LoRA-fine-tuning-model VS Chinese-Alpaca-7B</b></p>

<div align="center">

| Parameters | Amplitude |
| :---: | :---: |
| **Average Accuracy (before fine-tuning)** | 32.33% |
| **Average Accuracy (after fine-tuning)** | 34.79% |
| **Average Improvement** | +2.46% |
| **Max Improvement** | +22.46% |
| **Min Improvement** | -33.33% |
| **Average Improvement on CMMLU** | +4.83% |
| **Average Improvement on MMLU** | -0.33% |

</div>

![lora-vs-alpaca](https://github.com/IAMFULAO/Partial-fine-tuning-7B-model-Evaluation-with-Opencompass/blob/main/comparison/lora-alpaca/summary_comparison.png?raw=true)

## Additions

.gitignore: Using the same one as that in the opencompass to ignore all datasets, picture result, and large files that cannot be pushed successfully.

LICENSE: MIT <https://raw.githubusercontent.com/github/choosealicense.com/gh-pages/_licenses/mit.txt>
