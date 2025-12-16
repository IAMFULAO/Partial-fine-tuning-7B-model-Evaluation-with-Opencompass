<style>
.center
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

# Partial-fine-tuning-7B-model-Evaluation-with-Opencompass

Partial fine-tuning 7B model Evaluation with Opencompass

With the help of opencompass: <https://github.com/open-compass/opencompass> and LLaMA-Factory: <https://github.com/hiyouga/LLaMA-Factory>.

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

<p align="center"><b><font face="宋体" size=3>Llama-2-7B-chat VS LoRA-fine-tuning-model</font></b></p>

<div align="center"><font face="宋体" size=2>

| 参数 | 幅度|
| :---: | :---: |
| **平均准确率 (微调前)** | 31.77% |
| **平均准确率 (微调后)** | 32.33% |
| **平均提升幅度** | +0.56% |
| **最大提升** | +18.32% |
| **最小提升** | -17.48% |
| **CMMLU 平均提升** | +5.03% |
| **MMLU 平均提升** | -4.70% |

</font></div>

![origin-vs-lora](https://github.com/IAMFULAO/Partial-fine-tuning-7B-model-Evaluation-with-Opencompass/blob/main/comparison/origin-lora/summary_comparison.png?raw=true)


<p align="center"><b><font face="宋体" size=3>Llama-2-7B-chat VS Chinese-Alpaca-7B</font></b></p>

<div align="center"><font face="宋体" size=2>

| 参数 | 幅度|
| :---: | :---: |
| **平均准确率 (微调前)** | 31.77% |
| **平均准确率 (微调后)** | 34.79% |
| **平均提升幅度** | +3.02% |
| **最大提升** | +26.88% |
| **最小提升** | -43.88% |
| **CMMLU 平均提升** | +9.86% |
| **MMLU 平均提升** | -5.02% |

</font></div>

![origin-vs-alpaca](https://github.com/IAMFULAO/Partial-fine-tuning-7B-model-Evaluation-with-Opencompass/blob/main/comparison/origin-alpaca/summary_comparison.png?raw=true)


<p align="center"><b><font face="宋体" size=3>LoRA-fine-tuning-model VS Chinese-Alpaca-7B</font></b></p>

<div align="center"><font face="宋体" size=2>

| 参数 | 幅度|
| :---: | :---: |
| **平均准确率 (微调前)** | 32.33% |
| **平均准确率 (微调后)** | 34.79% |
| **平均提升幅度** | +2.46% |
| **最大提升** | +22.46% |
| **最小提升** | -33.33% |
| **CMMLU 平均提升** | +4.83% |
| **MMLU 平均提升** | -0.33% |

</font></div>

![lora-vs-alpaca](https://github.com/IAMFULAO/Partial-fine-tuning-7B-model-Evaluation-with-Opencompass/blob/main/comparison/lora-alpaca/summary_comparison.png?raw=true)

## Additions

.gitignore: Using the same one as that in the opencompass to ignore all datasets, picture result, and large files that cannot be pushed successfully.

LICENSE: MIT <https://raw.githubusercontent.com/github/choosealicense.com/gh-pages/_licenses/mit.txt>
