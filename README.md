# VADER: Visually Aimed Defeasible Reasoning

## Overview

This project contains an official implementation of the article "VADER: Visually Aimed Defeasible Reasoning". The repository includes evaluation code for vision-language models on multiple benchmarks including SEED-Bench and SugarCrepe.

## Repository Structure

```
.
├── SEED-Bench/              # SEED-Bench evaluation implementation
│   ├── eval.py             # Main evaluation script for SEED-Bench
│   ├── BLIP2_eval.py       # BLIP2 model evaluation
│   ├── InstructBlip_eval_rephrasing.py  # InstructBLIP evaluation with rephrasing
│   ├── evaluator_strategies/  # Model evaluation strategies
│   └── data/               # Dataset location
├── sugar-crepe/            # SugarCrepe benchmark evaluation
│   ├── main_eval.py        # Main evaluation script for SugarCrepe
│   └── data/               # SugarCrepe dataset
├── scripts/                # Helper scripts for running evaluations
└── data/                   # Additional data files

```

## Installation

### Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yaml
conda activate lama
```

### LLaMA2 Setup (Optional)

For running LLaMA2 models using the `transformers` package from Hugging Face, you will need to create a read token for the LLaMA resource. More information can be found in the [official blog post](https://huggingface.co/blog/llama2#using-transformers).

## Quickstart

### SEED-Bench Evaluation

SEED-Bench is a benchmark for evaluating multimodal LLMs with generative comprehension across 12 evaluation dimensions. For detailed information, see [SEED-Bench/README.md](SEED-Bench/README.md).

#### Basic Usage

Evaluate InstructBLIP on SEED-Bench:

```bash
cd SEED-Bench
python eval.py --model instruct_blip --anno_path SEED-Bench.json --output-dir results
```

#### BLIP2 Evaluation

Evaluate BLIP2 model on a specific question type:

```bash
cd SEED-Bench
python BLIP2_eval.py --question_type_id 1 --anno_path Image_questions.json --output_dir results
```

Question type IDs:
- 1: Scene Understanding
- 2: Instance Identity
- 3: Instance Attributes
- 4: Instance Location
- 5: Instances Counting
- 6: Spatial Relation
- 7: Instance Interaction
- 8: Visual Reasoning
- 9: Text Understanding
- 10: Action Recognition
- 11: Action Prediction
- 12: Procedure Understanding

#### InstructBLIP Evaluation with Rephrasing

```bash
cd SEED-Bench
python InstructBlip_eval_rephrasing.py --question_type_id 1 --output_dir results
```

### SugarCrepe Evaluation

SugarCrepe is a benchmark for faithful vision-language compositionality evaluation. For detailed information, see [sugar-crepe/README.md](sugar-crepe/README.md).

**Note**: The main_eval.py script in this repository has been modified to use BLIP2 models instead of the original CLIP models used in the SugarCrepe paper.

#### Basic Usage

```bash
cd sugar-crepe
python main_eval.py \
    --output ./output \
    --coco_image_root ./data/coco/images/val2017/ \
    --data_root ./data/
```

The script evaluates BLIP2 image-text matching models on the SugarCrepe benchmark.

## Data Preparation

### SEED-Bench

Download the SEED-Bench dataset from the [HuggingFace repository](https://huggingface.co/datasets/AILab-CVC/SEED-Bench). For detailed data preparation instructions, including video datasets, see [SEED-Bench/DATASET.md](SEED-Bench/DATASET.md).

After downloading, update the data directory paths in `SEED-Bench/eval.py`:
- `cc3m_dir`: Root directory for evaluation dimensions 1-9 (images)
- `dimension10_dir`: Something-Something v2 videos
- `dimension11_dir`: Epic-Kitchen 100 videos
- `dimension12_dir`: Breakfast dataset videos

### SugarCrepe

Download the COCO-2017 validation set from the [official website](https://cocodataset.org/#download) and extract it to `sugar-crepe/data/coco/images/val2017/`.

## Experiments

### Prompt Engineering Notes

According to experimental observations, certain prompt formats may exhibit biases. For example, the following prompt structure:

```python
"Question: The following is a multiple choice question. Choose an answer by its number
     ...: \n 1.There is a tower in the image\n 2. There is a castle in the image.\n Answer:"
```

has been observed to consistently return option A. For more discussion on system prompts with LLaMA 2, see this [Hugging Face discussion](https://discuss.huggingface.co/t/trying-to-understand-system-prompts-with-llama-2-and-transformers-interface/59016).

## Evaluation Strategies

The repository implements multiple evaluation strategies for different models:

- **BLIP2Models**: Answer ranking by concatenation and loss computation
- **InstructBlipModels**: Answer ranking with rephrasing strategies

These strategies are located in `SEED-Bench/evaluator_strategies/` and can be extended for custom models.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Citation

If you use this code in your research, please cite the VADER paper (citation details to be added upon publication).

## Acknowledgments

This repository builds upon:
- [SEED-Bench](https://arxiv.org/abs/2307.16125): Benchmarking Multimodal LLMs
- [SugarCrepe](https://arxiv.org/abs/2306.14610): Vision-Language Compositionality Evaluation
- [LAVIS](https://github.com/salesforce/LAVIS): Vision-Language Models Framework
