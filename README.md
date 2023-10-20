# Attribute Value Extraction using Large Language Models
This repository contains code and data for experiments on attribute value extraction using large language models.

## Requirements

We evaluate hosted LLMs, such as GPT-3.5 and GPT-4, as well as open-source LLMs based on LLAMA2 which can be run locally. 
Therefore, an OpenAI access tokens needs to be placed in a `.env` file at the root of the repository.
To obtain this OpenAI access token, users must [sign up](https://platform.openai.com/signup) for an OpenAI account.

## Installation

The codebase requires python 3.9 To install dependencies we suggest to use a conda virtual environment:

```
conda create -n avellms python=3.9
conda activate avellms
pip install -r requirements.txt
pip install pieutils/
```

## Dataset

The datasets subsets of OA-Mine and AE-110k can be found in the folder `data\processed_datasets`.

If you want to start from scratch, you can download the datasets and preprocess them yourself.
The dataset OA-Mine and AE-110k have to be added as raw data to the folder `data\raw`.
For OA-Mine the annoations can be downloaded [here](https://github.com/xinyangz/OAMine/tree/main/data).
For AE-110k the annoations can be downloaded [here](https://github.com/cubenlp/ACL19_Scaling_Up_Open_Tagging/blob/master/publish_data.txt).
The data is then preprocessed using the script `prepare_datasets.py` in the folder `preprocessing`.
The script can be run using the following command:

```
scripts\00_prepare_datasets.sh
```

## Prompts

![Prompt Designs](resources/zero_shot_prompt_designs.PNG)

We experiment with zero-shot prompts and prompts that use training data.
The prompts and the code to execute the prompts are defined in the folder `prompts`.
You can run the zero-shot prompts and prompts using training with the following scripts:

```
scripts/01_run_zero_shot_prompts.sh
scripts/02_run_prompts_with_training_data.sh
```

## Fine-tuning
The python scripts to prepare the data for the fine-tuning of the LLMs are located in the folder `finetuning`.
The fine-tuning can be run using the following scripts:

```
scripts/03_prepare_fine_tuning.sh
```

You find support for uploading the data to the OpenAI API and to start the finetuning in the folder `pieutils\open_ai_scripts`.
Further information can be found in [OpenAI's guides for fine-tuning](https://platform.openai.com/docs/guides/fine-tuning).
