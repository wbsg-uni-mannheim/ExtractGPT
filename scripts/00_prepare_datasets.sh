#!/usr/bin/env bash

export PYTHONPATH= # <path_to_repo>

# Prepare datasets OA-Mine and AE-110K
python preprocessing/prepare_datasets.py --dataset oa-mine
python preprocessing/prepare_datasets.py --dataset ae-110k