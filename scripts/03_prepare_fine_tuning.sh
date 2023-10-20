#!/usr/bin/env bash

export PYTHONPATH= # <path_to_repo>
#export CUDA_VISIBLE_DEVICES=0,1,2,3

datasets=( "oa-mine" "ae-110k" )

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
      python finetuning/1_zero_shot_list/ft_list.py --dataset ${dataset} --model $model
      python finetuning/2_zero_shot_schema/ft_schema_description_with_example_values.py --dataset ${dataset} --model $model
    done
done
