#!/usr/bin/env bash

export PYTHONPATH= # <path_to_repo>
#export CUDA_VISIBLE_DEVICES=0,1,2,3

datasets=( "oa-mine" "ae-110k" )
models=( "gpt-3.5-turbo-0613"  ) # "gpt-4-0613"
schema_types=( "json_schema" ) # "textual" "compact"
example_value_amounts=( 3 5 10 )
train_percentage=0.2

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
      for schema_type in "${schema_types[@]}"
      do
        for example_value_amount in "${example_value_amounts[@]}"
        do
          echo "Running experiments for $dataset, $model and $example_value_amount"
          python tasks/2_zero_shot_schema/zero_shot_schema_description_with_example_values.py --dataset ${dataset} --model $model --schema_type $schema_type --train_percentage $train_percentage --no_example_values $example_value_amount
        done
      done
    done
done
