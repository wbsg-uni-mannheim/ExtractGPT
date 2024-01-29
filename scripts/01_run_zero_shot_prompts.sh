#!/usr/bin/env bash

export PYTHONPATH= # <path_to_repo>
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run zero-shot experiments
datasets=( "oa-mine" "ae-110k" )
models=( "gpt-3.5-turbo-1106"  ) # "gpt-4-0613"
schema_types=( "textual" "json_schema" "compact" )
train_percentage=0.2 # 1.0

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
      python prompts/1_zero_shot_list/zero_shot_list.py --dataset $dataset --model $model
      for schema_type in "${schema_types[@]}"
      do
        echo "Running experiments for $dataset, $model and $schema_type"
        python prompts/2_zero_shot_schema/zero_shot_schema_description_with_example_values.py --dataset ${dataset} --model $model --schema_type $schema_type --train_percentage $train_percentage
        python prompts/2_zero_shot_schema/zero_shot_schema_description.py --dataset ${dataset} --model $model --schema_type $schema_type --verbose True
      done
    done
done

#model=stabilityai/StableBeluga-7B
#model_name=stable_beluga_7B
#
## Run Stable Beluga experiments for all datasets
#for dataset in "${datasets[@]}"
#do
#    python prompts/1_zero_shot_list/zero_shot_list.py --dataset $dataset --model $model --train_percentage $train_percentage > logs/${dataset}/zero_shot_list_${dataset}_${model_name}_${train_percentage}.log &
#    for schema_type in "${schema_types[@]}"
#    do
#      echo "Running experiments for $dataset,  $model and $schema_type"
#      python prompts/2_zero_shot_schema/zero_shot_schema_description_with_example_values.py --dataset ${dataset} --model $model --train_percentage $train_percentage --schema_type $schema_type > logs/${dataset}/zero_shot_schema_description_with_example_values_${dataset}_${model_name}_${train_percentage}_${schema_type}.log
#      python prompts/2_zero_shot_schema/zero_shot_schema_description.py --dataset ${dataset} --model $model --schema_type $schema_type --verbose True  > logs/${dataset}/zero_shot_schema_description_${dataset}_${model_name}_${schema_type}.log
#    done
#done
#
#model=stabilityai/StableBeluga2
#model_name=stable_beluga_2
#
## Run Stable Beluga experiments for all datasets
#for dataset in "${datasets[@]}"
#do
#    python prompts/1_zero_shot_list/zero_shot_list.py --dataset $dataset --model $model > logs/${dataset}/zero_shot_list_${dataset}_${model_name}.log
#    for schema_type in "${schema_types[@]}"
#    do
#      echo "Running experiments for $dataset,  $model and $schema_type"
#      python prompts/2_zero_shot_schema/zero_shot_schema_description.py --dataset ${dataset} --model $model --schema_type $schema_type --verbose True  > logs/${dataset}/zero_shot_schema_description_${dataset}_${model_name}_${schema_type}.log
#      python prompts/2_zero_shot_schema/zero_shot_schema_description_with_example_values.py --dataset ${dataset} --model $model --train_percentage $train_percentage --schema_type $schema_type > logs/${dataset}/zero_shot_schema_description_with_example_values_${dataset}_${model_name}_${train_percentage}_${schema_type}.log
#    done
#done
#
#
#model=upstage/SOLAR-0-70b-16bit
#model_name=upstage_SOLAR-0-70b-16bit
#
## Run Solar experiments for all datasets
#for dataset in "${datasets[@]}"
#do
#    python prompts/1_zero_shot_list/zero_shot_list.py --dataset $dataset --model $model > logs/${dataset}/zero_shot_list_${dataset}_${model_name}.log
#    for schema_type in "${schema_types[@]}"
#    do
#      echo "Running experiments for $dataset,  $model and $schema_type"
#      python prompts/2_zero_shot_schema/zero_shot_schema_description.py --dataset ${dataset} --model $model --schema_type $schema_type --verbose True  > logs/${dataset}/zero_shot_schema_description_${dataset}_${model_name}_${schema_type}.log
#      python prompts/2_zero_shot_schema/zero_shot_schema_description_with_example_values.py --dataset ${dataset} --model $model --train_percentage $train_percentage --schema_type $schema_type --verbose True > logs/${dataset}/zero_shot_schema_description_with_example_values_${dataset}_${model_name}_${train_percentage}_${schema_type}.log
#    done
#done