#!/usr/bin/env bash

export PYTHONPATH=/ceph/alebrink/01_projects/ExtractGPT/
export CUDA_VISIBLE_DEVICES=0,2,4,7
# Run experiments using training data
datasets=( "ae-110k" )
model="gpt-4-1106-preview" # "gpt-4-0613" ,
shots=( 10 )
percentages=( 0.2 1.0 )
example_selectors=( "SemanticSimilarity" )
no_example_values=5
schema_type="json_schema"


model=meta-llama/Meta-Llama-3-70B-Instruct
model_name=meta-llama_Meta-Llama-3-70B-Instruct

#for dataset in "${datasets[@]}"
#do
#    for shot in "${shots[@]}"
#    do
#      echo "Running experiments for $dataset with $shot shots"
#      for percentage in "${percentages[@]}"
#      do
#        for example_selector in "${example_selectors[@]}"
#        do
##          python prompts/5_in-context_learning/in_context_schema_description_with_example_values.py --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector --with_containment False --with_validation_error_handling False --schema_type $schema_type
#          python prompts/5_in-context_learning/in_context_list.py  --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector --verbose True > logs/${dataset}/in_context_list_${dataset}_${model_name}_shots_${shots}_${train_percentage}_${example_selector}.log
#        done
#      done
#    done
#done

schema_types=( "json_schema" "compact" )
train_percentage=0.2

#
## Run Meta-Llama-3-70B-Instruct experiments for all datasets
for dataset in "${datasets[@]}"
do
    #python prompts/1_zero_shot_list/zero_shot_list.py --dataset $dataset --model $model --verbose True > logs/${dataset}/zero_shot_list_${dataset}_${model_name}.log

    for schema_type in "${schema_types[@]}"
    do
      echo "Running experiments for $dataset,  $model and $schema_type"
      python prompts/2_zero_shot_schema/zero_shot_schema_description_with_example_values.py --dataset ${dataset} --model $model --schema_type $schema_type --verbose True > logs/${dataset}/zero_shot_schema_description_with_example_values_${dataset}_${model_name}_${train_percentage}_${schema_type}.log
      python prompts/2_zero_shot_schema/zero_shot_schema_description.py --dataset ${dataset} --model $model --schema_type $schema_type --verbose True  > logs/${dataset}/zero_shot_schema_description_${dataset}_${model_name}_${schema_type}.log

      #python prompts/2_zero_shot_schema/zero_shot_schema_description_with_example_values.py --dataset ${dataset} --model $model --schema_type $schema_type --verbose True
      #python prompts/2_zero_shot_schema/zero_shot_schema_description.py --dataset ${dataset} --model $model --schema_type $schema_type --verbose True
    done
done
