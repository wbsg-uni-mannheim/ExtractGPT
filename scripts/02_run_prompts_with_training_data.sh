#!/usr/bin/env bash

export PYTHONPATH=/ceph/alebrink/01_projects/ExtractGPT/
export CUDA_VISIBLE_DEVICES=0,2,4,7
# Run experiments using training data
datasets=( "oa-mine" "ae-110k" )
model="gpt-4-0125-preview" # "gpt-4-0613"
shots=( 10 )
percentages=( 1.0 0.2 )
example_selectors=( "SemanticSimilarity" )
no_example_values=5
schema_type="compact"


## First prompt
#for dataset in "${datasets[@]}"
#do
#    for shot in "${shots[@]}"
#    do
#      echo "Running experiments for $dataset with $shot shots"
#      for percentage in "${percentages[@]}"
#      do
#        for example_selector in "${example_selectors[@]}"
#        do
#          #python prompts/5_in-context_learning/in_context_schema_description_with_example_values.py --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector --with_containment False --with_validation_error_handling False --schema_type $schema_type --no_example_values $no_example_values
#          #python prompts/5_in-context_learning/in_context_multiple_attribute_values.py  --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector
#
#          # Ensemble Shuffle
#          #python prompts/5_in-context_learning/in_context_schema_description_with_example_values_ensemble_shuffle.py --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector --with_containment False --with_validation_error_handling False --schema_type $schema_type --no_example_values $no_example_values
#
#        done
#      done
#    done
#done

model=meta-llama/Meta-Llama-3-70B-Instruct
model_name=meta-llama_Meta-Llama-3-70B-Instruct

for percentage in "${percentages[@]}"
do
    for shot in "${shots[@]}"
    do
      for dataset in "${datasets[@]}"
      do
        echo "Running experiments for $dataset with $shot shots"
        for example_selector in "${example_selectors[@]}"
        do
          python prompts/5_in-context_learning/in_context_schema_description_with_example_values.py --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector --with_containment False --with_validation_error_handling False --schema_type $schema_type --verbose True
          #python prompts/5_in-context_learning/in_context_list.py  --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector --verbose True
        done
      done
    done
done


#model=stabilityai/StableBeluga-7B
#model_name=stable_beluga_7B
#
#for dataset in "${datasets[@]}"
#do
#    for shot in "${shots[@]}"
#    do
#      echo "Running experiments for $dataset with $shot shots"
#      for percentage in "${percentages[@]}"
#      do
#        for example_selector in "${example_selectors[@]}"
#        do
#          python prompts/5_in-context_learning/in_context_schema_description_with_example_values.py --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector --with_containment False --with_validation_error_handling False --schema_type $schema_type
#          python prompts/5_in-context_learning/in_context_list.py  --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector
#        done
#      done
#    done
#done
#
#
#model=stabilityai/StableBeluga2
#model_name=stable_beluga_2
#
#for dataset in "${datasets[@]}"
#do
#    for shot in "${shots[@]}"
#    do
#      echo "Running experiments for $dataset with $shot shots"
#      for percentage in "${percentages[@]}"
#      do
#        for example_selector in "${example_selectors[@]}"
#        do
#          python prompts/5_in-context_learning/in_context_schema_description_with_example_values.py --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector --with_containment False --with_validation_error_handling False --schema_type $schema_type
#          python prompts/5_in-context_learning/in_context_list.py  --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector
#        done
#      done
#    done
#done
#
#
#model=upstage/SOLAR-0-70b-16bit
#model_name=upstage_SOLAR-0-70b-16bit
#
#for dataset in "${datasets[@]}"
#do
#    for shot in "${shots[@]}"
#    do
#      echo "Running experiments for $dataset with $shot shots"
#      for percentage in "${percentages[@]}"
#      do
#        for example_selector in "${example_selectors[@]}"
#        do
#          python prompts/5_in-context_learning/in_context_schema_description_with_example_values.py --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector --with_containment False --with_validation_error_handling False --schema_type $schema_type
#          python prompts/5_in-context_learning/in_context_list.py  --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector
#        done
#      done
#    done
#done