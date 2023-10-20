#!/usr/bin/env bash

export PYTHONPATH= # <path_to_repo>
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Run experiments using training data
datasets=( "oa-mine" "ae-110k" )
model="gpt-3.5-turbo-0613" # "gpt-4-0613"
shots=( 10 )
percentages=( 0.2 1.0)
example_selectors=( "SemanticSimilarity" "MaxMarginalRelevance" "Random" )
no_example_values=3
schema_type="json_schema"

## First prompt
for dataset in "${datasets[@]}"
do
    for shot in "${shots[@]}"
    do
      echo "Running experiments for $dataset with $shot shots"
      for percentage in "${percentages[@]}"
      do
        for example_selector in "${example_selectors[@]}"
        do
          python prompts/2_schema-based_attribute_value_extraction/5_in-context_learning/in_context_learning_chatgpt_with_example_values.py --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector --with_containment False --with_validation_error_handling False --schema_type $schema_type --no_example_values $no_example_values
          python prompts/2_schema-based_attribute_value_extraction/5_in-context_learning/in_context_multiple_attribute_values.py  --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector
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
#          python prompts/2_schema-based_attribute_value_extraction/5_in-context_learning/in_context_schema_description_with_example_values.py --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector --with_containment False --with_validation_error_handling False --schema_type $schema_type
#          python prompts/2_schema-based_attribute_value_extraction/5_in-context_learning/in_context_list.py  --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector
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
#          python prompts/2_schema-based_attribute_value_extraction/5_in-context_learning/in_context_schema_description_with_example_values.py --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector --with_containment False --with_validation_error_handling False --schema_type $schema_type
#          python prompts/2_schema-based_attribute_value_extraction/5_in-context_learning/in_context_list.py  --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector
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
#          python prompts/2_schema-based_attribute_value_extraction/5_in-context_learning/in_context_schema_description_with_example_values.py --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector --with_containment False --with_validation_error_handling False --schema_type $schema_type
#          python prompts/2_schema-based_attribute_value_extraction/5_in-context_learning/in_context_list.py  --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector
#        done
#      done
#    done
#done