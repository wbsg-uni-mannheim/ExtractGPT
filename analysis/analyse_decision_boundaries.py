import gzip
import os
import json

result_path = 'prompts/runs/ExampleValueAnalysis'

# Load all files into a list
task_dicts = {}

for filename in os.listdir(result_path):
    with gzip.open(os.path.join(result_path, filename), 'r') as file:
        task_dict = json.loads(file.read())
        no_example_values = int(filename.split('example_values_')[1].split('_examples')[0])
        task_dicts[no_example_values] = task_dict

# Determine attributes
attributes = set()
for task_dict in task_dicts.values():
    for example in task_dict['examples']:
        category = example['category']
        for target in example['target_scores']:
            attributes.add(f'{category}-{target}')


# Build decision boundary records
record_id = 0
decision_boundary_records = []

for attribute in attributes:
    category, target = attribute.split('-')
    for current_no_example_values in range(1, len(task_dicts) +1):
        next_no_example_values = current_no_example_values + 1
        if next_no_example_values in task_dicts:
            current_task_dict = task_dicts[current_no_example_values]
            next_task_dict = task_dicts[next_no_example_values]
            for current_example, next_example in zip(current_task_dict['examples'], next_task_dict['examples']):
                # Check if prediction is correct
                print('HERE!')
                # Check is prediction for attribute in current example is correct
                current_prediction = json.loads(current_example['post_pred'])
                next_prediction = json.loads(next_example['post_pred'])
                if target in current_example['target_scores']:
                    eval_current_prediction = current_example['target_scores'][target] == current_prediction[target]
                    eval_next_prediction = next_example['target_scores'][target] == next_prediction[target]

                    if eval_current_prediction != eval_next_prediction:
                        # Attribute in Meta Model
                        current_meta_model_attribute = None
                        for meta_model_attribute in current_task_dict['meta_json_schema']['attributes']:
                            if meta_model_attribute['name'] == target:
                                current_meta_model_attribute = meta_model_attribute
                                break

                        next_meta_model_attribute = None
                        for meta_model_attribute in next_task_dict['meta_json_schema']['attributes']:
                            if meta_model_attribute['name'] == target:
                                next_meta_model_attribute = meta_model_attribute
                                break

                        decision_boundary_record = {
                            'record_id': record_id,
                            'category': category,
                            'attribute': target,
                            'input': current_example['input'],
                            'current_prediction': current_prediction,
                            'next_prediction': next_prediction,
                            'target': current_example['target_scores'][target],
                            'current_meta_model_attribute': current_meta_model_attribute,
                            'next_meta_model_attribute': next_meta_model_attribute
                        }
                        decision_boundary_records.append(decision_boundary_record)
                        record_id += 1

# Save decision boundary records to csv
import csv

csv_file = 'prompts/runs/ExampleValueAnalysis/decision_boundary_records.csv'
csv_columns = ['record_id', 'category', 'attribute', 'input', 'current_prediction', 'next_prediction', 'target', 'current_meta_model_attribute', 'next_meta_model_attribute']
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in decision_boundary_records:
            writer.writerow(data)
except IOError:
    print("I/O error")
