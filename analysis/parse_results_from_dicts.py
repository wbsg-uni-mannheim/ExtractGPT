
# Parse the results from the dictionaries and print the micro precision, recall and f1 scores.

import json
import gzip
import os

directory = 'prompts/runs/GPT-3_5_reruns/shots'

# iterate over all files in the directory
for filename in os.listdir(directory):
    # open the file
    with gzip.open(os.path.join(directory, filename), 'r') as file:
        # load the json file
        task_dict = json.loads(file.read())
        # print the micro precision, recall and f1 scores seperated by tab
        print('File:', filename.replace('task_run_chatchatgpt_description_with_example_values_in_context_learning_', 'json-val').replace('task_run_chatmultiple_attribute_values-great-incontext', 'list').split('gpt-3.5')[0])
        print('Micro Precision\tMicro Recall\tMicro F1')
        print(f'{task_dict["results"]["micro"]["micro_precision"]:.1f}\t{task_dict["results"]["micro"]["micro_recall"]:.1f}\t{task_dict["results"]["micro"]["micro_f1"]:.1f}')