import gzip
import json

from error_analysis import error_analysis


def attribute_analysis(paths_to_task_dicts):
    # Compare the results of multiple runs of the same task.
    results_per_attribute = {}

    for path_task_dict in paths_to_task_dicts:
        with gzip.open(path_task_dict, 'r') as f:
            task_dict = json.load(f)
            print(path_task_dict)
            for attribute in task_dict['results']:
                if attribute not in results_per_attribute:
                    results_per_attribute[attribute] = []
                results_per_attribute[attribute].append(task_dict['results'][attribute])

    #for attribute in results_per_attribute:
    #    print(attribute)
    #    print(results_per_attribute[attribute])
    #    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    #Run Error analysis for attribute Specialty_Coffee
    for path_task_dict in paths_to_task_dicts:
        error_analysis(path_task_dict, 'Specialty_Coffee')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

if "__main__" == __name__:
    paths_to_task_dicts = [ "prompts/runs/attribute_analysis/task_run_chatmultiple_attribute_values-great_ae-110k_gpt-3.5-turbo-0613_2023-08-26_14-20-18.gz",
                            "prompts/runs/attribute_analysis/task_run_chatchatgpt_description_with_all_example_values_10_examples_delimiter_ae-110k_gpt-3.5-turbo-0613_2023-09-11_14-55-30.gz",
                            "prompts/runs/attribute_analysis/task_run_chatchatgpt_description_with_example_values_in_context_learning_ae_110k_gpt_3_5_turbo_0613_3_SemanticSimilarityDifferentAttributeValues_0_2_ae-110k_gpt-3.5-turbo-0613_2023-09-13_14-54-34.gz"]
    attribute_analysis(paths_to_task_dicts)