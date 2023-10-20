import gzip
import json

from pieutils import combine_example
from pieutils.evaluation import calculate_recall_precision_f1_multiple_attributes
from pieutils.preprocessing import load_known_attribute_values


def error_analysis(path_task_dict, attribute_category):
    with gzip.open(path_task_dict, 'r') as f:
        task_dict = json.load(f)

    targets = [example['target_scores'] for example in task_dict['examples']]
    preds = [example['pred'] for example in task_dict['examples']]
    categories = [example['category'] for example in task_dict['examples']]
    postprocessed_preds = [pred if pred is not None else '' for pred in preds]
    # postprocessed_preds = [pred.split(':')[-1].split('\n')[0].strip() for pred in preds]
    # postprocessed_preds = ['' if pred is None else pred for pred in postprocessed_preds]

    #print(postprocessed_preds)

    task_dict['examples'] = [combine_example(example, pred, post_pred)
                             for example, pred, post_pred in
                             zip(task_dict['examples'], postprocessed_preds, postprocessed_preds)]

    task_dict['results'] = calculate_recall_precision_f1_multiple_attributes(targets, postprocessed_preds, categories,
                                                                             task_dict['known_attributes'])

    #print(task_dict['results'])
    print(task_dict['results']['micro'])
    known_attributes_values = load_known_attribute_values(task_dict['dataset_name'], n_examples=9999)

    targets_with_unknown_attribute_values = []
    targets_with_known_attribute_values = []
    for example in task_dict['examples']:
        unknown_target = {}
        known_target = {}
        for target in example['target_scores']:
            if target not in known_attributes_values[example['category']]:
                unknown_target[target] = example['target_scores'][target]
            else:
                for value in example['target_scores'][target]:
                    if value not in known_attributes_values[example['category']][target]:
                        if target not in unknown_target:
                            unknown_target[target] = []
                        unknown_target[target].append(value)
                    else:
                        if target not in known_target:
                            known_target[target] = []
                        known_target[target].append(value)
        targets_with_unknown_attribute_values.append(unknown_target)
        targets_with_known_attribute_values.append(known_target)

    results_unseen_attributes = calculate_recall_precision_f1_multiple_attributes(targets_with_unknown_attribute_values, postprocessed_preds, categories,
                                                                             task_dict['known_attributes'])
    results_seen_attributes = calculate_recall_precision_f1_multiple_attributes(targets_with_known_attribute_values, postprocessed_preds, categories,
                                                                             task_dict['known_attributes'])

    #print(task_dict['results'])
    print('Unseen attributes:')
    print(results_unseen_attributes['micro'])
    print(f'{results_unseen_attributes["micro"]["micro_precision"]:.2f} \t {results_unseen_attributes["micro"]["micro_recall"]:.2f} \t {results_unseen_attributes["micro"]["micro_f1"]:.2f}')

    print('Seen attributes:')
    print(results_seen_attributes['micro'])
    print(f'{results_seen_attributes["micro"]["micro_precision"]:.2f} \t {results_seen_attributes["micro"]["micro_recall"]:.2f} \t {results_seen_attributes["micro"]["micro_f1"]:.2f}')

    # for example in task_dict['examples']:
    #     if attribute_category is not None and example['category'] not in attribute_category:
    #         continue
    #
    #     try:
    #         pred = json.loads(example['post_pred'])
    #         #pred = example['post_pred']
    #         for target in example['target_scores']:
    #             if attribute_category is not None and target not in attribute_category:
    #                 continue
    #             if target not in pred:
    #                 if 'I do not know.' in example['target_scores'][target]:
    #                     continue
    #                 print('Attribute: ' + target)
    #                 print('Input: ' + example['input'])
    #                 print('Description: ' + example['description'])
    #                 print('Features' + example['features'])
    #                 print(example['rest'])
    #                 print(example['target_scores'][target])
    #                 print(example['post_pred'])
    #                 print()
    #             else:
    #                 if pred[target] not in example['target_scores'][target]:
    #                     if (pred[target] == 'n/a' or pred[target] is None) and 'I do not know.' in example['target_scores'][target]:
    #                         continue
    #                     print('Attribute: ' + target)
    #                     print('Input: ' + example['input'])
    #                     print('Description: ' + example['description'])
    #                     print('Features: ' + example['features'])
    #                     print(example['rest'])
    #                     print(example['target_scores'][target])
    #                     print(target)
    #                     print(pred[target])
    #                     print()
    #     except:
    #         print(example['input'])
    #         #print(example['description'])
    #         #print(example['features'])
    #         #print(example['rest'])
    #         print(example['target_scores'])
    #         print(example['post_pred'])
    #         print()



if "__main__" == __name__:
    path_task_dict = "prompts/runs/task_run_chatchatgpt_description_with_example_values_in-context_learning_mave_gpt-3.5-turbo-16k-0613_2023-08-30_15-14-23.gz"
    error_analysis(path_task_dict)
