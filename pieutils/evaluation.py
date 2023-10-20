import json
from collections import defaultdict
from itertools import product

# Constants for evaluation dictionary keys
from pieutils import combine_example

NN = 'nn'
NV = 'nv'
VN = 'vn'
VC = 'vc'
VW = 'vw'


def calculate_evaluation_metrics_multiple_attributes(targets, preds, categories, unique_category,
                                                     unique_attribute):
    """Calculate evaluation metrics (nn, nv, vn, vc, vw) for a specific attribute-category combination in multi attribute setting.

    Args:
        targets (list): List of target values.
        preds (list): List of predicted values.
        categories (list): List of categories.
        attributes (list): List of attributes.
        unique_category: The unique category to evaluate.
        unique_attribute: The unique attribute to evaluate.

    Returns:
        dict: A dictionary containing the counts of each evaluation metric.
    """
    eval_dict = {
        NN: 0,
        NV: 0,
        VN: 0,
        VC: 0,
        VW: 0
    }

    for target, pred, category in zip(targets, preds, categories):
        if unique_category != category or unique_attribute not in target:
            # Evaluate per attribute/category
            continue

        target_values = [value.strip() if value != "n/a" else None for value in target[unique_attribute]]
        try:
            #processed_attribute_name = unique_attribute.lower().replace(' ', '_')
            # print(processed_attribute_name)
            prediction = json.loads(pred)[unique_attribute] if unique_attribute in json.loads(
                pred) else None
            #print(prediction)
            prediction = prediction if prediction != "n/a" and prediction != "None" else None
        except:
            print('Not able to decode prediction: \n {}'.format(pred))
            prediction = None

        if target_values[0] is None and prediction is None:
            eval_dict[NN] += 1
        elif target_values[0] is None and prediction is not None:
            eval_dict[NV] += 1
        elif target_values[0] is not None and prediction is None:
            eval_dict[VN] += 1
        elif prediction in target_values:
            eval_dict[VC] += 1
        else:
            eval_dict[VW] += 1

    return eval_dict


def calculate_recall_precision_f1_multiple_attributes(targets, preds, categories, known_attributes):
    """Calculate recall, precision and f1 for the extractions."""
    unique_categories = list(set(categories))

    result_dict = defaultdict(dict)
    total_eval = defaultdict(int)

    for unique_category in unique_categories:
        for unique_attribute in known_attributes[unique_category]:

            eval_dict = calculate_evaluation_metrics_multiple_attributes(targets, preds, categories, unique_category,
                                                     unique_attribute)

            if eval_dict[NV] + eval_dict[VC] + eval_dict[VW] + eval_dict[VN] == 0:
                # If there are no values to evaluate, skip this attribute-category combination.
                continue

            precision, recall, f1 = calculate_precision_recall_f1(eval_dict)

            total_eval = update_total_evaluation_metrics(total_eval, eval_dict)

            result_dict[f'{unique_attribute}_{unique_category}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

    macro_precision, macro_recall, macro_f1 = calculate_macro_scores(result_dict)
    micro_precision, micro_recall, micro_f1 = calculate_micro_scores(total_eval)

    result_dict['macro'] = {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }
    result_dict['micro'] = {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1
    }

    return result_dict


def calculate_recall_precision_f1(targets, preds, categories, attributes):
    """Calculate recall, precision and f1 for the extractions.

    Args:
        targets (list): List of target values.
        preds (list): List of predicted values.
        categories (list): List of categories.
        attributes (list): List of attributes.

    Returns:
        dict: A dictionary containing recall, precision, and f1 scores for each attribute-category combination,
              as well as macro and micro scores.
    """
    result_dict = defaultdict(dict)
    total_eval = defaultdict(int)

    for unique_category, unique_attribute in product(set(categories), set(attributes)):
        eval_dict = calculate_evaluation_metrics(targets, preds, categories, attributes, unique_category, unique_attribute)

        if eval_dict[NV] + eval_dict[VC] + eval_dict[VW] + eval_dict[VN] == 0:
            # If there are no values to evaluate, skip this attribute-category combination.
            continue
        precision, recall, f1 = calculate_precision_recall_f1(eval_dict)

        total_eval = update_total_evaluation_metrics(total_eval, eval_dict)

        result_dict[f'{unique_attribute}_{unique_category}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    macro_precision, macro_recall, macro_f1 = calculate_macro_scores(result_dict)
    micro_precision, micro_recall, micro_f1 = calculate_micro_scores(total_eval)

    result_dict['macro'] = {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }
    result_dict['micro'] = {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1
    }

    return result_dict

def calculate_evaluation_metrics(targets, preds, categories, attributes, unique_category, unique_attribute):
    """Calculate evaluation metrics (nn, nv, vn, vc, vw) for a specific attribute-category combination.

    Args:
        targets (list): List of target values.
        preds (list): List of predicted values.
        categories (list): List of categories.
        attributes (list): List of attributes.
        unique_category: The unique category to evaluate.
        unique_attribute: The unique attribute to evaluate.

    Returns:
        dict: A dictionary containing the counts of each evaluation metric.
    """
    eval_dict = {
        NN: 0,
        NV: 0,
        VN: 0,
        VC: 0,
        VW: 0
    }

    for target, pred, category, attribute in zip(targets, preds, categories, attributes):
        if unique_attribute != attribute or unique_category != category:
            continue

        target_values = [value.strip() if value != "n/a" else None for value in target]
        prediction = pred if pred != "n/a" else None

        if target_values[0] is None and prediction is None:
            eval_dict[NN] += 1
        elif target_values[0] is None and prediction is not None:
            eval_dict[NV] += 1
        elif target_values[0] is not None and prediction is None:
            eval_dict[VN] += 1
        elif prediction in target_values:
            eval_dict[VC] += 1
        else:
            eval_dict[VW] += 1

    return eval_dict

def calculate_precision_recall_f1(eval_dict):
    """Calculate precision, recall, and f1 scores based on the evaluation metrics.

    Args:
        eval_dict (dict): A dictionary containing the counts of each evaluation metric.

    Returns:
        tuple: A tuple containing precision, recall, and f1 scores.
    """
    precision = round((eval_dict[VC] / (eval_dict[NV] + eval_dict[VC] + eval_dict[VW])) * 100, 2) if (eval_dict[NV] + eval_dict[VC] + eval_dict[VW]) > 0 else 0
    recall = round((eval_dict[VC] / (eval_dict[VN] + eval_dict[VC] + eval_dict[VW])) * 100, 2) if (eval_dict[VN] + eval_dict[VC] + eval_dict[VW]) > 0 else 0
    f1 = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) > 0 else 0

    return precision, recall, f1

def update_total_evaluation_metrics(total_eval, eval_dict):
    """Update the total counts of evaluation metrics.

    Args:
        total_eval (dict): A dictionary containing the total counts of each evaluation metric.
        eval_dict (dict): A dictionary containing the counts of each evaluation metric.

    Returns:
        dict: The updated total_eval dictionary.
    """
    for metric in eval_dict:
        total_eval[metric] += eval_dict[metric]
    return total_eval

def calculate_macro_scores(result_dict):
    """Calculate macro scores (precision, recall, and f1) based on the result dictionary.

    Args:
        result_dict (dict): The result dictionary containing attribute-category specific evaluation metrics.
        attributes (list): List of attributes.
        categories (list): List of categories.

    Returns:
        tuple: A tuple containing macro precision, recall, and f1 scores.
    """
    precision_scores = [result['precision'] for result in result_dict.values()]
    macro_precision = round(sum(precision_scores) / len(precision_scores), 2)

    recall_scores = [result['recall'] for result in result_dict.values()]
    macro_recall = round(sum(recall_scores) / len(recall_scores), 2)

    f1_scores = [result['f1'] for result in result_dict.values()]
    macro_f1 = round(sum(f1_scores) / len(f1_scores), 2)

    return macro_precision, macro_recall, macro_f1

def calculate_micro_scores(total_eval):
    """Calculate micro scores (precision, recall, and f1) based on the total evaluation counts.

    Args:
        total_eval (dict): A dictionary containing the total counts of each evaluation metric.

    Returns:
        tuple: A tuple containing micro precision, recall, and f1 scores.
    """
    micro_precision = round((total_eval[VC] / (total_eval[NV] + total_eval[VC] + total_eval[VW])) * 100, 2) if (total_eval[NV] + total_eval[VC] + total_eval[VW]) > 0 else 0
    micro_recall = round((total_eval[VC] / (total_eval[VN] + total_eval[VC] + total_eval[VW])) * 100, 2) if (total_eval[VN] + total_eval[VC] + total_eval[VW]) > 0 else 0
    micro_f1 = round(2 * micro_precision * micro_recall / (micro_precision + micro_recall), 2) if (micro_precision + micro_recall) > 0 else 0

    return micro_precision, micro_recall, micro_f1


def evaluate_predictions(preds, task_dict, multiple_attribute=True):
    """Evaluate the task dictionary based on the predictions and return the evaluation metrics."""
    if multiple_attribute:
        targets = [example['target_scores'] for example in task_dict['examples']]

        categories = [example['category'] for example in task_dict['examples']]
        postprocessed_preds = [pred.json() if pred is not None else '' for pred in preds]

        task_dict['examples'] = [combine_example(example, pred, post_pred)
                                 for example, pred, post_pred in
                                 zip(task_dict['examples'], postprocessed_preds, postprocessed_preds)]

        results = calculate_recall_precision_f1_multiple_attributes(targets, postprocessed_preds,
                                                                                 categories,
                                                                                 task_dict['known_attributes'])

    else:
        targets = [example['target_scores'] for example in task_dict['examples']]

        categories = [example['category'] for example in task_dict['examples']]
        attributes = [example['attribute'] for example in task_dict['examples']]

        # postprocessed_preds = [pred.replace('\n','').replace('Answer: ', '').replace('AI: ', '').strip() for pred in preds]
        postprocessed_preds = [pred.split(':')[-1].split('\n')[0].strip() for pred, attribute in zip(preds, attributes)]
        postprocessed_preds = ['' if pred is None else pred for pred in postprocessed_preds]

        task_dict['examples'] = [combine_example(example, pred, post_pred)
                                 for example, pred, post_pred in zip(task_dict['examples'], preds, postprocessed_preds)]

        results = calculate_recall_precision_f1(targets, postprocessed_preds, categories, attributes)

    print(f'Task: {task_dict["task_name"]} on dataset: {task_dict["dataset_name"]}')
    print(results['micro'])
    print(f"{results['micro']['micro_precision']}\t{results['micro']['micro_recall']}\t{results['micro']['micro_f1']}")
    #print(results)

    return results
