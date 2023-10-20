import copy
import json
import os
import random

import click
from dotenv import load_dotenv
from tqdm import tqdm

from pieutils.config import RAW_DATA_SET_SOURCES, PROCESSED_DATASETS, MAVE_PROCECCESSED_DATASETS, \
    OPENTAG_PROCECCESSED_DATASETS
#from config import RAW_DATA_SET_SOURCES, PROCESSED_DATASETS, MAVE_PROCECCESSED_DATASETS, \
#    OPENTAG_PROCECCESSED_DATASETS


def update_task_dict_from_file_oa_mine(directory_path, task_dict,  first_n=50):
    task_dict['known_attributes'] = {}
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            category = ' '.join([token.capitalize() for token in filename.replace('.jsonl', '').split('_')])
            print('Load records for category: {}'.format(category))
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as f:
                for line in f.readlines()[:first_n]:
                    record = json.loads(line)
                    example = {'input': record['title'], 'category': category, 'target_scores': {}}
                    if category not in task_dict['known_attributes']:
                        task_dict['known_attributes'][category] = []
                    for attribute in record['entities']:
                        example['target_scores'][attribute['label']] = {attribute['value']: 1}
                        if attribute['label'] not in task_dict['known_attributes'][category]:
                            task_dict['known_attributes'][category].append(attribute['label'])

                    task_dict['examples'].append(example)
    return task_dict


def update_task_dict_from_test_set(task_dict):
    """Update task dict from test set."""
    directory_path = f'{PROCESSED_DATASETS}/{task_dict["dataset_name"]}/'
    file_path = os.path.join(directory_path, 'test.jsonl')
    task_dict['known_attributes'] = {}

    with open(file_path, 'r') as f:
        for line in f.readlines():
            record = json.loads(line)
            task_dict['examples'].append(record)
            if record['category'] not in task_dict['known_attributes']:
                task_dict['known_attributes'][record['category']] = []
            for attribute in record['target_scores']:
                if attribute not in task_dict['known_attributes'][record['category']]:
                    task_dict['known_attributes'][record['category']].append(attribute)

    return task_dict


def load_known_attribute_values_oa_mine(directory_path, known_attributes, skip_records=50):
    """Load 5 example attribute values from annotations."""
    known_attribute_values = {}
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            category = ' '.join([token.capitalize() for token in filename.replace('.jsonl', '').split('_')])
            if category not in known_attributes:
                continue
            print('Load attribute values for category: {}'.format(category))
            known_attribute_values[category] = {}
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as f:
                for line in f.readlines()[skip_records:]:
                    record = json.loads(line)
                    for attribute in record['entities']:
                        attribute_name = attribute['label']
                        if attribute_name in known_attributes[category]:
                            if attribute_name not in known_attribute_values[category]:
                                known_attribute_values[category][attribute_name] = []
                            if attribute['value'] not in known_attribute_values[category][attribute_name] \
                                    and len(known_attribute_values[category][attribute_name]) < 5:
                                known_attribute_values[category][attribute_name].append(attribute['value'])
    return known_attribute_values


def load_known_attribute_values(dataset_name, n_examples=5, consider_casing=False, train_percentage=1.0, test_set=False):
    """Loads known attribute values from train set."""
    known_attribute_values = {}
    if consider_casing:
        known_attribute_values_casing = {}
    directory_path = f'{PROCESSED_DATASETS}/{dataset_name}/'
    if test_set:
        file_path = os.path.join(directory_path, f'test.jsonl')
    else:
        if train_percentage < 1.0:
            file_path = os.path.join(directory_path, f'train_{train_percentage}.jsonl')
        else:
            file_path = os.path.join(directory_path, 'train.jsonl')
    with open(file_path, 'r') as f:
        for line in f.readlines():
            record = json.loads(line)
            category = record['category']
            if category not in known_attribute_values:
                known_attribute_values[category] = {}
                if consider_casing:
                    known_attribute_values_casing[category] = {}
            for attribute in record['target_scores']:
                if attribute not in known_attribute_values[category]:
                    known_attribute_values[category][attribute] = []
                    if consider_casing:
                        known_attribute_values_casing[category][attribute] = []
                for value in record['target_scores'][attribute]:
                    if value not in known_attribute_values[category][attribute] and value != 'n/a' \
                            and len(known_attribute_values[category][attribute]) < n_examples:
                        if consider_casing:
                            if value.lower() not in known_attribute_values_casing[category][attribute]:
                                known_attribute_values[category][attribute].append(value)
                                known_attribute_values_casing[category][attribute].append(value.lower())
                        else:
                            known_attribute_values[category][attribute].append(value)

    return known_attribute_values


def load_known_attribute_values_oa_mine_seed(directory_path, known_attributes, first_n=10):
    """Deprecated method to load attribute values from seed files"""
    known_attribute_values = {}
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            category = ' '.join([token.capitalize() for token in filename.replace('.seed.jsonl', '').split('_')])
            if category not in known_attributes:
                continue
            print('Load attribute values for category: {}'.format(category))
            known_attribute_values[category] = {}
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as f:
                for line in f.readlines()[:first_n]:
                    record = json.loads(line)
                    attribute = ' '.join(record[0].split('_')).capitalize()
                    values = record[1]
                    known_attribute_values[category][attribute] = values
    return known_attribute_values


def save_train_test_splits(dataset_name, train_examples, test_examples):
    # Save train and test splits
    directory_path_preprocessed = f'{PROCESSED_DATASETS}/{dataset_name}/'
    if not os.path.exists(directory_path_preprocessed):
        os.makedirs(directory_path_preprocessed)

    # Save train split
    with open(os.path.join(directory_path_preprocessed, 'train.jsonl'), 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')

    # Save test split
    with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'w') as f:
        for example in test_examples:
            f.write(json.dumps(example) + '\n')

    print(f'Saved train and test splits for {dataset_name}.')


def preprocess_oa_mine(dataset_name='oa-mine'):
    """Preprocess oa_mine by changing the format of the data and creating train and test splits."""

    directory_path = RAW_DATA_SET_SOURCES[dataset_name]
    examples = []
    # Load records and create examples
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            category = ' '.join([token.capitalize() for token in filename.replace('.jsonl', '').split('_')])
            print('Load records for category: {}'.format(category))
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as f:
                for line in f.readlines():
                    record = json.loads(line)
                    example = {'input': record['title'], 'category': category, 'target_scores': {}}
                    for attribute in record['entities']:
                        example['target_scores'][attribute['label']] = {attribute['value']: 1}
                    examples.append(example)

    # Create train and test splits
    random.shuffle(examples)
    train_examples = []
    test_examples = []
    categories = list(set([example['category'] for example in examples]))
    for category in categories:
        category_examples = [example for example in examples if example['category'] == category]
        train_examples.extend(category_examples[:int(len(category_examples) * 0.75)])
        test_examples.extend(category_examples[int(len(category_examples) * 0.75):])

    # Save train and test splits
    save_train_test_splits(dataset_name, train_examples, test_examples)


def preprocess_mave(dataset_name='mave', categories=None):
    if categories is None:
        categories = ['Flash Memory Cards', 'Digital Cameras', 'Laptops', 'Watches', 'Mobile Phones']

    records_per_split = {'train': 300, 'test': 100}
    test_ids = []
    test_examples = []
    train_examples = []
    for split in ['test', 'train']:
        directory_path = f'{RAW_DATA_SET_SOURCES[dataset_name]}/{split}/00_All'
        examples = {}
        dict_count_records = {category: 0 for category in categories}
        positives = 0
        negatives = 0
        for filename in sorted(list(os.listdir(directory_path)), reverse=True):
            if '.jsonl' in filename:
                with open(f'{directory_path}/{filename}', 'r') as f:
                    for line in tqdm(f):
                        record = json.loads(line)
                        if record['category'] in categories:
                            if record['id'] not in examples:
                                if dict_count_records[record['category']] < records_per_split[split] \
                                        and record['id'] not in test_ids:
                                    dict_count_records[record['category']] += 1
                                    input = '\n'.join([paragraph['text'] for paragraph in record['paragraphs'] if paragraph['source'] == 'title'])
                                    description = '\n'.join([paragraph['text'] for paragraph in record['paragraphs'] if
                                                       paragraph['source'] == 'description'])
                                    features = '\n'.join([paragraph['text'] for paragraph in record['paragraphs'] if
                                                             paragraph['source'] == 'feature'])
                                    rest = '\n'.join([paragraph['text'] for paragraph in record['paragraphs'] if
                                                          paragraph['source'] not in ['title', 'description', 'feature']])
                                    examples[record['id']] = {'input': input, 'description': description,
                                                              'features': features, 'rest': rest,
                                                              'category': record['category'], 'target_scores': {}}
                                else:
                                    # We have enough examples for this category or the record is already contained in the train split
                                    continue

                            for attribute in record['attributes']:
                                if attribute['key'] not in examples[record['id']]['target_scores']:
                                    examples[record['id']]['target_scores'][attribute['key']] = {}

                                if len(attribute['evidences']) == 0:
                                    examples[record['id']]['target_scores'][attribute['key']]['n/a'] = 1
                                    negatives += 1
                                else:
                                    for evidence in attribute['evidences']:
                                        if evidence['value'] not in examples[record['id']]['target_scores'][attribute['key']]:
                                            examples[record['id']]['target_scores'][attribute['key']][evidence['value']] = 1
                                    positives += 1

        # Unpack examples
        if split == 'test':
            print(f'Test: Positives: {positives}, Negatives: {negatives}')
            test_ids = list(examples.keys())
            test_examples = list(examples.values())
        elif split == 'train':
            print(f'Train: Positives: {positives}, Negatives: {negatives}')
            train_examples = list(examples.values())
        # Examples per Category
        #for category in categories:
        #    print(f'Examples for category {category}: {len([example for example in examples if example["category"] == category])}')

    # Save train and test splits
    save_train_test_splits(dataset_name, train_examples, test_examples)

def preprocess_mave_V2(dataset_name='mave_v2', categories=None):
    if categories is None:
        categories = ['Flash Memory Cards', 'Digital Cameras', 'Laptops', 'Watches', 'Mobile Phones']

    records_per_split = {'train': 300, 'test': 100}
    test_ids = []
    test_examples = []
    train_examples = []
    for split in ['test', 'train']:
        directory_path = f'{RAW_DATA_SET_SOURCES[dataset_name]}/{split}/00_All'
        examples = {}
        dict_count_records = {category: 0 for category in categories}
        positives = 0
        negatives = 0
        for filename in sorted(list(os.listdir(directory_path)), reverse=True):
            if '.jsonl' in filename:
                with open(f'{directory_path}/{filename}', 'r') as f:
                    for line in tqdm(f):
                        record = json.loads(line)
                        if record['category'] in categories:
                            if record['id'] not in examples:
                                if dict_count_records[record['category']] < records_per_split[split] \
                                        and record['id'] not in test_ids:
                                    dict_count_records[record['category']] += 1
                                    input = '\n'.join([paragraph['text'] for paragraph in record['paragraphs'] if paragraph['source'] == 'title'])
                                    paragraphs = [paragraph['text'] for paragraph in record['paragraphs'] if paragraph['source'] != 'title']
                                    examples[record['id']] = {'input': input, 'paragraphs': paragraphs,
                                                              'category': record['category'], 'target_scores': {}}
                                else:
                                    # We have enough examples for this category or the record is already contained in the train split
                                    continue

                            for attribute in record['attributes']:
                                if attribute['key'] not in examples[record['id']]['target_scores']:
                                    examples[record['id']]['target_scores'][attribute['key']] = {}

                                if len(attribute['evidences']) == 0:
                                    examples[record['id']]['target_scores'][attribute['key']]['n/a'] = 1
                                    negatives += 1
                                else:
                                    for evidence in attribute['evidences']:
                                        if evidence['value'] not in examples[record['id']]['target_scores'][attribute['key']]:
                                            examples[record['id']]['target_scores'][attribute['key']][evidence['value']] = 1
                                    positives += 1

        # Unpack examples
        if split == 'test':
            print(f'Test: Positives: {positives}, Negatives: {negatives}')
            test_ids = list(examples.keys())
            test_examples = list(examples.values())
        elif split == 'train':
            print(f'Train: Positives: {positives}, Negatives: {negatives}')
            train_examples = list(examples.values())
        # Examples per Category
        #for category in categories:
        #    print(f'Examples for category {category}: {len([example for example in examples if example["category"] == category])}')

    # Save train and test splits
    save_train_test_splits(dataset_name, train_examples, test_examples)

def preprocess_ae_110k(dataset_name='ae-110k'):
    """Preprocess oa_mine by changing the format of the data and creating train and test splits."""

    directory_path = RAW_DATA_SET_SOURCES[dataset_name]
    records = {}
    categories = {'Type': [], 'Category': [], 'Item Type': []}
    # Load records and create examples
    file_path = os.path.join(directory_path, 'publish_data.txt')
    with open(file_path, 'r') as f:
        for line in f.read().splitlines():
            record = line.split('\u0001')
            if len(record) == 3:
                if record[0] not in records:
                    records[record[0]] = []
                attribute = ' '.join([value.capitalize() for value in record[1].strip().split(' ')])
                records[record[0]].append((attribute, record[2]))
                if attribute in categories and record[2] not in categories[attribute]:
                    categories[attribute].append(record[2])
            else:
                print('Record has wrong format: {}'.format(record))

    # Create examples
    examples = []
    for input in records:
        example = {'input': input, 'category': '', 'target_scores': {}}
        add_example = False
        category_identifiers = list(categories.keys())
        for attribute, value in records[input]:
            if attribute in category_identifiers and value != 'NULL':
                if example['category'] != '':
                    # Category is already assigned.
                    continue
                example['category'] = ' '.join([part.capitalize() for part in value.split(' ')])
                if 'Guitar' in example['category']:
                    example['category'] = 'Guitar'
                if 'Stove' in example['category']:
                    example['category'] = 'Stove'
                if 'Bikini' in example['category']:
                    example['category'] = 'Bikini'
                if 'One Piece' in example['category']:
                    example['category'] = 'One Piece Swimsuit'
                if 'Rod' in example['category']:
                    example['category'] = 'Fishing Rod'
                if 'Shorts' in example['category']:
                    example['category'] = 'Shorts'
                if 'Jerseys' in example['category']:
                    example['category'] = 'Shirts'
                add_example = True
            else:
                # Preprocess attribute
                if attribute == 'Description':
                    continue

                if attribute == 'Maerial':
                    attribute = 'Material'
                for i in range(9):
                    attribute = attribute.replace(str(i), '').strip()
                if attribute == 'Model Number/ Sku':
                    attribute = 'Model Number'
                if attribute == 'Features':
                    attribute = 'Feature'
                if attribute == 'Pattern Type':
                    attribute = 'Pattern'

                if attribute not in example['target_scores']:
                    example['target_scores'][attribute] = {}

                if value != 'NULL':
                    example['target_scores'][attribute][value] = 1
                else:
                    example['target_scores'][attribute]['n/a'] = 1

        if add_example and len(example['target_scores']) > 1:
            examples.append(example)

    # Group categories:
    swimwear = ['One Piece Swimsuit', 'Two Pieces', 'Briefs', 'Bikini']
    for example in examples:
        if example['category'] in swimwear:
            #example['target_scores']['Item Type'] = {example['category']: 1}
            example['category'] = 'Swimwear'

    optics = ['Monocular', 'Binoculars']
    for example in examples:
        if example['category'] in optics:
            #example['target_scores']['Item Type'] = {example['category']: 1}
            example['category'] = 'Optics & Binoculars'

    print('Number of examples: {}'.format(len(examples)))
    known_attributes_per_category = {}
    known_attribute_values_per_category = {}
    for example in examples:
        if example['category'] not in known_attributes_per_category:
            known_attributes_per_category[example['category']] = []
            known_attribute_values_per_category[example['category']] = {}
        for attribute in example['target_scores']:
            known_attributes_per_category[example['category']].append(attribute)
            if attribute not in known_attribute_values_per_category[example['category']]:
                known_attribute_values_per_category[example['category']][attribute] = set()
            for value in example['target_scores'][attribute]:
                known_attribute_values_per_category[example['category']][attribute].add(value)

    # Records per category
    records_per_category = {}
    for example in examples:
        if example['category'] not in records_per_category:
            records_per_category[example['category']] = []
        records_per_category[example['category']].append(example)

    print('______________________________________________________________')
    relevant_categories = [category for category in records_per_category if len(records_per_category[category]) > 10]
    relevant_categories_after_attribute_filter = {}
    for category in known_attributes_per_category:
        if category in relevant_categories:
            # Count number of attribute appearances
            attribute_counts = {}
            attribute_values = {}
            for attribute in known_attributes_per_category[category]:
                if known_attribute_values_per_category[category][attribute] == {'n/a'}:
                    continue
                known_attribute_values_per_category[category][attribute].discard('n/a')
                if len(known_attribute_values_per_category[category][attribute]) > 1:
                    if attribute not in attribute_counts:
                        attribute_counts[attribute] = 0
                        attribute_values[attribute] = set()
                    attribute_counts[attribute] += 1

            # Attributes that appear more than 5 times and have more than 1 value
            relevant_attributes = [attribute for attribute in attribute_counts if attribute_counts[attribute] > 5]
            if len(relevant_attributes) > 2:
                print('Category: {}, Known attributes: {}'.format(category, sorted(relevant_attributes)))
                relevant_categories_after_attribute_filter[category] = relevant_attributes

    # Filter examples based on relevant categories and attributes
    filtered_examples = []
    for example in examples:
        if example['category'] in relevant_categories_after_attribute_filter:
            filtered_example = {'input': example['input'], 'category': example['category'], 'target_scores': {}}
            for attribute in example['target_scores']:
                if attribute in relevant_categories_after_attribute_filter[example['category']]:
                    filtered_example['target_scores'][attribute] = example['target_scores'][attribute]
            if len(filtered_example['target_scores']) > 1:
                filtered_examples.append(filtered_example)

    # Records per category
    records_per_category = {}
    for example in filtered_examples:
        if example['category'] not in records_per_category:
            records_per_category[example['category']] = 0
        records_per_category[example['category']] += 1

    # Filter categories with less than 50 examples
    filtered_examples_second_time = []
    for example in filtered_examples:
        if example['category'] in records_per_category and records_per_category[example['category']] > 50:
            filtered_examples_second_time.append(example)

    # Subsample categories with more than 500 examples
    records_per_category = {}
    for example in filtered_examples_second_time:
        if example['category'] not in records_per_category:
            records_per_category[example['category']] = []
        records_per_category[example['category']] += [example]

    filtered_examples_third_time = []
    for category in records_per_category:
        if len(records_per_category[category]) > 400:
            random.shuffle(records_per_category[category])
            records_per_category[category] = records_per_category[category][:400]

        filtered_examples_third_time += records_per_category[category]

    print('Number of examples after filtering: {}'.format(len(filtered_examples_third_time)))

    # Create train and test splits
    random.shuffle(filtered_examples_third_time)
    train_examples = []
    test_examples = []
    categories = list(set([example['category'] for example in filtered_examples_third_time]))
    for category in categories:
        category_examples = [example for example in filtered_examples_third_time if example['category'] == category]
        train_examples.extend(category_examples[:int(len(category_examples) * 0.75)])
        test_examples.extend(category_examples[int(len(category_examples) * 0.75):])

    # Save train and test splits
    save_train_test_splits(dataset_name, train_examples, test_examples)


def convert_to_mave_dataset(dataset_name, percentage=1, skip_test=False):
    # Load dataset
    # Save train and test splits
    directory_path_preprocessed = f'{PROCESSED_DATASETS}/{dataset_name}/'

    if dataset_name == 'mave':
        # Read train split
        train_examples = []
        if percentage == 1:
            with open(os.path.join(directory_path_preprocessed, 'train.jsonl'), 'r') as f:
                for line in f.readlines():
                    example = json.loads(line)
                    train_examples.append(example['input'])
        else:
            with open(os.path.join(directory_path_preprocessed, f'train_{percentage}.jsonl'), 'r') as f:
                for line in f.readlines():
                    example = json.loads(line)
                    train_examples.append(example['input'])

        # Read test split
        test_examples = []
        if not skip_test:
            with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'r') as f:
                for line in f.readlines():
                    example = json.loads(line)
                    test_examples.append(example['input'])

        print('Number of train examples: {}'.format(len(train_examples)))
        print('Number of test examples: {}'.format(len(test_examples)))

        # Identify records in original splits
        print('Identifying records in original splits...')
        records = {'train': {'positives': [], 'negatives': []}, 'test': {'positives': [], 'negatives': []}}
        for split in ['test', 'train']:
            directory_path = f'{RAW_DATA_SET_SOURCES[dataset_name]}/{split}/00_All'
            for filename in sorted(list(os.listdir(directory_path)), reverse=True):
                if '.jsonl' in filename:
                    print('Processing file: {}'.format(filename))
                    record_type = filename.replace('.jsonl', '').replace('mave_', '')
                    with open(f'{directory_path}/{filename}', 'r') as f:
                        for line in tqdm(f.readlines()):
                            record = json.loads(line)
                            title = [paragraph['text'] for paragraph in record['paragraphs'] if paragraph['source'] == 'title'][0]
                            if title in train_examples or title in test_examples:
                                records[split][record_type].append(record)

    else:
        # Read train split
        train_examples = []
        if percentage == 1:
            with open(os.path.join(directory_path_preprocessed, 'train.jsonl'), 'r') as f:
                for line in f.readlines():
                    example = json.loads(line)
                    train_examples.append(example)
        else:
            with open(os.path.join(directory_path_preprocessed, f'train_{percentage}.jsonl'), 'r') as f:
                for line in f.readlines():
                    example = json.loads(line)
                    train_examples.append(example)

        # Read test split
        test_examples = []
        if not skip_test:
            with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'r') as f:
                for line in f.readlines():
                    example = json.loads(line)
                    test_examples.append(example)

        print('Number of train examples: {}'.format(len(train_examples)))
        print('Number of test examples: {}'.format(len(test_examples)))
        # Convert training and test records to mave format
        print('Converting training and test records to mave format...')
        records = {'train': {'positives': [], 'negatives': []}, 'test': {'positives': [], 'negatives': []}}

        # Record format: {'id': str, 'category': str, 'paragraphs': [{'text': str, 'source': str}],
        #                 'attributes': [{'key': str, 'evidences': ['value': str, 'pid': int, "begin": int, "end": int]}]}

        # Train records
        print('Converting train records...')
        current_id = 0
        for example in tqdm(train_examples):
            record = {'category': example['category'], 'paragraphs': [{'text': example['input'], 'source': 'title'}], 'attributes': []}
            for attribute, value in example['target_scores'].items():
                specific_record = copy.deepcopy(record)
                specific_record['id'] = str(current_id)

                target_value = list(value.keys())[0]
                if target_value != "n/a":
                    begin = example['input'].find(target_value)
                    end = begin + len(target_value)
                    specific_record['attributes'] = [{'key': attribute, 'evidences': [{'value': target_value, 'pid': 0, 'begin': begin, 'end': end}]}]
                    records['train']['positives'].append(specific_record)
                    current_id += 1
                else:
                    specific_record['attributes'] = [{'key': attribute, 'evidences': []}]
                    records['train']['negatives'].append(specific_record)
                    current_id += 1

        for example in tqdm(test_examples):
            record = {'category': example['category'], 'paragraphs': [{'text': example['input'], 'source': 'title'}],
                      'attributes': []}
            for attribute, value in example['target_scores'].items():
                specific_record = copy.deepcopy(record)
                specific_record['id'] = str(current_id)

                target_value = list(value.keys())[0]
                if target_value != "n/a":
                    begin = example['input'].find(target_value)
                    end = begin + len(target_value)
                    specific_record['attributes'] = [{'key': attribute, 'evidences': [
                        {'value': target_value, 'pid': 0, 'begin': begin, 'end': end}]}]
                    records['test']['positives'].append(specific_record)
                    current_id += 1
                else:
                    specific_record['attributes'] = [{'key': attribute, 'evidences': []}]
                    records['test']['negatives'].append(specific_record)
                    current_id += 1

    # Split train records into train and validation
    print('Splitting train records into train and validation...')
    random.shuffle(records['train']['positives'])
    random.shuffle(records['train']['negatives'])
    records['validation'] = {'positives': records['train']['positives'][:int(len(records['train']['positives']) * 0.1)],
                             'negatives': records['train']['negatives'][:int(len(records['train']['negatives']) * 0.1)]}
    records['train']['positives'] = records['train']['positives'][int(len(records['train']['positives']) * 0.1):]
    records['train']['negatives'] = records['train']['negatives'][int(len(records['train']['negatives']) * 0.1):]

    # Save records
    print('Saving records...')
    for split in ['test', 'validation', 'train']:
        if skip_test and split == 'test':
            continue
        for record_type in ['positives', 'negatives']:
            if percentage == 1:
                directory_path_preprocessed_mave = f'{MAVE_PROCECCESSED_DATASETS}/splits/PRODUCT/{split}/{dataset_name}'
            else:
                directory_path_preprocessed_mave = f'{MAVE_PROCECCESSED_DATASETS}/splits/PRODUCT/{split}/{dataset_name}_{percentage}'
            if not os.path.exists(directory_path_preprocessed_mave):
                os.makedirs(directory_path_preprocessed_mave)

            with open(f'{directory_path_preprocessed_mave}/mave_{record_type}.jsonl', 'w') as f:
                file_content = [json.dumps(record) for record in records[split][record_type]]
                f.write('\n'.join(file_content))

def convert_to_open_tag_format(dataset_name, percentage=1, skip_test=False):
    """Convert records to OpenTag format.
        Format 1: {"id": 19, "title": "热风2019年春季新款潮流时尚男士休闲皮鞋透气低跟豆豆鞋h40m9107", "attribute": "款式", "value": "豆豆鞋"}
        Format 2: "热风2019年春季新款潮流时尚男士休闲皮鞋透气低跟豆豆鞋h40m9107<$$$>款式<$$$>豆豆鞋<$$$>19" - Split by <$$$>
    """
    directory_path_preprocessed = f'{PROCESSED_DATASETS}/{dataset_name}/'
    # Read train split
    train_examples = []
    if percentage == 1:
        with open(os.path.join(directory_path_preprocessed, 'train.jsonl'), 'r') as f:
            for line in f.readlines():
                example = json.loads(line)
                train_examples.append(example)
    else:
        with open(os.path.join(directory_path_preprocessed, f'train_{percentage}.jsonl'), 'r') as f:
            for line in f.readlines():
                example = json.loads(line)
                train_examples.append(example)

    # Read test split
    test_examples = []
    if not skip_test:
        with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'r') as f:
            for line in f.readlines():
                example = json.loads(line)
                test_examples.append(example)

    print('Number of train examples: {}'.format(len(train_examples)))
    print('Number of test examples: {}'.format(len(test_examples)))
    # Convert training and test records to mave format
    print('Converting training and test records to mave format...')

    converted_train_examples = []
    converted_test_examples = []

    # Convert train examples
    record_id = 0
    for example in tqdm(train_examples):
        for attribute, value in example['target_scores'].items():
            for target_value in example['target_scores'][attribute]:
                for part in ['input', 'description', 'features', 'rest']:
                    if part in example and example[part] is not None:
                        if target_value != "n/a":
                            if target_value in example[part]:
                                record = {'id': record_id, 'title': example[part], 'attribute': '', 'value': '', 'category': example['category']}
                                record['attribute'] = attribute
                                record['value'] = target_value
                                converted_train_examples.append(record)
                                record_id += 1
                        else:
                            record = {'id': record_id, 'title': example[part], 'attribute': '', 'value': '', 'category': example['category']}
                            record['attribute'] = attribute
                            record['value'] = None
                            converted_train_examples.append(record)
                            record_id += 1

    # Convert test examples
    for example in tqdm(test_examples):
        for attribute, value in example['target_scores'].items():
            for target_value in example['target_scores'][attribute]:
                for part in ['input', 'description', 'features', 'rest']:
                    if part in example and example[part] is not None:
                        if target_value != "n/a":
                            if target_value in example[part]:
                                record = {'id': record_id, 'title': example[part], 'attribute': '', 'value': '', 'category': example['category']}
                                record['attribute'] = attribute
                                record['value'] = target_value
                                converted_test_examples.append(record)
                                record_id += 1
                        else:
                            record = {'id': record_id, 'title': example[part], 'attribute': '', 'value': '', 'category': example['category']}
                            record['attribute'] = attribute
                            record['value'] = None
                            converted_test_examples.append(record)
                            record_id += 1

    # Convert examples to second format
    print('Converting examples to second format...')
    converted_examples_format_2 = []
    random.shuffle(converted_train_examples)
    split_converted_train_examples = int(len(converted_train_examples) * 0.9)
    converted_train_examples_format_2 = converted_train_examples[:split_converted_train_examples]
    converted_valid_examples_format_2 = converted_train_examples[split_converted_train_examples:]
    for example in tqdm(converted_train_examples_format_2):
        example_format_2 = f'{example["title"]}<$$$>{example["category"]}-{example["attribute"]}<$$$>{example["value"]}<$$$>train'
        converted_examples_format_2.append(example_format_2)

    for example in tqdm(converted_valid_examples_format_2):
        example_format_2 = f'{example["title"]}<$$$>{example["category"]}-{example["attribute"]}<$$$>{example["value"]}<$$$>valid'
        converted_examples_format_2.append(example_format_2)

    for example in tqdm(converted_test_examples):
        example_format_2 = f'{example["title"]}<$$$>{example["category"]}-{example["attribute"]}<$$$>{example["value"]}<$$$>test'
        converted_examples_format_2.append(example_format_2)

    # Save records
    print('Saving records...')
    for split in ['test', 'train']:
        if skip_test and split == 'test':
            continue

        directoryname = f'{dataset_name}_{split}'
        if percentage != 1 and split == 'train':
            directoryname = f'{dataset_name}_{split}_{str(percentage).replace(".", "_")}'

        directory_path_preprocessed_opentag = f'{OPENTAG_PROCECCESSED_DATASETS}/{dataset_name}/{directoryname}'
        if not os.path.exists(directory_path_preprocessed_opentag):
            os.makedirs(directory_path_preprocessed_opentag)

        if split == 'test':
            with open(f'{directory_path_preprocessed_opentag}/test_sample.json', 'w', encoding='utf-8') as f:
                file_content = [json.dumps(record) for record in converted_test_examples]
                f.write('\n'.join(file_content))

        else:
            with open(f'{directory_path_preprocessed_opentag}/train_sample.json', 'w', encoding='utf-8') as f:
                file_content = [json.dumps(record) for record in converted_train_examples]
                f.write('\n'.join(file_content))

    directory_path_preprocessed_opentag = f'{OPENTAG_PROCECCESSED_DATASETS}/{dataset_name}'
    if not os.path.exists(directory_path_preprocessed_opentag):
        os.makedirs(directory_path_preprocessed_opentag)
    # Format 2
    with open(f'{directory_path_preprocessed_opentag}/{dataset_name}_{str(percentage).replace(".", "_")}.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(converted_examples_format_2))


def reduce_training_set_size(dataset_name, percentage):
    """Reduces the size of the training set of a dataset to the specified percentage of the original size."""
    directory_path_preprocessed = f'{PROCESSED_DATASETS}/{dataset_name}/'
    # Read train split
    train_examples = []
    with open(os.path.join(directory_path_preprocessed, 'train.jsonl'), 'r') as f:
        for line in f.readlines():
            example = json.loads(line)
            train_examples.append(example)

    # Stratified sampling per category
    categories = list(set([example['category'] for example in train_examples]))
    train_examples_reduced = []
    for category in categories:
        examples_category = [example for example in train_examples if example['category'] == category]
        train_examples_reduced += random.sample(examples_category, int(len(examples_category)*percentage))

    # Save reduced training set
    with open(os.path.join(directory_path_preprocessed, f'train_{percentage}.jsonl'), 'w') as f:
        for example in train_examples_reduced:
            f.write(json.dumps(example) + '\n')
