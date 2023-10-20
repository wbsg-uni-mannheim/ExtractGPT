import json
import os
import random

import click
import tiktoken
from dotenv import load_dotenv
from tqdm import tqdm

from config import PROCESSED_DATASETS


def mostly_numeric(s):
    # Count the number of numeric characters
    numeric_count = sum(c.isdigit() for c in s)

    # Check if more than half of the characters are numeric
    return numeric_count > len(s) / 2


def is_all_numeric(s):
    return s.isdigit()


# Load dataset

@click.command
@click.option('--dataset', default='mave', help='Dataset name')
def main(dataset):
    print(f'Dataset name: {dataset}')

    # Load dataset
    directory_path_preprocessed = f'{PROCESSED_DATASETS}/{dataset}/'

    loaded_datasets = {'train': [], 'test': []}
    with open(os.path.join(directory_path_preprocessed, 'train_0.2.jsonl'), 'r') as f:
        for line in tqdm(f.readlines()):
            loaded_datasets['train'].append(json.loads(line))

    with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'r') as f:
        for line in tqdm(f.readlines()):
            loaded_datasets['test'].append(json.loads(line))

    # Calculate statistics
    # Number of Products
    unique_categories = set()
    unique_attributes = set()
    unique_category_attribute_combinations = set()
    attributes_by_category = {'train': {}, 'test': {}}
    records_by_category = {'train': {}, 'test': {}}
    no_attributes_by_product = {'train': [], 'test': []}
    no_negatives = 0
    attribute_values_by_catgory_attribute = {'train': {}, 'test': {}}

    # Calculate no. tokens
    encoding = tiktoken.get_encoding("cl100k_base")

    num_title_tokens_by_product = []
    num_description_tokens_by_product = []
    num_feature_tokens_by_products = []
    num_rest_tokens_by_products = []

    # Iterate over all records
    for split, records in loaded_datasets.items():
        for record in records:
            unique_categories.add(record['category'])
            if record['category'] not in records_by_category[split]:
                records_by_category[split][record['category']] = []
            records_by_category[split][record['category']].append(record)
            no_attributes_by_product[split].append(len(record['target_scores']))
            for attribute, attribute_values in record['target_scores'].items():
                unique_category_attribute_combinations.add(f'{attribute}-{record["category"]}')
                unique_attributes.add(f'{attribute}')

                if record['category'] not in attributes_by_category[split]:
                    attributes_by_category[split][record['category']] = {}
                if attribute not in attributes_by_category[split][record['category']]:
                    attributes_by_category[split][record['category']][attribute] = {'positive': 0, 'negative': 0 }

                #attributes_by_category[record['category']][attribute] += 1
                if 'n/a' in attribute_values.keys():
                    attributes_by_category[split][record['category']][attribute]['negative'] += 1
                else:
                    attributes_by_category[split][record['category']][attribute]['positive'] += 1

                if record['category'] not in attribute_values_by_catgory_attribute[split]:
                    attribute_values_by_catgory_attribute[split][record['category']] = {}

                if attribute not in attribute_values_by_catgory_attribute[split][record['category']]:
                    attribute_values_by_catgory_attribute[split][record['category']][attribute] = set()

                if 'n/a' in attribute_values.keys():
                    no_negatives += 1

                for attribute_value in attribute_values.keys():
                    if attribute_value != 'n/a':
                        attribute_values_by_catgory_attribute[split][record['category']][attribute].add(attribute_value)

            # Calculate no. tokens
            num_title_tokens_by_product.append(len(encoding.encode(record['input'])))
            if 'description' in record:
                num_description_tokens_by_product.append(len(encoding.encode(record['description'])))
            if 'features' in record:
                num_feature_tokens_by_products.append(len(encoding.encode(record['features'])))
            if 'rest' in record:
                num_rest_tokens_by_products.append(len(encoding.encode(record['rest'])))


    print(f'No. Product Offers: \t {len(loaded_datasets["train"]) + len(loaded_datasets["test"])}')

    print(f'No. Product Offers Train: \t {len(loaded_datasets["train"])}')
    print(f'No. Product Offers Test: \t {len(loaded_datasets["test"])}')

    print(f'No. Attributes: \t  {len(unique_attributes)}')
    print(f'No. Categories: \t {len(unique_categories)}')
    print(f'No. Category & Attribute Combinations: \t {len(unique_category_attribute_combinations)}')

    # Number of annotations
    print(f'Number of annotations: \t {sum(no_attributes_by_product["train"]) + sum(no_attributes_by_product["test"])}')
    # Number of annotations
    print(f'Number of train annotations: \t {sum(no_attributes_by_product["train"])}')
    # Number of annotations
    print(f'Number of test annotations: \t {sum(no_attributes_by_product["test"])}')
    # Number of positive annotations
    print(f'Number of positive annotations: \t {sum(no_attributes_by_product["train"]) + sum(no_attributes_by_product["test"]) - no_negatives}')
    # Number of negative annotations
    print(f'Number of negative annotations: \t {no_negatives}')

    print('')
    print(f'Categories: \t {", ".join(list(unique_categories))}')
    #print(f'Attributes: \t {", ".join(list(unique_attributes))}')
    #print(f'Category & Attribute Combinations: \t {", ".join(list(unique_category_attribute_combinations))}')

    print('')
    for split in ['train', 'test']:
        # Products per category calculations
        products_per_category = [len(records) for records in records_by_category[split].values()]
        print(f'Average products per category in {split}: \t {round(sum(products_per_category) / len(products_per_category), 2)}')
        print(f'Median products per category in {split}: \t {sorted(products_per_category)[len(products_per_category) // 2]}')
        print(f'Min products per category in {split}: \t {min(products_per_category)}')
        print(f'Max products per category in {split}: \t {max(products_per_category)}')


        # No. attributes per product calculations
        print(
            f'Average no. attributes per product in {split}: \t {round(sum(no_attributes_by_product[split]) / len(no_attributes_by_product[split]), 2)}')
        print(
            f'Median no. attributes per product in {split}: \t {sorted(no_attributes_by_product[split])[len(no_attributes_by_product[split]) // 2]}')
        print(f'Min no. attributes per product in {split}: \t {min(no_attributes_by_product[split])}')
        print(f'Max no. attributes per product in {split}: \t {max(no_attributes_by_product[split])}')

        # No. attributes per category calculations
        attributes_per_category = [len(attributes) for attributes in attributes_by_category[split].values()]
        print(
            f'Average no. attributes per category in {split}: \t {round(sum(attributes_per_category) / len(attributes_per_category), 2)}')
        print(f'Median no. attributes per category in {split}: \t {sorted(attributes_per_category)[len(attributes_per_category) // 2]}')
        print(f'Min no. attributes per category in {split}: \t {min(attributes_per_category)}')
        print(f'Max no. attributes per category in {split}: \t {max(attributes_per_category)}')

        # No. annotations per category & attribute calculations
        annotations_per_category = [sum([sum(attribute.values()) for attribute in attributes.values()]) for attributes in attributes_by_category[split].values()]
        print(
            f'Average no. annotations per category in {split}: \t {round(sum(annotations_per_category) / len(annotations_per_category), 2)}')
        print(f'Median no. annotations per category in {split}: \t {sorted(annotations_per_category)[len(annotations_per_category) // 2]}')
        print(f'Min no. annotations per category in {split}: \t {min(annotations_per_category)}')
        print(f'Max no. annotations per category in {split}: \t {max(annotations_per_category)}')

        # No. annotations per category calculations
        annotations_per_category_attribute = []
        for attributes in attributes_by_category[split].values():
            annotations_per_category_attribute.extend([sum(attribute.values()) for attribute in attributes.values()])
        print(
            f'Average no. annotations per category & attribute: \t {round(sum(annotations_per_category_attribute) / len(annotations_per_category_attribute), 2)}')
        print(f'Median no. annotations per category & attribute: \t {sorted(annotations_per_category_attribute)[len(annotations_per_category_attribute) // 2]}')
        print(f'Min no. annotations per category & attribute: \t {min(annotations_per_category_attribute)}')
        print(f'Max no. annotations per category & attribute: \t {max(annotations_per_category_attribute)}')

    # Calculate unique number of attributes per category
    unique_attributes_per_category = {}
    for split in ['train', 'test']:
        unique_attributes_per_category[split] = {}
        for category in attributes_by_category[split].keys():
            if category not in unique_attributes_per_category[split]:
                unique_attributes_per_category[split][category] = set()
            for attribute in attributes_by_category[split][category].keys():
                unique_attributes_per_category[split][category].add(attribute)

    # Calculate number of attribute-value pairs in train and test
    print('')
    no_attribute_value_pairs = {'train': 0, 'test': 0}
    for split in ['train', 'test']:
        # Calculate number of products per category
        for category in records_by_category[split].keys():
            no_attribute_value_pairs[split] += len(records_by_category[split][category]) * len(unique_attributes_per_category[split])

        print(f'Number of attribute-value pairs in {split}: \t {no_attribute_value_pairs[split]}')
    print('')

    # No. attribute values per attribute calculations
    no_attribute_values_by_attribute = []
    no_normalized_attribute_values_by_attribute = []
    no_numeric_attributes = 0
    no_mostly_numeric_attributes = 0

    attribute_value_length_by_attribute = {'short': 0, 'medium': 0, 'long': 0}

    token_length_of_attribute_values = []

    # Calculate total number of attribute values
    attribute_values_by_catgory_attribute_all = {}

    for split in ['train', 'test']:
        no_attribute_values_split = 0
        for category in attribute_values_by_catgory_attribute[split].keys():
            for attribute in attribute_values_by_catgory_attribute[split][category].keys():
                if category not in attribute_values_by_catgory_attribute_all:
                    attribute_values_by_catgory_attribute_all[category] = {}
                if attribute not in attribute_values_by_catgory_attribute_all[category]:
                    attribute_values_by_catgory_attribute_all[category][attribute] = set()
                no_attribute_values_split += len(attribute_values_by_catgory_attribute[split][category][attribute])
                for attribute_value in attribute_values_by_catgory_attribute[split][category][attribute]:
                    attribute_values_by_catgory_attribute_all[category][attribute].add(attribute_value)
        print(f'Number of attribute values in {split}: \t {no_attribute_values_split}')

    no_attribute_values = 0
    no_values_per_attribute_category = {}
    for category in attribute_values_by_catgory_attribute_all.keys():
        for attribute in attribute_values_by_catgory_attribute_all[category].keys():
            no_attribute_values += len(attribute_values_by_catgory_attribute_all[category][attribute])
            no_values_per_attribute_category[f'{category}-{attribute}'] = len(attribute_values_by_catgory_attribute_all[category][attribute])

    # Print total number of attribute values
    print(f'Total number of attribute values: \t {no_attribute_values}')
    print(f'Average number of attribute values per category-attribute: \t {round(no_attribute_values / len(no_values_per_attribute_category), 2)}')
    print(f'Median number of attribute values per category-attribute: \t {sorted(no_values_per_attribute_category.values())[len(no_values_per_attribute_category.values()) // 2]}')

    # Calculate number of attribute values from test set that are not in train set
    no_attribute_values_test_not_in_train = 0
    for category in attribute_values_by_catgory_attribute['test'].keys():
        for attribute in attribute_values_by_catgory_attribute['test'][category].keys():
            for attribute_value in attribute_values_by_catgory_attribute['test'][category][attribute]:
                if attribute not in attribute_values_by_catgory_attribute['train'][category]:
                    no_attribute_values_test_not_in_train += 1
                elif attribute_value not in attribute_values_by_catgory_attribute['train'][category][attribute]:
                    no_attribute_values_test_not_in_train += 1

    # Print number of attribute values from test set that are not in train set
    print(f'Number of attribute values from test set that are not in train set: \t {no_attribute_values_test_not_in_train}')

    # for category in attribute_values_by_catgory_attribute.keys():
    #     no_attribute_values_by_attribute.extend(
    #         [len(attribute_values) for attribute_values in attribute_values_by_catgory_attribute[category].values()])
    #
    #     for attribute_values in attribute_values_by_catgory_attribute[category].values():
    #         numerical_values = sum([1 for attribute_value in attribute_values if attribute_value.isdigit()])
    #         mostly_numeric_values = sum([1 for attribute_value in attribute_values if mostly_numeric(attribute_value)])
    #         normalized_values = set()
    #         attr_value_length = {'short': 0, 'medium': 0, 'long': 0}
    #         for attribute_value in attribute_values:
    #             attribute_value_length = len(encoding.encode(attribute_value))
    #             token_length_of_attribute_values.append(attribute_value_length)
    #             normalized_value = ' '.join([value.lower() for value in sorted(attribute_value.split(' '))])
    #             normalized_values.add(normalized_value)
    #             if attribute_value_length <= 2:
    #                 attr_value_length['short'] += 1
    #             elif attribute_value_length <= 6:
    #                 attr_value_length['medium'] += 1
    #             else:
    #                 attr_value_length['long'] += 1
    #
    #         if attr_value_length['short'] > attr_value_length['medium'] and attr_value_length['short'] > attr_value_length['long']:
    #             attribute_value_length_by_attribute['short'] += 1
    #         elif attr_value_length['medium'] > attr_value_length['short'] and attr_value_length['medium'] > attr_value_length['long']:
    #             attribute_value_length_by_attribute['medium'] += 1
    #         else:
    #             attribute_value_length_by_attribute['long'] += 1
    #
    #         no_normalized_attribute_values_by_attribute.append(len(normalized_values))
    #
    #         if numerical_values > 0.5 * len(attribute_values_by_catgory_attribute[category].values()):
    #             no_numeric_attributes += 1
    #         if mostly_numeric_values > 0.5 * len(attribute_values_by_catgory_attribute[category].values()):
    #             no_mostly_numeric_attributes += 1
    #
    #
    # print(
    #     f'Average no. attribute values per attribute: \t {round(sum(no_attribute_values_by_attribute) / len(no_attribute_values_by_attribute), 2)}')
    # print(
    #     f'Median no. attribute values per attribute: \t {sorted(no_attribute_values_by_attribute)[len(no_attribute_values_by_attribute) // 2]}')
    # print(f'Min no. attribute values per attribute: \t {min(no_attribute_values_by_attribute)}')
    # print(f'Max no. attribute values per attribute: \t {max(no_attribute_values_by_attribute)}')
    #
    # print(
    #     f'Average no. normalized attribute values per attribute: \t {round(sum(no_normalized_attribute_values_by_attribute) / len(no_normalized_attribute_values_by_attribute), 2)}')
    # print(
    #     f'Median no. normalized attribute values per attribute: \t {sorted(no_normalized_attribute_values_by_attribute)[len(no_normalized_attribute_values_by_attribute) // 2]}')
    # print(f'Min no. normalized attribute values per attribute: \t {min(no_normalized_attribute_values_by_attribute)}')
    # print(f'Max no. normalized attribute values per attribute: \t {max(no_normalized_attribute_values_by_attribute)}')


    # print(f'No. numeric attributes: \t {no_numeric_attributes}')
    # print(f'No. mostly numeric attributes: \t {no_mostly_numeric_attributes}')
    #
    # print(f'No. Attribute with mainly short attribute values: \t {attribute_value_length_by_attribute["short"]}')
    # print(f'No. Attribute with mainly medium attribute values: \t {attribute_value_length_by_attribute["medium"]}')
    # print(f'No. Attribute with mainly long attribute values: \t {attribute_value_length_by_attribute["long"]}')
    #
    # # Average Token Length of attribute values
    # print(
    #     f'Average token length of attribute values: \t {round(sum(token_length_of_attribute_values) / len(token_length_of_attribute_values), 2)}')
    # print(
    #     f'Median token length of attribute values: \t {sorted(token_length_of_attribute_values)[len(token_length_of_attribute_values) // 2]}')
    # print(f'Min token length of attribute values: \t {min(token_length_of_attribute_values)}')
    # print(f'Max token length of attribute values: \t {max(token_length_of_attribute_values)}')
    #
    # print('')
    # # Total no. tokens
    # print(f'Total no. tokens: \t {sum(num_title_tokens_by_product) + sum(num_description_tokens_by_product) + sum(num_feature_tokens_by_products) + sum(num_rest_tokens_by_products)}')
    #
    # # Average no. tokens per title
    # print(f'Average no. tokens per title: \t {round(sum(num_title_tokens_by_product) / len(num_title_tokens_by_product), 2)}')
    # # Median no. tokens per title
    # print(f'Median no. tokens per title: \t {sorted(num_title_tokens_by_product)[len(num_title_tokens_by_product) // 2]}')
    # print(f'Min no. tokens per title: \t {min(num_title_tokens_by_product)}')
    # print(f'Max no. tokens per title: \t {max(num_title_tokens_by_product)}')
    #
    # if len(num_description_tokens_by_product) > 0:
    #     # Average no. tokens per description
    #     print(f'Average no. tokens per description: \t {round(sum(num_description_tokens_by_product) / len(num_description_tokens_by_product), 2)}')
    #     # Median no. tokens per description
    #     print(f'Median no. tokens per description: \t {sorted(num_description_tokens_by_product)[len(num_description_tokens_by_product) // 2]}')
    #     print(f'Min no. tokens per description: \t {min(num_description_tokens_by_product)}')
    #     print(f'Max no. tokens per description: \t {max(num_description_tokens_by_product)}')
    #
    # if len(num_feature_tokens_by_products) > 0:
    #     # Average no. tokens per feature
    #     print(f'Average no. tokens per feature: \t {round(sum(num_feature_tokens_by_products) / len(num_feature_tokens_by_products), 2)}')
    #     # Median no. tokens per feature
    #     print(f'Median no. tokens per feature: \t {sorted(num_feature_tokens_by_products)[len(num_feature_tokens_by_products) // 2]}')
    #     print(f'Min no. tokens per feature: \t {min(num_feature_tokens_by_products)}')
    #     print(f'Max no. tokens per feature: \t {max(num_feature_tokens_by_products)}')
    #
    # if len(num_rest_tokens_by_products) > 0:
    #     # Average no. tokens per rest
    #     print(f'Average no. tokens per rest: \t {round(sum(num_rest_tokens_by_products) / len(num_rest_tokens_by_products))}')
    #     # Median no. tokens per rest
    #     print(f'Median no. tokens per rest: \t {sorted(num_rest_tokens_by_products)[len(num_rest_tokens_by_products) // 2]}')
    #     print(f'Min no. tokens per rest: \t {min(num_rest_tokens_by_products)}')
    #     print(f'Max no. tokens per rest: \t {max(num_rest_tokens_by_products)}')

    # print('')
    # # List attributes by category
    # for category in set(list(attributes_by_category['train'].keys()) + list(attributes_by_category['test'].keys())):
    #     print(f'Category: {category}')
    #     # List attributes sorted by no. annotations
    #     print(f'Attributes sorted by no. annotations (train):')
    #     for attribute in sorted(attributes_by_category['train'][category], key=lambda x: len(attributes_by_category['train'][category][x]), reverse=True):
    #         print(f'\t{attribute}: {len(attributes_by_category["train"][category][attribute])}')

if __name__ == '__main__':
    load_dotenv()
    random.seed(42)
    main()
