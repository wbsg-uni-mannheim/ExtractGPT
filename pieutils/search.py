import json
import os
import random
from typing import List, Dict

import faiss
import torch
from dotenv import load_dotenv
from langchain import FewShotPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import SemanticSimilarityExampleSelector, MaxMarginalRelevanceExampleSelector
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)

#from config import FAISS_INDEXES, PROCESSED_DATASETS
from pieutils.config import FAISS_INDEXES, PROCESSED_DATASETS


def load_train_mave(directory, categories):
    # Structure of train records: {id: {'category': category, 'title': title, 'extractions': {key: value}}}
    train_records = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if 'jsonl' in filename:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    record = json.loads(line)
                    if record['id'] not in train_records:
                        train_records[record['id']] = {'category': record['category'], 'title':
                            [paragraph['text'] for paragraph in record['paragraphs'] if paragraph['source'] == 'title'][
                                0], 'extractions': {}}

                    # Add Extractions
                    for extraction in record['attributes']:
                        key = extraction['key']
                        values = [evidence['value'] for evidence in extraction['evidences'] if evidence['pid'] == 0]
                        value = values[0] if len(values) > 0 else 'n/a'
                        train_records[record['id']]['extractions'][key] = value
    return train_records


def load_train_oa_mine(dataset, categories, skip_first_n=50):
    """Load train records for oa_mine dataset.
        :param dataset: Path to dataset directory
        :param categories: List of categories to load"""
    # TODO: Change from skipping to train/test split

    # Structure of train records: {id: {'category': category, 'title': title, 'extractions': {key: value}}}
    train_records = {}
    for filename in os.listdir(dataset):
        if os.path.isfile(os.path.join(dataset, filename)) and 'jsonl' in filename:
            category = ' '.join([token.capitalize() for token in filename.replace('.jsonl', '').split('_')])
            if category in categories:
                file_path = os.path.join(dataset, filename)
                with open(file_path, 'r') as f:
                    for line in f.readlines()[skip_first_n:]:
                        record = json.loads(line)
                        example = {'title': record['title'], 'category': category, 'extractions': {}}
                        for attribute in record['entities']:
                            example['extractions'][attribute['label']] = attribute['value']

                        if len(example['extractions']) > 0:
                            train_records[record['asin']] = example
    return train_records


def load_train_for_vector_store(dataset_name, categories, train_percentage=1.0):
    """Load dataset train records for vector store.
        If categories is specified the subset is considered otherwise all categories are loaded."""

    directory_path = f'{PROCESSED_DATASETS}/{dataset_name}/'
    if train_percentage < 1.0:
        file_path = os.path.join(directory_path, f'train_{train_percentage}.jsonl')
    else:
        file_path = os.path.join(directory_path, 'train.jsonl')

    # Structure of train records: {id: {'category': category, 'input': input, 'extractions': {key: value}}}
    train_records = {}
    example_id = 0
    with open(file_path, 'r') as f:
        for line in f.readlines():
            # Load record from title, description, features and rest of the attributes. - Mainly relevant for MAVE
            record = json.loads(line)
            for part in ['input', 'description', 'features', 'rest', 'paragraphs']:
                if part in record:
                    if categories is None or record['category'] in categories:
                        if part == 'paragraphs':
                            for paragraph in record[part]:
                                train_records[str(example_id)] = {'input': paragraph,
                                                                  'category': record['category'],
                                                                  'part': part,
                                                                  'extractions': {}}
                                for key, value_dict in record['target_scores'].items():
                                    if 'n/a' in value_dict:
                                        train_records[str(example_id)]['extractions'][key] = "n/a"
                                    else:
                                        contained_values = [k for k, v in value_dict.items() if k in paragraph]
                                        if len(contained_values) > 0:
                                            train_records[str(example_id)]['extractions'][key] = max(contained_values,
                                                                                                     key=len)
                                        else:
                                            train_records[str(example_id)]['extractions'][key] = "n/a"
                                example_id += 1
                        else:
                            train_records[str(example_id)] = {'input': record[part], 'part': part,
                                                              'category': record['category'],
                                                              'extractions': {}}
                            for key, value_dict in record['target_scores'].items():
                                if 'n/a' in value_dict:
                                    train_records[str(example_id)]['extractions'][key] = "n/a"
                                else:
                                    contained_values = [k for k, v in value_dict.items() if k in record[part]]
                                    if len(contained_values) > 0:
                                        train_records[str(example_id)]['extractions'][key] = max(contained_values,
                                                                                                 key=len)
                                    else:
                                        train_records[str(example_id)]['extractions'][key] = "n/a"
                            example_id += 1
    return train_records


def initialize_vector_store(dataset_name, load_from_local=False, categories=None, train_percentage=1.0):
    # Load the training data.
    if dataset_name in ['mave', 'mave_v2', 'ae-110k', 'oa-mine']:
        train_records = load_train_for_vector_store(dataset_name, categories=categories, train_percentage=train_percentage)
    else:
        raise ValueError(f'Dataset {dataset_name} not supported!')

    categories = list(set([record['category'] for record in train_records.values()]))
    category_to_vector_store = {}
    # Load the product titles along with metadata into the vector store.

    if load_from_local:
        for category in categories:
            folder_path = f'{FAISS_INDEXES}/{dataset_name}'
            index_name = f'{dataset_name}_faiss_index_{category}'
            category_to_vector_store[category] = FAISS.load_local(folder_path=folder_path,
                                                                  embeddings=OpenAIEmbeddings(), index_name=index_name)
    else:
        for category in categories:
            inputs = [record['input'] for record in train_records.values() if record['category'] == category]
            metadatas = [{'input': record['input'], 'output': json.dumps(record['extractions'])}
                         for record in train_records.values() if record['category'] == category]
            ids = [key for key in train_records.keys() if train_records[key]['category'] == category]
            category_to_vector_store[category] = FAISS.from_texts(inputs, OpenAIEmbeddings(), metadatas=metadatas,
                                                                  ids=ids)
            # Check if directory for vector store exists.
            folder_path = f'{FAISS_INDEXES}/{dataset_name}'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # Save the vector store locally.
            index_name = f'{dataset_name}_faiss_index_{category}'
            category_to_vector_store[category].save_local(folder_path=folder_path, index_name=index_name)
    return category_to_vector_store


def apply_pydantic_model(extractions, pydantic_model):
    """Apply the pydantic model and return the json string."""
    pydantic_dict = json.loads(pydantic_model(**extractions).json())
    for key in pydantic_dict:
        if pydantic_dict[key] is None:
            pydantic_dict[key] = 'n/a'

    pydantic_json = json.dumps(pydantic_dict)
    return pydantic_json


def cluster_examples_using_kmeans(examples, max_cluster_size=5, sliding_wino_size=0):
    """Cluster examples using k-means clustering. Assign each example to the cluster with the nearest centroid."""

    titles = [example['input'] for example in examples]

    emb = OpenAIEmbeddings(chunk_size=1)
    embeddings = emb.embed_documents(titles)
    tensor_of_text_embeddings = torch.tensor(embeddings)

    # Cluster the examples.
    d = tensor_of_text_embeddings.shape[1]
    nclusters = len(examples) // max_cluster_size if len(examples) % max_cluster_size == 0 else len(
        examples) // max_cluster_size + 1
    niter = 20
    kmeans = faiss.Kmeans(d, nclusters, niter=niter, verbose=True, min_points_per_centroid=max_cluster_size, seed=42)
    kmeans.train(tensor_of_text_embeddings)

    # Get the cluster assignments.
    D, I = kmeans.index.search(tensor_of_text_embeddings, 1)

    # clustering = AgglomerativeClustering(n_clusters=nclusters, metric='cosine', linkage='average').fit(embeddings)

    # Assign each example to the cluster with the nearest centroid.
    examples = [{'input': example['input'], 'category': example['category'],
                 'cluster_id': str(cluster[0]) if 'cluster_id' not in example else example['cluster_id'] + str(
                     cluster[0])} for example, cluster in zip(examples, I)]

    # Add a sliding window over large clusters.
    if sliding_wino_size > 0:
        for cluster_id in [example['cluster_id'] for example in examples]:
            clustered_examples = [example for example in examples if example['cluster_id'] == cluster_id]
            if len(clustered_examples) > max_cluster_size:
                # Apply sliding window over large clusters.
                # Sort examples by input.
                clustered_examples = sorted(clustered_examples, key=lambda x: x['input'])
                # Apply sliding window.
                clustered_example_windows = [clustered_examples[i:i + max_cluster_size] for i in
                                             range(0, len(clustered_examples), max_cluster_size)]
                for i, clustered_example_window in enumerate(clustered_example_windows):
                    for j, example in enumerate(clustered_example_window):
                        example['cluster_id'] = example['cluster_id'] + str(i)
                clustered_examples = [example for clustered_example_window in clustered_example_windows for example in
                                      clustered_example_window]
                for original_example, clustered_example in zip(
                        [example for example in examples if example['cluster_id'] == cluster_id], clustered_examples):
                    original_example['cluster_id'] = clustered_example['cluster_id']

    return examples


def cluster_examples_using_sliding_window(examples, max_cluster_size=5):
    """Cluster examples using a sliding window."""
    clustered_examples = sorted(examples, key=lambda x: x['input'])
    # Apply sliding window.
    clustered_example_windows = [clustered_examples[i:i + max_cluster_size] for i in
                                 range(0, len(clustered_examples), max_cluster_size)]
    for i, clustered_example_window in enumerate(clustered_example_windows):
        for j, example in enumerate(clustered_example_window):
            example['cluster_id'] = str(i)
    clustered_examples = [example for clustered_example_window in clustered_example_windows for example in
                          clustered_example_window]

    return clustered_examples


def test_clustering():
    directory_path = 'data/oa_mine/annotations/'
    examples_for_clustering = {}
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            category = ' '.join([token.capitalize() for token in filename.replace('.jsonl', '').split('_')])
            if category not in examples_for_clustering:
                examples_for_clustering[category] = []
            print('Load records for category: {}'.format(category))
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as f:
                for line in f.readlines()[:50]:
                    record = json.loads(line)
                    example = {'input': record['title'], 'category': category}
                    examples_for_clustering[category].append(example)

    # Cluster examples
    # examples = cluster_examples_using_sliding_window(examples_for_clustering['Breakfast Cereal'], 5)
    examples = cluster_examples_using_kmeans(examples_for_clustering['Breakfast Cereal'], 5)

    assert len(examples) == len(examples_for_clustering['Breakfast Cereal'])
    for cluster_id in list(set([example['cluster_id'] for example in examples])):
        clustered_examples = [example['input'] for example in examples if example['cluster_id'] == cluster_id]
        print('Cluster {}: \n {}'.format(cluster_id, '\n '.join(clustered_examples)))


def convert_example_to_pydantic_model_example(example, pydantic_model):
    """Convert example to pydantic model."""
    pydantic_example = pydantic_model(**json.loads(example['output']))
    pydantic_example_json = pydantic_example.json(indent=4).replace('null', '"n/a"')
    return {'input': example['input'], 'output': pydantic_example_json}


class CategoryAwareMaxMarginalRelevanceExampleSelector(BaseExampleSelector):

    def __init__(self, dataset_name, categories, category_2_pydantic_models=None, load_from_local=False, k=5,
                 tabular=False) -> None:
        """Initialize the example selector from the datasets"""
        category_to_vector_store = initialize_vector_store(dataset_name, load_from_local=load_from_local,
                                                           categories=categories)
        # Initialize Different example selectors for each category.
        self.example_selector_by_category = {
            category: MaxMarginalRelevanceExampleSelector(vectorstore=category_to_vector_store[category], k=k) for
            category in category_to_vector_store}
        self.category_2_pydantic_models = category_2_pydantic_models
        if self.category_2_pydantic_models is not None:
            self.tabular = tabular
        elif tabular:
            print('Tabular is set to True but no pydantic models are provided. Tabular will be set to False.')
            self.tabular = False

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.example_selector_by_category[example['category']].add_example(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        category = input_variables['category']
        selected_examples = self.example_selector_by_category[category].select_examples(
            {'input': input_variables['input']})
        if self.category_2_pydantic_models is not None:

            if self.tabular:
                # Convert examples to tabular format.
                inputs = '\n'.join([example['input'] for example in selected_examples])
                selected_outputs = {'records': [json.loads(example['output']) for example in selected_examples]}
                pydantic_examples = self.category_2_pydantic_models[category](**selected_outputs)
                pydantic_example_json = pydantic_examples.json(indent=4).replace('null', '"n/a"')
                selected_examples = [{'inputs': inputs, 'outputs': pydantic_example_json}]
            else:
                # Convert examples to pydantic models.
                pydantic_model = self.category_2_pydantic_models[input_variables['category']]
                selected_examples = [convert_example_to_pydantic_model_example(example, pydantic_model) for example in
                                     selected_examples]

        return selected_examples

class CategoryAwareSemanticSimilarityExampleSelector(BaseExampleSelector):

    def __init__(self, dataset_name, categories, category_2_pydantic_models=None, load_from_local=False, k=5,
                 tabular=False, train_percentage=1) -> None:
        """Initialize the example selector from the datasets"""
        category_to_vector_store = initialize_vector_store(dataset_name, load_from_local=load_from_local,
                                                           categories=categories, train_percentage=train_percentage)
        # Initialize Different example selectors for each category.
        self.example_selector_by_category = {
            category: SemanticSimilarityExampleSelector(vectorstore=category_to_vector_store[category], k=k) for
            category in category_to_vector_store}
        self.category_2_pydantic_models = category_2_pydantic_models
        if self.category_2_pydantic_models is not None:
            self.tabular = tabular
        elif tabular:
            print('Tabular is set to True but no pydantic models are provided. Tabular will be set to False.')
            self.tabular = False

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.example_selector_by_category[example['category']].add_example(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        category = input_variables['category']
        selected_examples = self.example_selector_by_category[category].select_examples(
            {'input': input_variables['input']})
        if self.category_2_pydantic_models is not None:

            if self.tabular:
                # Convert examples to tabular format.
                inputs = '\n'.join([example['input'] for example in selected_examples])
                selected_outputs = {'records': [json.loads(example['output']) for example in selected_examples]}
                pydantic_examples = self.category_2_pydantic_models[category](**selected_outputs)
                pydantic_example_json = pydantic_examples.json(indent=4).replace('null', '"n/a"')
                selected_examples = [{'inputs': inputs, 'outputs': pydantic_example_json}]
            else:
                # Convert examples to pydantic models.
                pydantic_model = self.category_2_pydantic_models[input_variables['category']]
                selected_examples = [convert_example_to_pydantic_model_example(example, pydantic_model) for example in
                                     selected_examples]

        return selected_examples

class CategoryAwareSemanticSimilarityDifferentAttributeValuesExampleSelector(BaseExampleSelector):

    def __init__(self, dataset_name, categories, category_2_pydantic_models=None, load_from_local=False, k=5,
                 tabular=False, train_percentage=1) -> None:
        """Initialize the example selector from the datasets"""
        self.k = k
        category_to_vector_store = initialize_vector_store(dataset_name, load_from_local=load_from_local,
                                                           categories=categories, train_percentage=train_percentage)
        # Initialize Different example selectors for each category.
        self.example_selector_by_category = {
            category: SemanticSimilarityExampleSelector(vectorstore=category_to_vector_store[category], k=self.k*20) for
            category in category_to_vector_store}
        self.category_2_pydantic_models = category_2_pydantic_models
        if self.category_2_pydantic_models is not None:
            self.tabular = tabular
        elif tabular:
            print('Tabular is set to True but no pydantic models are provided. Tabular will be set to False.')
            self.tabular = False

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.example_selector_by_category[example['category']].add_example(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        category = input_variables['category']
        selected_examples = self.example_selector_by_category[category].select_examples(
            {'input': input_variables['input']})

        # Select examples with different attribute values
        sub_selected_examples = [selected_examples[0]]
        if len(sub_selected_examples) < self.k:
            for example in selected_examples:
                # Attribute Values of the already selected examples
                sub_selected_attribute_values = { }
                for sub_example in sub_selected_examples:
                    sub_output = json.loads(sub_example['output'])
                    for attribute in sub_output:
                        if attribute not in sub_selected_attribute_values:
                            sub_selected_attribute_values[attribute] = set()
                        sub_selected_attribute_values[attribute].add(sub_output[attribute])

                # Check if the example has different attribute values
                different_attribute_values = False
                output = json.loads(example['output'])
                for attribute in output:
                    if attribute not in sub_selected_attribute_values:
                        different_attribute_values = True
                        break
                    if output[attribute] not in sub_selected_attribute_values[attribute]:
                        different_attribute_values = True
                        break
                if different_attribute_values:
                    sub_selected_examples.append(example)
                    if len(sub_selected_examples) == self.k:
                        break

            # Add more examples if not enough
            if len(sub_selected_examples) < self.k:
                for example in selected_examples:
                    if example not in sub_selected_examples:
                        sub_selected_examples.append(example)
                        if len(sub_selected_examples) == self.k:
                            break

        if self.category_2_pydantic_models is not None:
            # Convert examples to pydantic models.
            pydantic_model = self.category_2_pydantic_models[input_variables['category']]
            sub_selected_examples = [convert_example_to_pydantic_model_example(example, pydantic_model) for example in
                                 sub_selected_examples]

        return sub_selected_examples

class CategoryAwareRandomExampleSelector(BaseExampleSelector):

    def __init__(self, dataset_name, categories, category_2_pydantic_models=None, k=5) -> None:
        """Initialize the example selector from the datasets"""
        self.k = k
        self.category_2_pydantic_models = category_2_pydantic_models
        if dataset_name in ['mave', 'mave_v2', 'ae-110k', 'oa-mine']:
            examples = load_train_for_vector_store(dataset_name, categories=categories)
        else:
            raise ValueError(f'Dataset {dataset_name} not supported!')

        # Split Examples by category.
        self.examples_by_category = {}
        for example in examples.values():
            category = example['category']
            if category not in self.examples_by_category:
                self.examples_by_category[category] = []
            self.examples_by_category[category].append(
                {'input': example['input'], 'output': json.dumps(example['extractions'])})

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.examples_by_category[example['category']].append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        category = input_variables['category']
        selected_examples = random.sample(self.examples_by_category[category], self.k)
        if self.category_2_pydantic_models is not None:
            pydantic_model = self.category_2_pydantic_models[input_variables['category']]
            selected_examples = [convert_example_to_pydantic_model_example(example, pydantic_model) for example in
                                 selected_examples]
        return selected_examples


class CategoryAwareFixedExampleSelector(BaseExampleSelector):

    def __init__(self, dataset_name, categories, category_2_pydantic_models=None, load_from_local=False, k=5,
                 tabular=False) -> None:
        """Initialize the example selector CategoryAwareFixedExampleSelector -
            Idea: Retrieve a fixed set of examples for each category using the first request.
            Then, use the same set of examples for all subsequent requests."""
        self.categoryAwareMaxMarginalRelevanceExampleSelector = CategoryAwareMaxMarginalRelevanceExampleSelector(
            dataset_name, categories, category_2_pydantic_models, load_from_local, k, tabular)
        self.example_selector_by_category = {category: [] for category in categories}

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.categoryAwareMaxMarginalRelevanceExampleSelector.add_example(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        category = input_variables['category']
        if len(self.example_selector_by_category[category]) == 0:
            self.example_selector_by_category[category] = self.categoryAwareMaxMarginalRelevanceExampleSelector.select_examples(
                input_variables)
        return self.example_selector_by_category[category]


if __name__ == '__main__':
    load_dotenv()
    random.seed(42)

    # Test clustering
    # test_clustering()
    # Initialize Vector Store with training data

    dataset_name = 'ae-110k'
    category = 'Baseball Caps'
    # category_to_vector_store = initialize_vector_store(dataset_name, load_from_local=False,
    #                                                    categories=[category])
    #
    # # Create shot
    query = 'Unisex Baseball Cap Jurassic Park Dinosaur Print Mens Womens Baseball Caps Couples Adjustable Snapback Caps Hats'
    #
    # # Similarity Selector
    # category_to_example_sim_selector = {
    #     category: SemanticSimilarityExampleSelector(vectorstore=category_to_vector_store[category], k=5) for category in
    #     category_to_vector_store}
    # shots = category_to_example_sim_selector[category].select_examples({'input': query})
    # print(shots)
    #
    # # Max Rel Selector
    # category_to_example_max_rel_selector = {
    #     category: MaxMarginalRelevanceExampleSelector(vectorstore=category_to_vector_store[category], k=5) for category
    #     in
    #     category_to_vector_store}
    # shots = category_to_example_max_rel_selector[category].select_examples({'input': query})
    # print(shots)
    #
    # # Custom Example Selector
    # category_example_selector = CategoryAwareMaxMarginalRelevanceExampleSelector(dataset_name, [category],
    #                                                                              load_from_local=True, k=5)
    #


    example_selector = CategoryAwareSemanticSimilarityDifferentAttributeValuesExampleSelector(dataset_name, [category], load_from_local=True, k=5)
    # Define the few-shot prompt.
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        # The input variables select the values to pass to the example_selector
        input_variables=["input", "category"],
        example_selector=example_selector,
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        ),
    )
    print(few_shot_prompt.format(input=query, category=category))
