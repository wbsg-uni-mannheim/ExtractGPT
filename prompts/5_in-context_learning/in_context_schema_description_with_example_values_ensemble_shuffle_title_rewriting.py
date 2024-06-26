import json
# Update Task Dict
import random
from datetime import datetime
from json import JSONDecodeError

import click
import pandas as pd
import torch
from dotenv import load_dotenv
from langchain import LLMChain, HuggingFacePipeline
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    AIMessagePromptTemplate, FewShotChatMessagePromptTemplate
from pydantic import ValidationError
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from pieutils.pieutils import save_populated_task, create_pydanctic_model_from_pydantic_meta_model, \
    create_dict_of_pydanctic_product, convert_to_json_schema, parse_llm_response_to_json, ensemble_predictions, \
    save_meta_model
from pieutils.evaluation import evaluate_predictions
from pieutils.fusion import fuse_models
from pieutils.preprocessing import update_task_dict_from_test_set, load_known_attribute_values
from pieutils.pydantic_models import ProductCategory
from pieutils.search import CategoryAwareMaxMarginalRelevanceExampleSelector, CategoryAwareRandomExampleSelector, \
    CategoryAwareFixedExampleSelector, CategoryAwareSemanticSimilarityExampleSelector, \
    CategoryAwareSemanticSimilarityDifferentAttributeValuesExampleSelector


@click.command()
@click.option('--dataset', default='oa-mine', help='Dataset Name')
@click.option('--model', default='gpt-3.5-turbo-0613', help='Model name')
@click.option('--verbose', default=False, help='Verbose mode')
@click.option('--shots', default=3, help='Number of shots for few-shot learning')
@click.option('--example_selector', default='SemanticSimilarity', help='Example selector for few-shot learning')
@click.option('--train_percentage', default=1.0, help='Percentage of training data used for in-context learning')
@click.option('--with_containment', default=False, help='Use containment')
@click.option('--with_validation_error_handling', default=False, help='Use validation error handling')
@click.option('--schema_type', default='json_schema',
              help='Schema to use - json_schema, json_schema_no_type or compact')
@click.option('--no_example_values', default=10, help='Number of example values extracted from training set')
@click.option('--temperature', default=0.0, help='Temperature value between 0.0 and 2.0')
def main(dataset, model, verbose, shots, example_selector, train_percentage, with_containment,
         with_validation_error_handling, schema_type, no_example_values, temperature):
    # Load task template
    with open('prompts/task_template.json', 'r') as f:
        task_dict = json.load(f)

    task_dict[
        'task_name'] = "chatgpt_description_with_example_values_in-context_learning_{}_{}_{}_{}_{}_{}-{}_{}_ensembling_shuffle_title_rewriting".format(
        dataset, model, shots, example_selector, train_percentage, 'containment_check' if with_containment else '',
        'validation_error_handling' if with_containment else '', schema_type, temperature).replace('.', '_').replace(
        '-', '_').replace('/', '_')
    task_dict['task_prefix'] = "Split the product title by whitespace. " \
                               "Extract the valid attribute values from the product {part} in JSON format. Keep the " \
                               "exact surface form of all attribute values. All valid attributes are provided in the " \
                               "JSON schema. Unknown attribute values should be marked as n/a."

    task_dict['dataset_name'] = dataset
    task_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_dict['shots'] = shots
    task_dict['example_selector'] = example_selector
    task_dict['train_percentage'] = train_percentage

    # Load examples into task dict
    task_dict = update_task_dict_from_test_set(task_dict)
    known_attribute_values = load_known_attribute_values(task_dict['dataset_name'], n_examples=no_example_values)
    if verbose:
        print('Known attribute values:')
        print(known_attribute_values)

    # # Filter examples top 5 examples
    # task_dict['examples'] = task_dict['examples'][:5]
    # # Filter known attribute values based on first example
    # known_attribute_values = {k: v for k, v in known_attribute_values.items() if k in task_dict['examples'][0]['category']}
    # task_dict['known_attributes'] = {k: v for k, v in task_dict['known_attributes'].items() if k in task_dict['examples'][0]['category']}

    # Initialize model
    task_dict['model'] = model
    if 'gpt-3' in task_dict['model'] or 'gpt-4' in task_dict['model']:
        llm = ChatOpenAI(model_name=task_dict['model'], temperature=0)
    else:
        tokenizer = AutoTokenizer.from_pretrained(task_dict['model'], use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(task_dict['model'], torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True, device_map="auto")
        model.tie_weights()

        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0, top_p=0.95)
        llm = HuggingFacePipeline(pipeline=pipe)
    default_llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0)

    # Ask LLM to generate meta models for each product category and attribute

    # Create Chains
    system_message_prompt = SystemMessagePromptTemplate.from_template("You are a world class algorithm for creating "
                                                                      "descriptions of product categories and their "
                                                                      "attributes following this JSON schema: \n {schema}.")
    human_task_meta_model = HumanMessagePromptTemplate.from_template("Write short descriptions for the product category"
                                                                     " {category} and the attributes {attributes} that are helpful to identify relevant attribute values in product titles."
                                                                     "The descriptions should not be longer than one sentence."
                                                                     "The following attribute values are known for each attribute: \n"
                                                                     "{known_attribute_values}. \n"
                                                                     "Respond with a JSON object following the provided schema.")
    prompt_meta_model = ChatPromptTemplate(messages=[system_message_prompt, human_task_meta_model])
    pydantic_models = {}
    models_json = {}
    for category in task_dict['known_attributes']:
        print('Create model for category: {}'.format(category))

        chain = LLMChain(
            prompt=prompt_meta_model,
            llm=default_llm,
            verbose=verbose
        )
        # chain_meta_model = create_structured_output_chain(ProductCategory, llm, prompt_meta_model, verbose=True)
        known_attribute_values_per_category = json.dumps(known_attribute_values[category])
        response = chain.run({'schema': convert_to_json_schema(ProductCategory, False), 'category': category,
                              'attributes': ', '.join(task_dict['known_attributes'][category]),
                              'known_attribute_values': known_attribute_values_per_category})
        pred = ProductCategory(**json.loads(response))
        if verbose:
            print(pred)
        pydantic_models[category] = create_pydanctic_model_from_pydantic_meta_model(pred)
        models_json[category] = create_dict_of_pydanctic_product(pydantic_models[category])

    # Persist models
    save_meta_model(task_dict['task_name'], 'default_gpt3_5', task_dict['dataset_name'], models_json)

    # Create Chains
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        "You are a world class algorithm for extracting information in structured JSON formats. \n {schema} ")
    human_task_prompt = HumanMessagePromptTemplate.from_template(task_dict['task_prefix'])

    # Add few shot sampling
    if example_selector == 'MaxMarginalRelevance':
        category_example_selector = CategoryAwareMaxMarginalRelevanceExampleSelector(task_dict['dataset_name'],
                                                                                     list(task_dict[
                                                                                              'known_attributes'].keys()),
                                                                                     category_2_pydantic_models=pydantic_models,
                                                                                     load_from_local=True, k=shots)
    elif example_selector == 'SemanticSimilarity':
        category_example_selector = CategoryAwareSemanticSimilarityExampleSelector(task_dict['dataset_name'],
                                                                                   list(task_dict[
                                                                                            'known_attributes'].keys()),
                                                                                   category_2_pydantic_models=pydantic_models,
                                                                                   load_from_local=False, k=shots,
                                                                                   train_percentage=train_percentage)
    elif example_selector == 'SemanticSimilarityDifferentAttributeValues':
        category_example_selector = CategoryAwareSemanticSimilarityDifferentAttributeValuesExampleSelector(
            task_dict['dataset_name'],
            list(task_dict['known_attributes'].keys()),
            category_2_pydantic_models=pydantic_models,
            load_from_local=True, k=shots, train_percentage=train_percentage)

    elif example_selector == 'Random':
        category_example_selector = CategoryAwareRandomExampleSelector(task_dict['dataset_name'],
                                                                       list(task_dict['known_attributes'].keys()),
                                                                       category_2_pydantic_models=pydantic_models,
                                                                       k=shots)
    elif example_selector == 'Fixed':
        category_example_selector = CategoryAwareFixedExampleSelector(task_dict['dataset_name'],
                                                                      list(task_dict['known_attributes'].keys()),
                                                                      category_2_pydantic_models=pydantic_models,
                                                                      load_from_local=True, k=shots)
    else:
        raise NotImplementedError("Example selector {} not implemented".format(example_selector))

    # Define the few-shot prompt.
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        # The input variables select the values to pass to the example_selector
        input_variables=["input", "category", "attribute_order"],
        example_selector=category_example_selector,
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        ),
    )

    human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate(messages=[system_message_prompt, human_task_prompt, few_shot_prompt, human_task_prompt,
                                          human_message_prompt])
    chain = LLMChain(
        prompt=prompt,
        llm=llm,
        verbose=verbose
    )

    # Containment chain:
    ai_response = AIMessagePromptTemplate.from_template("{response}")
    human_verfification_request = HumanMessagePromptTemplate.from_template(
        'The attribute value(s) "{values}" is/are not found as exact substrings in the product title "{input}". Update all attribute values such that they are substrings of the product title.')
    verification_prompt_list = [system_message_prompt, human_task_prompt, human_message_prompt, ai_response,
                                human_verfification_request]
    verification_prompt = ChatPromptTemplate(messages=verification_prompt_list)
    verification_chain = LLMChain(
        prompt=verification_prompt,
        llm=llm,
        verbose=verbose
    )

    # Error handling chain:
    ai_response = AIMessagePromptTemplate.from_template("{response}")
    human_error_handling_request = HumanMessagePromptTemplate.from_template(
        "Change the attributes '{error}' to string values. List values should be concatenated by ' '.")
    error_handling_prompt_list = [system_message_prompt, human_task_prompt, human_message_prompt, ai_response,
                                  human_error_handling_request]
    error_handling_prompt = ChatPromptTemplate(messages=error_handling_prompt_list)
    error_handling_chain = LLMChain(
        prompt=error_handling_prompt,
        llm=llm,
        verbose=verbose
    )
    
    # Rewrite title chain:
    system_message_rewriting_prompt = SystemMessagePromptTemplate.from_template(
        "You are an expert in creating product titles.")
    human_task_rewriting_prompt = HumanMessagePromptTemplate.from_template('Provide 2 alternative product titles for the following product offer: {input}. Respond with a json formatted as "title1": "[title]", "title2":"[title]".')

    prompt_rewriting = ChatPromptTemplate(messages=[system_message_rewriting_prompt, human_task_rewriting_prompt])
    rewriting_chain = LLMChain(
        prompt=prompt_rewriting,
        llm=llm,
        verbose=verbose
    )

    # Merge Extraction Results
    system_message_merge_prompt = SystemMessagePromptTemplate.from_template(
        "You are a world class algorithm for merging attribute values extracted from different parts of a product offer.")
    human_task_merge_prompt = HumanMessagePromptTemplate.from_template("Verfiy that the attribute values extracted by various algorithms haven been correctly extracted from the product title and merge them into a single JSON object following the provided schema. ")
    human_task_merge_input_prompt = HumanMessagePromptTemplate.from_template("Product Title: {input} \n Extracted Attribute Values: {extracted_values}")
    prompt_merge = ChatPromptTemplate(messages=[system_message_merge_prompt, human_task_merge_prompt, human_task_merge_input_prompt])
    merge_chain = LLMChain(
        prompt=prompt_merge,
        llm=llm,
        verbose=verbose
    )


    def validation_error_handling(category, input_text, response, error, part='title'):

        response_dict = json.loads(response)
        error_attributes = [error['loc'][0] for error in error.errors() if error['type'] == 'type_error.str']

        # Update all attributes that are not strings to strings by LLM
        pred = None
        error_handling_response = error_handling_chain.run(input=input_text,
                                                           schema=convert_to_json_schema(pydantic_models[category]),
                                                           response=json.dumps(response_dict),
                                                           error="', '".join(error_attributes), part=part)
        try:
            # Convert json string into pydantic model
            print(json.loads(error_handling_response))
            pred = pydantic_models[category](**json.loads(error_handling_response))
        except ValidationError as valError:
            pred = validation_error_handling(category, input_text, response, valError)
        except JSONDecodeError as e:
            print('Error: {}'.format(e))

        return pred, error_handling_response

    def select_and_run_llm(category, input_text, part='title', with_validation_error_handling=True,
                           with_containment=True, schema_type='json_schema'):

        # Shuffle the attributes and ensemble the prediction using majority voting
        preds = []
        num_ensembles = 1  # Fix to 1 for now

        # Get the rewritten titles
        rewritten_titles = [input_text]
        response = rewriting_chain.run(input=input_text)
        # Split response into titles
        try:
            response_dict = json.loads(response)
            rewritten_titles = rewritten_titles + [response_dict['title1'], response_dict['title2']]
        except JSONDecodeError as e:
            print('Error: {}'.format(e))

        for rewritten_title in rewritten_titles:
            for i in range(0, num_ensembles):
                pred = None
                try:
                    # Keep the original order for the first ensemble
                    properties = None
                    attribute_order = None
                    if i > 0:
                        properties = list(pydantic_models[category].schema()['properties'].items())
                        random.seed(i)
                        random.shuffle(properties)
                        attribute_order = [p[0] for p in properties]

                    # Run Chain
                    response = chain.run(input=rewritten_title,
                                         schema=convert_to_json_schema(pydantic_models[category], schema_type=schema_type,
                                                                       properties=properties),
                                         part=part, category=category, attribute_order=attribute_order)

                    try:
                        # Convert json string into pydantic model
                        if verbose:
                            print(json.loads(response))
                        pred = pydantic_models[category](**json.loads(response))
                    except ValidationError as valError:
                        if with_validation_error_handling:
                            pred, response = validation_error_handling(category, rewritten_title, response, valError)
                        else:
                            print('Validation Error: {}'.format(valError))
                    except JSONDecodeError as e:
                        converted_response = parse_llm_response_to_json(response)
                        if verbose:
                            print(converted_response)
                        try:
                            pred = pydantic_models[category](**converted_response)
                        except ValidationError as valError:
                            if with_validation_error_handling:
                                pred, response = validation_error_handling(category, rewritten_title, response, valError)
                            else:
                                print('Validation Error: {}'.format(valError))
                        except Exception as e:
                            print(e)
                    except Exception as e:
                        print(e)
                        print('Response: ')
                        print(response)

                    if pred is not None and with_containment:
                        # Verify if the extracted attribute values are contained in the input text and make sure that no loop is entered.
                        not_found_values = [v for v in pred.dict().values() if
                                            v is not None and v != 'n/a' and v not in rewritten_title]
                        previous_responses = []
                        while len(not_found_values) > 0 and response not in previous_responses:
                            # Verify extracted attribute values
                            if verbose:
                                print('Not found values: {}'.format(not_found_values))
                            # Try this only once to avoid infinite loops
                            verified_response = verification_chain.run(input=rewritten_title,
                                                                       schema=convert_to_json_schema(
                                                                           pydantic_models[category]),
                                                                       response=response,
                                                                       values='", "'.join(not_found_values),
                                                                       part=part)
                            previous_responses.append(response)
                            try:
                                if verbose:
                                    print(json.loads(verified_response))
                                pred = pydantic_models[category](**json.loads(verified_response))
                                not_found_values = [v for v in pred.dict().values() if
                                                    v is not None and v != 'n/a' and v not in rewritten_title]
                            except ValidationError as valError:
                                if with_validation_error_handling:
                                    pred, verified_response = validation_error_handling(category, rewritten_title,
                                                                                        verified_response,
                                                                                        valError, part)
                                else:
                                    print('Validation Error: {}'.format(valError))
                            except JSONDecodeError as e:
                                print('Error: {}'.format(e))
                except Exception as e:
                    print(e)

                preds.append(pred)

        # Ensemble the predictions using an LLM
        pred = None
        if len(preds) > 0:
            # Majority Voting
            #pred = ensemble_predictions(preds, pydantic_models[category])
            example_values_string = '\n'.join([json.dumps(p.dict()) for p in preds])
            # Merge by LLM
            merge_response = merge_chain.run(input=input_text, extracted_values=example_values_string, schema=convert_to_json_schema(pydantic_models[category]))
            try:
                pred = pydantic_models[category](**json.loads(merge_response))
                if verbose:
                    print(pred)
            except ValidationError as valError:
                if with_validation_error_handling:
                    pred, merge_response = validation_error_handling(category, input_text, merge_response, valError)
                else:
                    print('Validation Error: {}'.format(valError))
            except JSONDecodeError as e:
                print('Error: {}'.format(e))

        return pred

    # select_and_run_llm('Safety Mask', 'Reusable Fashion Protective Sponge Face Masks Breathable Anti-Dust Elastic Ear Loop Mask, 3 Black and 3 Grey')

    with get_openai_callback() as cb:
        # Mave dataset has multiple sources. We need to iterate over all of them.
        if dataset == 'mave':
            part_predictions = {}
            parts = ['title', 'description', 'features', 'attributes']
            # Run LLM for each part of the product
            for part in parts:
                example_part = part if part != 'title' else 'input'
                example_part = example_part if example_part != 'attributes' else 'rest'
                preds = [select_and_run_llm(example['category'], example[example_part], part=part,
                                            with_containment=with_containment,
                                            with_validation_error_handling=with_validation_error_handling,
                                            schema_type=schema_type) for example in
                         tqdm(task_dict['examples'], disable=not verbose)]
                part_predictions[example_part] = preds

            # Combine all predictions using a majority vote
            preds = [fuse_models(pydantic_models[example['category']], pred_title, pred_description, pred_features,
                                 pred_rest)
                     for example, pred_title, pred_description, pred_features, pred_rest
                     in zip(task_dict['examples'], part_predictions['input'], part_predictions['description'],
                            part_predictions['features'], part_predictions['rest'])]

        # Mave dataset has multiple sources. We need to iterate over all of them.
        elif dataset == 'mave_v2':
            hashed_predictions = {category: {} for category in
                                  task_dict['known_attributes']}  # Cache predictions per category
            preds = []
            # Run LLM for each part of the product
            for example in tqdm(task_dict['examples'], disable=not verbose):
                example_preds = []
                hashed_input = hash(example['input'])
                if hashed_input in hashed_predictions[example['category']]:
                    example_preds.append(hashed_predictions[example['category']][hashed_input])
                    print("Cached")
                    continue
                title_prediction = select_and_run_llm(example['category'], example['input'], part='title',
                                                      with_containment=with_containment,
                                                      with_validation_error_handling=with_validation_error_handling,
                                                      schema_type=schema_type)
                hashed_predictions[example['category']][hashed_input] = title_prediction
                example_preds.append(title_prediction)

                for description in example['paragraphs']:
                    hashed_description = hash(description)
                    if hashed_description in hashed_predictions[example['category']]:
                        example_preds.append(hashed_predictions[example['category']][hashed_description])
                        print("Cached")
                        continue

                    description_prediction = select_and_run_llm(example['category'], description, part='description',
                                                                with_containment=with_containment,
                                                                with_validation_error_handling=with_validation_error_handling,
                                                                schema_type=schema_type)
                    hashed_predictions[example['category']][hashed_description] = description_prediction
                    example_preds.append(description_prediction)

                pred = fuse_models(pydantic_models[example['category']], *example_preds)
                if verbose:
                    print(pred)
                preds.append(pred)

        else:
            preds = [select_and_run_llm(example['category'], example['input'], with_containment=with_containment,
                                        with_validation_error_handling=with_validation_error_handling,
                                        schema_type=schema_type) for example in
                     tqdm(task_dict['examples'], disable=not verbose)]
        task_dict['total_tokens'] = cb.total_tokens
        print(f"Total Tokens: {cb.total_tokens}")

    # Calculate recall, precision and f1
    task_dict['results'] = evaluate_predictions(preds, task_dict)
    # Save populated task and make sure that it is saved in the correct folder!
    save_populated_task(task_dict['task_name'], task_dict)


if __name__ == '__main__':
    load_dotenv()
    random.seed(42)
    main()
