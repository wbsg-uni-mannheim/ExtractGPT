import gzip
import json
import os
from typing import Optional

from pydantic import create_model, Field, ValidationError


def ensemble_predictions(preds, pydantic_model):
    # Use majority voting
    pred = {}
    attributes = pydantic_model.__fields__.keys()
    for attribute in attributes:
        # Get all predictions for the attribute
        attribute_preds = [pred.dict()[attribute] for pred in preds.values()
                           if pred is not None and attribute in pred.dict()]
        # Get the most common prediction
        if len(attribute_preds) == 0:
            pred[attribute] = 'n/a'
        else:
            pred[attribute] = max(attribute_preds, key=attribute_preds.count)

    try:
        pred = pydantic_model(**pred)
    except ValidationError as e:
        print(e)
        pred = None

    return pred


def save_meta_model(task_name, llm_used_for_creation, dataset_name, models_json):
    """Persist a meta model to a file."""
    meta_model_dir = 'prompts/meta_models'
    if not os.path.exists(meta_model_dir):
        os.makedirs(meta_model_dir)

    path = '{}/models_by_{}_{}.json'.format(meta_model_dir, task_name, llm_used_for_creation, dataset_name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(models_json, f, indent=4)


def prepare_example_task_prefixes(example, task_prefix):
    example['task_prefix'] = task_prefix.replace('[PLACEHOLDER]', example['attribute'])
    return example


def combine_example(example, pred, post_pred):
    """Format examples to save the predictions"""
    example['pred'] = pred
    example['post_pred'] = post_pred
    return example

def create_pydanctic_models_from_known_attributes(known_attributes):
    """Create Pydantic models for the known attributes."""
    pydantic_models = {}
    for category in known_attributes:
        # Define field specs:
        if category[-1] == 's':
            # If the category name ends with an 's', it is a plural category, e.g. 'customers'.
            fields_spec = {attribute: (f'The {attribute} of {category}.', Optional[str]) for
                           attribute in known_attributes[category]}
        else:
            fields_spec = {attribute: (f'The {attribute} of a {category}.', Optional[str]) for
                           attribute in known_attributes[category]}
        model_description = f'Relevant customer information about a {category}.'
        pydantic_models[category] = create_pydanctic_model(category, _model_description=model_description, **fields_spec)

    return pydantic_models


def create_pydanctic_model(_model_name, _model_description='', **fields):
    # Create a dictionary to hold the fields for the dynamic model
    model_fields = {}

    # Convert the field specifications to Pydantic field declarations
    for field_name, (field_description, field_type) in fields.items():
        model_fields[field_name] = (field_type, Field(description=field_description))

    # Use the create_model function to create a dynamic Pydantic model
    dynamic_model_class = create_model(_model_name, **model_fields)
    dynamic_model_class.__doc__ = _model_description

    return dynamic_model_class

def create_pydanctic_model_with_examples(_model_name, _model_description='', **fields):
    # Create a dictionary to hold the fields for the dynamic model
    model_fields = {}

    # Convert the field specifications to Pydantic field declarations
    for field_name, (field_description, field_type, field_examples) in fields.items():
        model_fields[field_name] = (field_type, Field(description=field_description, examples=field_examples))

    # Use the create_model function to create a dynamic Pydantic model
    dynamic_model_class = create_model(_model_name, **model_fields)
    dynamic_model_class.__doc__= _model_description

    return dynamic_model_class


def create_pydanctic_model_from_pydantic_meta_model(pydantic_meta_model, known_attribute_values=None):
    # Create a dictionary to hold the fields for the dynamic model
    meta_model = dict(pydantic_meta_model)
    model_name = meta_model['name']
    model_description = meta_model['description']
    model_fields = {}
    with_examples = False
    for field in meta_model['attributes']:
        attribute = dict(field)
        if 'examples' in attribute:
            if 'type' in attribute and attribute['type'] == 'integer':
                model_fields[attribute['name']] = (attribute['description'], Optional[int], attribute['examples'])
            else:
                if known_attribute_values is not None and attribute['name'] in known_attribute_values:
                    model_fields[attribute['name']] = (attribute['description'], Optional[str], known_attribute_values[attribute['name']])
                else:
                    model_fields[attribute['name']] = (attribute['description'], Optional[str], attribute['examples'])
            with_examples = True
        else:
            if 'type' in attribute and attribute['type'] == 'integer':
                model_fields[attribute['name']] = (attribute['description'], Optional[int])
            else:
                model_fields[attribute['name']] = (attribute['description'], Optional[str])

    # Convert the model specifications to Pydantic model
    if with_examples:
        dynamic_model_class = create_pydanctic_model_with_examples(model_name, _model_description=model_description, **model_fields)
    else:
        dynamic_model_class = create_pydanctic_model(model_name, _model_description=model_description, **model_fields)
    return dynamic_model_class


def create_tabular_pydanctic_model_from_pydantic_meta_model(pydantic_meta_model):
    """Create a tabular Pydantic model from a Pydantic meta model."""
    dynamic_model_class = create_pydanctic_model_from_pydantic_meta_model(pydantic_meta_model)

    # Make this a tabular model
    model_name = f'{dynamic_model_class.__name__}s'
    model_description = f'A tabular representation of {dynamic_model_class.__name__}s.'
    model_fields = {'records': (f'A list of {model_name}.', list[dynamic_model_class])}
    tabular_dynamic_model_class = create_pydanctic_model(model_name, _model_description=model_description, **model_fields)

    return tabular_dynamic_model_class



def create_dict_of_pydanctic_product(pydantic_product):
    # Create a dictionary to hold the fields for the dynamic model
    schema = pydantic_product.schema()
    pydantic_model_schema = {'name': schema['title'], 'description': schema['description'], 'attributes': []}
    for field in schema['properties']:
        if 'examples' in schema['properties'][field] and schema['properties'][field]['examples'] is not None and len(schema['properties'][field]['examples']) > 0:
            pydantic_model_schema['attributes'].append({'name': field, 'description': schema['properties'][field]['description'], 'examples': schema['properties'][field]['examples']})
        else:
            pydantic_model_schema['attributes'].append({'name': field, 'description': schema['properties'][field]['description']})

    return pydantic_model_schema


def extract_attribute(answer, attribute):
    """Extract an attribute value for the open extraction."""
    if '\n' in answer:
        for part in answer.split('\n'):
            if attribute in part:
                if ':' in part:
                    return part.split(':')[1].strip()
    return "n/a"


def save_populated_task(task, task_dict):
    """Save the populated task to a file."""
    model_name = task_dict['model'].replace(':', '_').replace('/', '_')
    result_file = 'task_run_chat{}_{}_{}_{}.gz'.format(task, task_dict['dataset_name'], model_name, task_dict['timestamp'])
    path_to_result_file = 'prompts/runs/{}'.format(result_file)

    # Check if the path to the result file exists
    if not os.path.exists('prompts/runs'):
        os.makedirs('prompts/runs')

    with gzip.open(path_to_result_file, 'wt', encoding='utf-8') as fp:
        json.dump(task_dict, fp, indent=4, ensure_ascii=False)


def update_handlabelled_testset(source_path, target_path):
    loaded_dicts = []

    with open(source_path, 'r') as f:
        joint_lines = ''.join([line for line in f])
        json_dicts = joint_lines.split('}{')
        for json_dict in json_dicts:
            if json_dict[0] != '{':
                json_dict = '{' + json_dict
            if json_dict[-1] != '}':
                json_dict = json_dict + '}'
            loaded_dict = json.loads(json_dict)
            loaded_dicts.append(loaded_dict)

    with open(target_path, 'w+', encoding='utf-8') as f:
        for record in loaded_dicts:
            f.write('{}\n'.format(json.dumps(record)))


def convert_to_json_schema(pydantic_model, replace_description=True, schema_type='json_schema', properties=None, replacement_attribute_values=None):
    """Convert a Pydantic model to a JSON schema.
        schema_type: Schema to use - json_schema, json_schema_no_type or compact
        """
    # Get the schema of the Pydantic model if schema_properties is not provided
    schema = pydantic_model.schema()
    if properties is None:
        properties = list(schema['properties'].items())

    if schema_type in ("json_schema", "json_schema_no_type"):
        parameters = {'type': 'object', 'properties': {}}

        for property_name, property_schema in properties:
            if property_name in ("title"):
                continue
            if schema_type == "json_schema_no_type":
                parameters['properties'][property_name] = {k: v for k, v in property_schema.items() if k not in ("title", "type")}
            else:
                parameters['properties'][property_name] = {k: v for k, v in property_schema.items() if k not in ("title")}

        if replacement_attribute_values is not None:
            for attribute, values in replacement_attribute_values.items():
                parameters['properties'][attribute]['examples'] = list(values)

        # Add required fields
        if 'required' in schema:
            parameters['required'] = schema['required']

        # Add additional definitions
        if 'definitions' in schema:
            parameters['definitions'] = {}
            for definition_name, definition_schema in schema['definitions'].items():
                parameters['definitions'][definition_name] = {'description': definition_schema['description'],
                                                              'type': 'object', 'properties': {}}

                for property_name, property_schema in definition_schema['properties'].items():
                    if property_name in ("title"):
                        continue
                    if schema_type == "json_schema_no_type":
                        parameters['definitions'][definition_name]['properties'][property_name] = {k: v for k, v in property_schema.items()
                                                                                                    if k not in ("title", "type")}
                    else:
                        parameters['definitions'][definition_name]['properties'][property_name] = {k: v for k, v in property_schema.items()
                                                                                if k not in ("title")}
                # Add required fields
                if 'required' in definition_schema:
                    parameters['definitions'][definition_name]['required'] = definition_schema['required']

        # Replace description
        if replace_description:
            schema[
                "description"] = f"Correctly extracted `{pydantic_model.__name__}` with all the required parameters with correct types."

        convert_schema = {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": parameters,
        }
    elif schema_type == "compact":
        """Compact version of the JSON schema"""
        convert_schema = { }
        for property_name, property_schema in properties:
            if property_name in ("title"):
                continue
            if 'description' in property_schema:
                if 'examples' in property_schema:
                    if replacement_attribute_values is not None and property_name in replacement_attribute_values:
                        convert_schema[property_name] = f"{property_schema['description']} - Examples: {', '.join(replacement_attribute_values[property_name])}"
                    else:
                        convert_schema[property_name] = f"{property_schema['description']} - Examples: {', '.join(property_schema['examples'])}"
                else:
                    convert_schema[property_name] = property_schema['description']

    elif schema_type == "textual":
        """Textual version of the JSON schema"""
        introduction = f"A product offer from the product category {schema['title']} has the following attributes: {', '.join(schema['properties'].keys())}."
        attributes = []
        for property_name, property_schema in properties:
            if property_name in ("title"):
                continue
            if 'description' in property_schema:
                if 'examples' in property_schema:
                    if replacement_attribute_values is not None and property_name in replacement_attribute_values:
                        attribute_text = f"The attribute {property_name} is defined as: {property_schema['description']} Known attribute values are {', '.join(replacement_attribute_values[property_name])}."
                    else:
                        attribute_text = f"The attribute {property_name} is defined as: {property_schema['description']} Known attribute values are {', '.join(property_schema['examples'])}."

                else:
                    attribute_text = f"The attribute {property_name} is defined as: {property_schema['description']}"
                attributes.append(attribute_text)
        convert_schema = introduction + '\n ' + '\n'.join(attributes)
    else:
        raise ValueError(f"Unknown schema type: {schema_type}")

    return convert_schema


def parse_llm_response_to_json(response):
    """Convert a response to a JSON object. This method is implemented for LLama 7B"""
    """Example response:  Stove
                            Fuel: Gas
                            Stove Type: Portable
                            Brand Name: Hewolf
                            Material: Stainless Steel
                            Applicable Seasoning Type: n/a
                            Model: n/a """

    # Split Response by new line
    response_parts = response.split('\n')
    response_parts = [part.strip() for part in response_parts if part.strip() != '']

    # Try to parse line as a JSON object
    for response_part in response_parts:
        parsed_response_part = response_part.replace('Human:', '').replace('AI:', '').replace('System:', '')\
            .replace('Algorithm:', '').replace('"s', "'s").replace("'", '"').strip()
        try:
            response_dict = json.loads(parsed_response_part)
            return response_dict
        except:
            print(parsed_response_part)
            pass

    # Try to parse line as a JSON object - 2nd attempt
    response_dict = {}
    # Parse response as a dictionary
    for response_part in response_parts:
        parsed_response_part = response_part.replace('Human:', '')\
            .replace('AI:', '').replace('System:', '').strip()
        if ':' in parsed_response_part:
            parsed_response_part = parsed_response_part.split(':')
            if len(parsed_response_part) == 2 and type(parsed_response_part[0]) == str and type(parsed_response_part[1]) == str:
                response_dict[parsed_response_part[0].replace('"', '').replace("'", "").strip()] = parsed_response_part[1].replace('"', '').replace("'", "").strip().rstrip(',!')

    return response_dict
