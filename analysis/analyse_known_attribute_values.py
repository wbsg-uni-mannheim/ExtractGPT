from pieutils.preprocessing import load_known_attribute_values


for dataset in ['oa-mine', 'ae-110k']:
    for train_percentage in [0.2, 1.0]:
        known_attribute_values = load_known_attribute_values(dataset, n_examples=10,
                                                             train_percentage=train_percentage)
        print('Dataset:', dataset)
        print('Train percentage:', train_percentage)
        # Count the number of examples for each attribute
        total_number_of_examples = 0
        values_per_attribute = []
        for category, attribute in known_attribute_values.items():
            for values in attribute.values():
                #print(attribute, len(values))
                values_per_attribute.append(len(values))
                total_number_of_examples += len(values)

        print('Average number of values per attribute:', sum(values_per_attribute) / len(values_per_attribute))
        print('Median number of values per attribute:', sorted(values_per_attribute)[len(values_per_attribute) // 2])
        print('Total number of examples:', total_number_of_examples)
        print('------------------------------------------')
