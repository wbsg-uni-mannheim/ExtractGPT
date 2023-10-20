from pieutils.preprocessing import load_known_attribute_values

datasets = ['OA-Mine', 'AE-110k']
no_example_values = [3, 5, 10]
train_percentage = 0.2

for dataset in datasets:
    for n_examples in no_example_values:
        known_attribute_values = load_known_attribute_values(dataset, n_examples=n_examples,
                                                     train_percentage=train_percentage)
        attribute_values_test = load_known_attribute_values(dataset, n_examples=9999,
                                                     test_set=True)

        count_known_attribute_values = 0
        for product_category in known_attribute_values:
            for attribute in known_attribute_values[product_category]:
                count_known_attribute_values += len(known_attribute_values[product_category][attribute])

        print(f"{count_known_attribute_values} \t {dataset} \t {n_examples} \t {train_percentage}")

        # Evaluate how many of the known attribute values are in the test set
        count_known_attribute_values_test = 0
        count_attribute_values_test = 0
        for product_category in attribute_values_test:
            for attribute in attribute_values_test[product_category]:
                for attribute_value in attribute_values_test[product_category][attribute]:
                    if attribute in known_attribute_values[product_category] and attribute_value in known_attribute_values[product_category][attribute]:
                        count_known_attribute_values_test += 1
                count_attribute_values_test += len(attribute_values_test[product_category][attribute])

        percentage_known_attribute_values_test = round(count_known_attribute_values_test / count_attribute_values_test, 2)
        print(f"{count_known_attribute_values} \t {percentage_known_attribute_values_test} \t {dataset} \t {n_examples} \t {train_percentage}")