import random

import click
from dotenv import load_dotenv

from pieutils.preprocessing import preprocess_oa_mine, preprocess_ae_110k, \
    convert_to_mave_dataset, reduce_training_set_size, convert_to_open_tag_format


@click.command()
@click.option('--dataset', default='oa-mine', help='Dataset name')
def main(dataset):
    """Preprocess datasets."""
    if dataset == 'oa-mine':
        preprocess_oa_mine()
    elif dataset == 'ae-110k':
        preprocess_ae_110k()
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

    for percentage in [0.2, 1.0]:
        reduce_training_set_size(dataset, percentage)
        convert_to_mave_dataset(dataset, percentage=percentage, skip_test=True)
        convert_to_open_tag_format(dataset, percentage=0.2, skip_test=False)


if __name__ == '__main__':
    load_dotenv()
    random.seed(42)
    main()
