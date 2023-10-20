import random

import click
from dotenv import load_dotenv

from pieutils.search import load_train_for_vector_store


@click.command()
@click.option('--dataset', default='mave', help='Dataset name')
def main(dataset):
    # Load training data like for the vector stores
    train_records = load_train_for_vector_store(dataset, categories=None)




if __name__ == '__main__':
    load_dotenv()
    random.seed(42)
    main()