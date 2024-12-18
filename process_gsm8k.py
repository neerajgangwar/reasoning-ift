import os
import json
import hydra
import random
import logging
from pathlib import Path
from omegaconf import DictConfig
from datasets import load_dataset


logger = logging.getLogger(__name__)


def write_to_file(filepath, dataset):
    assert not os.path.exists(filepath), f'{filepath} already exists'
    assert filepath.endswith('.json')
    with open(filepath, 'w') as f:
        examples = [{'question': e['question'], 'answer': e['answer']} for e in dataset]
        json.dump(examples, f, indent=4)


@hydra.main(version_base=None, config_path='src/conf/preprocess', config_name='gsm8k')
def main(conf: DictConfig):
    logging.info(f'conf: {conf}')
    if conf.seed is not None:
        random.seed(conf.seed)

    dataset = load_dataset('gsm8k', 'main')

    # Train and dev sets
    total_train_examples = len(dataset['train'])
    all_indices = list(range(total_train_examples))
    val_indices = random.sample(all_indices, conf.n_val_examples)
    train_indices = set(all_indices) - set(val_indices)
    train_examples = dataset['train'].select(train_indices)
    val_examples = dataset['train'].select(val_indices)

    # Test set
    test_examples = dataset['test']

    Path(conf.save_dir).mkdir(parents=True, exist_ok=True)
    write_to_file(os.path.join(conf.save_dir, 'train.json'), train_examples)
    write_to_file(os.path.join(conf.save_dir, 'val.json'), val_examples)
    write_to_file(os.path.join(conf.save_dir, 'test.json'), test_examples)


if __name__ == '__main__':
    main()
