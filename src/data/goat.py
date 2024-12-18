import json
import random
import gzip
from typing import Dict, Union
from torch.utils.data import Dataset
from datasets import load_dataset


class GoatArithmeticDataset(Dataset):
    def __init__(self, filepath: str, num_examples: int) -> None:
        super().__init__()
        self.num_examples = num_examples
        with gzip.open(filepath, 'r') as f:
            self.dataset = json.load(f)
        # self.dataset = load_dataset('tiedong/goat', split=split)
        if num_examples != -1:
            # idx_list = random.sample(list(range(len(self.dataset))), num_examples)
            self.dataset = random.sample(self.dataset, num_examples)


    def __len__(self) -> int:
        return len(self.dataset)


    def __getitem__(self, idx: int) -> Dict[str, Union[str, float]]:
        example = self.dataset[idx]
        answer = example['output']
        target = example['answer']
        output = f'{answer}'
        if output[-1] != '.':
            output = f'{output}.'
        output = f'{output} The answer is {target}.'
        gold_answer = output.replace('The answer is', '####')
        if not ' R ' in target and target != 'undefined':
            target = float(target)

        return {
            'question': example['instruction'],
            'gold_answer': gold_answer,
            'output': output,
            'target': target,
        }
