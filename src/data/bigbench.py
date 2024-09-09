import json
from typing import Any
from torch.utils.data import Dataset, ConcatDataset
from .constants import *


class BigBenchTask(Dataset):
    def __init__(self, filepath: str, data_mode: str, few_shot_input: bool) -> None:
        assert not few_shot_input, f'few_shot_input is not supported'

        super().__init__()
        self.data_mode = data_mode
        self.few_shot_input = few_shot_input
        self.input_template = GSM8K_INPUT_TEMPLATES[self.data_mode]
        self.prompt = GSM8K_PROMPTS[self.data_mode]
        self.prompt_delim = GSM8K_PROMPT_DELIMS[self.data_mode]

        with open(filepath, 'r') as f:
            self.examples = json.load(f)['examples']


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, index: int) -> Any:
        example = self.examples[index]
        question = example['input']
        input_text = self.input_template.format(question)
        return {
            'input': input_text,
            'question': question,
            'output': '',
            'target': float(example['target']),
            'data_mode': self.data_mode,
        }


class BigBenchDataset(ConcatDataset):
    def __init__(self, filepaths: str, data_mode: str, few_shot_input: bool) -> None:
        datasets = [BigBenchTask(filepath, data_mode, few_shot_input) for filepath in filepaths]
        super(BigBenchDataset, self).__init__(datasets)
