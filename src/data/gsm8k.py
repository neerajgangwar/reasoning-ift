import re
import json
import random
import gzip
from abc import ABC, abstractmethod
from typing import Dict, Union
from torch.utils.data import Dataset
from .constants import GSM8K_PROMPTS, GSM8K_INPUT_TEMPLATES, GSM8K_PROMPT_DELIMS


def extract_answer(answer_str: str) -> float:
    target = answer_str.split('####')[-1].strip()
    assert answer_str.endswith(f'#### {target}'), f'answer: {answer_str}, target: {target}'
    return float(target.replace(',', ''))


class GSM8kDataset(Dataset):
    def __init__(self, filepath: str, num_examples: int=-1):
        super(GSM8kDataset, self).__init__()

        with gzip.open(filepath, 'rt') as f:
            self.examples = json.load(f)
            if num_examples != -1:
                random.shuffle(self.examples)
                self.examples = self.examples[:num_examples]


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, index: int) -> Dict[str, Union[str, float]]:
        example = self.examples[index]
        target = extract_answer(example['answer'])
        output = example['answer'].replace('####', 'The answer is')
        output = re.sub(f'<<.*?>>', '', output)

        return {
            'question': example['question'],
            'gold_answer': example['answer'],
            'output': output,
            'target': target,
        }


class GSM8kGeneratedDataset(Dataset):
    def __init__(self, filepath: str, num_examples: int=-1):
        super(GSM8kGeneratedDataset, self).__init__()

        with gzip.open(filepath, 'rt') as f:
            self.examples = json.load(f)
            if num_examples != -1:
                random.shuffle(self.examples)
                self.examples = self.examples[:num_examples]


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, index: int) -> Dict[str, Union[str, float]]:
        example = self.examples[index]
        target = extract_answer(example['gold_answer'])
        return {
            'question': example['question'],
            'output': example['gen_answer'],
            'gold_answer': example['gold_answer'],
            'target': target,
        }


class ZeroShotGSM8kDataset(Dataset):
    def __init__(self, dataset: Union[GSM8kDataset, GSM8kGeneratedDataset], data_mode: str) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_mode = data_mode
        self.input_template = GSM8K_INPUT_TEMPLATES[self.data_mode]


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index: int) -> Dict[str, Union[str, float]]:
        example = self.dataset[index]
        input_text = self.input_template.format(example['question'])
        return {
            'input': input_text,
            'output': example['output'],
            'target': example['target'],
            'question': example['question'],
            'gold_answer': example['gold_answer'],
            'data_mode': self.data_mode,
        }


class FewShotGSM8kDataset(ABC, Dataset):
    def __init__(self, dataset: Union[GSM8kDataset, GSM8kGeneratedDataset], data_mode: str) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_mode = data_mode
        self.input_template = GSM8K_INPUT_TEMPLATES[self.data_mode]
        self.prompt_delim = GSM8K_PROMPT_DELIMS[self.data_mode]


    @abstractmethod
    def get_prompt(self) -> str:
        pass

    def __len__(self) -> int:
        return len(self.dataset)


    def __getitem__(self, index: int) -> Dict[str, Union[str, float]]:
        example = self.dataset[index]
        prompt = self.get_prompt().strip()
        question = self.input_template.format(example['question'])
        input_text = f'{prompt}{self.prompt_delim}{question}'

        return {
            'input': input_text,
            'output': example['output'],
            'target': example['target'],
            'question': example['question'],
            'gold_answer': example['gold_answer'],
            'data_mode': self.data_mode,
        }


class StaticFewShotGSM8kDataset(FewShotGSM8kDataset):
    def __init__(self, dataset: Union[GSM8kDataset, GSM8kGeneratedDataset], data_mode: str) -> None:
        self.prompt = GSM8K_PROMPTS[data_mode]
        super().__init__(dataset=dataset, data_mode=data_mode)


    def get_prompt(self) -> str:
        return self.prompt


class DynamicFewShotGSM8kDataset(FewShotGSM8kDataset):
    def __init__(self, dataset: GSM8kGeneratedDataset, prompt_source: Union[GSM8kDataset, GSM8kGeneratedDataset], data_mode: str, prompt_size: int) -> None:
        super().__init__(dataset, data_mode)
        self.data_mode = data_mode
        self.prompt_size = prompt_size
        self.prompt_examples = []
        seen_questions = set()
        for example in prompt_source:
            if example['question'] in seen_questions:
                continue
            self.prompt_examples.append(example)
            seen_questions.add(example['question'])


    def get_prompt(self) -> str:
        idx_list = random.sample(range(len(self.prompt_examples)), self.prompt_size)
        examples = []
        for idx in idx_list:
            question = self.prompt_examples[idx]['question']
            answer = self.prompt_examples[idx]['output']
            prompt = self.input_template.format(question)
            prompt = f'{prompt}{answer}'
            examples.append(prompt)

        return self.prompt_delim.join(examples)
