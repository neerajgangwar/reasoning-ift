import json
import re
import random
from typing import Dict, Any
from xml.etree import ElementTree as ET
from torch.utils.data import Dataset
from datasets import load_dataset
from .constants import *


class MultiArithDataset(Dataset):
    def __init__(self, split: str, data_mode: str, few_shot_input: bool) -> None:
        self.split = split
        self.dataset = load_dataset('ChilleD/MultiArith', split=split)

        self.data_mode = data_mode
        self.few_shot_input = few_shot_input
        self.input_template = GSM8K_INPUT_TEMPLATES[self.data_mode]
        self.prompt = GSM8K_PROMPTS[self.data_mode]
        self.prompt_delim = GSM8K_PROMPT_DELIMS[self.data_mode]


    def __len__(self) -> int:
        return len(self.dataset)


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.dataset[idx]
        input_text = self.input_template.format(example['question'])
        if self.few_shot_input:
            input_text = f'{self.prompt}{self.prompt_delim}{input_text}'
        return {
            'input': input_text,
            'question': example['question'],
            'output': '',
            'target': float(example['final_ans']),
            'data_mode': self.data_mode,
        }


class ASDivDataset(Dataset):
    def __init__(self, filepath: str, data_mode: str, few_shot_input: bool) -> None:
        self.filepath = filepath
        self.data_mode = data_mode
        self.few_shot_input = few_shot_input
        self.input_template = GSM8K_INPUT_TEMPLATES[self.data_mode]
        self.prompt = GSM8K_PROMPTS[self.data_mode]
        self.prompt_delim = GSM8K_PROMPT_DELIMS[self.data_mode]

        dataset = ET.parse(self.filepath)
        self.examples = []
        problem_set = dataset.findall('ProblemSet')
        assert len(problem_set) == 1
        for problem in problem_set[0].findall('Problem'):
            body = problem.findall('Body')
            question = problem.findall('Question')
            answer = problem.findall('Answer')
            assert len(body) == len(question) == len(answer) == 1
            self.examples.append({
                'body': body[0].text,
                'question': question[0].text,
                'answer': answer[0].text,
            })



    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        question = f'{example["body"]} {example["question"]}'
        target = re.findall(NUMBER_REGEX, example['answer'])
        if len(target) == 1:
            target = float(target[0])
        else:
            target = target

        input_text = self.input_template.format(question)
        if self.few_shot_input:
            input_text = f'{self.prompt}{self.prompt_delim}{input_text}'
        return {
            'input': input_text,
            'question': question,
            'output': '',
            'target': target,
            'data_mode': self.data_mode,
        }


class SVAMPDataset(Dataset):
    def __init__(self, filepath: str, data_mode: str, few_shot_input: bool) -> None:
        self.filepath = filepath
        self.data_mode = data_mode
        self.few_shot_input = few_shot_input
        self.input_template = GSM8K_INPUT_TEMPLATES[self.data_mode]
        self.prompt = GSM8K_PROMPTS[self.data_mode]
        self.prompt_delim = GSM8K_PROMPT_DELIMS[self.data_mode]

        with open(self.filepath) as f:
            self.examples = json.load(f)


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        body = example['Body'] if example['Body'].endswith('.') else f'{example["Body"]}.'
        question = f'{body} {example["Question"]}'
        input_text = self.input_template.format(question)
        if self.few_shot_input:
            input_text = f'{self.prompt}{self.prompt_delim}{input_text}'
        return {
            'input': input_text,
            'question': question,
            'output': '',
            'target': float(example['Answer']),
            'data_mode': self.data_mode,
        }


class GSMICDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        question_key: str,
        data_mode: str,
        few_shot_input: bool,
        data_seed: int,
        num_examples: int,
    ) -> None:
        self.question_key = question_key
        self.data_mode = data_mode
        self.few_shot_input = few_shot_input
        self.input_template = GSM8K_INPUT_TEMPLATES[self.data_mode]
        self.prompt = GSM8K_PROMPTS[self.data_mode]
        self.prompt_delim = GSM8K_PROMPT_DELIMS[self.data_mode]

        with open(filepath, 'r') as f:
            seen_ques = set()
            examples = [seen_ques.add(e[question_key]) or e for e in json.load(f) if e[question_key] not in seen_ques]
            if num_examples == -1:
                self.examples = examples
            else:
                assert len(examples) >= num_examples
                random.seed(data_seed)
                self.examples = random.sample(examples, num_examples)


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, index: int) -> Any:
        example = self.examples[index]
        question = example[self.question_key]
        if self.few_shot_input:
            input_text = f'{self.prompt}{self.prompt_delim}{self.input_template.format(question)}'
        else:
            input_text = self.input_template.format(question)

        return {
            'input': input_text,
            'question': question,
            'output': '',
            'target': float(example['answer'].replace(',', '')),
            'data_mode': self.data_mode,
        }


class GSMSysDataset(Dataset):
    def __init__(self, filepath: str, data_mode: str, few_shot_input: bool) -> None:
        self.data_mode = data_mode
        self.few_shot_input = few_shot_input
        self.input_template = GSM8K_INPUT_TEMPLATES[self.data_mode]
        self.prompt = GSM8K_PROMPTS[self.data_mode]
        self.prompt_delim = GSM8K_PROMPT_DELIMS[self.data_mode]

        with open(filepath, 'r') as f:
            self.examples = json.load(f)


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, index: int) -> Any:
        example = self.examples[index]
        question = example['question']
        if self.few_shot_input:
            input_text = f'{self.prompt}{self.prompt_delim}{self.input_template.format(question)}'
        else:
            input_text = self.input_template.format(question)

        return {
            'input': input_text,
            'question': question,
            'output': '',
            'target': float(example['label'].replace(',', '')),
            'data_mode': self.data_mode,
        }
