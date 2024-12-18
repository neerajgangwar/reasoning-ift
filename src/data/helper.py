import random
from typing import Any
from torch.utils.data import Dataset
from src.utils import chunks


# class PromptHelper:
#     def __init__(self, prompt: str, input_template: str) -> None:
#         self.prompt = prompt
#         self.input_template = input_template


#     def few_shot_input(self, question: str) -> str:
#         return f'{self.prompt}\n{self.input_template.format(question)}'


#     def zero_shot_input(self, question: str) -> str:
#         return self.input_template.format(question)



class BatchedDataset(Dataset):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool) -> None:
        dataset_idxs = list(range(len(dataset)))
        if shuffle:
            random.shuffle(dataset_idxs)

        examples = [dataset[idx] for idx in dataset_idxs]
        self.batches = list(chunks(examples, batch_size))


    def __len__(self) -> int:
        return len(self.batches)


    def __getitem__(self, index: int) -> Any:
        return self.batches[index]
